import json, uuid, boto3, os, traceback, requests, math, operator, time
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated, Any, Dict, Optional, List
from typing import Literal
from supabase import create_client, Client
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed


#############################################################################
# SETUP
#############################################################################
# region

EXPECTED_API_KEY = os.environ.get("AGENT_API_KEY")

model = ChatOpenAI(model="gpt-4o", temperature=0)
openai_client = OpenAI()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(url, key)

lambda_client = boto3.client("lambda", region_name=os.environ.get("AWS_REGION", "us-east-1"))

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    itinerary: Optional[Dict[str, Any]]
    itinerary_saved: bool

# endregion

#############################################################################
# SYSTEM PROMPT
#############################################################################
# region

SYSTEM_PROMPT = (
    "You are an expert travel assistant helping users discover attractions, plan trips, "
    "and explore destinations around the world. Your goal is to provide genuinely helpful, "
    "accurate, and personalized travel guidance and help users build an itinerary.\n\n"

    "You have access to the following tools:\n"
    "- resolve_attractions_tool: Resolves attractions to real database IDs, checking for duplicates and creating new records as needed. Call this before saving to the itinerary.\n"
    "- save_itinerary_tool: Saves updates to the itinerary.\n\n"

    "Discovery:\n"
    "- Have a natural conversation to understand the user's destination and preferences.\n"
    "- Suggest attractions freely from your own knowledge — do NOT call any tools during discovery.\n"
    "- When presenting attractions, show a maximum of 10 and select the most relevant based on the user's preferences.\n\n"

    "Saving to Itinerary:\n"
    "- When the user is ready to save attractions, call resolve_attractions_tool with ALL attractions they want to add.\n"
    "- When the user selects attractions by number from your list, include all selected attractions in a single call.\n"
    "- Each attraction must use the nearest large, well-known city name — never an island, region, or the attraction name itself.\n"
    "- After resolve_attractions_tool returns, use the exact attraction_id values it provides in save_itinerary_tool.\n"
    "- Custom events (e.g. breakfast, travel, swimming) can be added directly to days without calling resolve_attractions_tool. "
        "Use a large random negative integer for attractionId (e.g. -1774315440974) and set attractionName to the event name. "
        "Use different attraction_ids for each custom event (even if there is another event just like it)\n"
    "- Always call resolve_attractions_tool BEFORE save_itinerary_tool.\n"
    "- Always call save_itinerary_tool BEFORE responding to the user.\n"
    "- Always confirm the saved itinerary to the user after saving.\n"
    "- When displaying times to the user, always use 12-hour AM/PM format (e.g. 9:00 AM, 2:30 PM).\n\n"

    "Itinerary Editing:\n"
    "- If an itinerary is provided in your context, the user may want to edit it.\n"
    "- When adding attractions, always call resolve_attractions_tool first.\n"
    "- When making changes that don't modify the days or unscheduled field (dates, trip name, notes), call save_itinerary_tool directly.\n"
    "- Never call save_itinerary_tool unless the user has explicitly requested a change.\n"
    "- Never save a stop with a null or missing attractionId.\n\n"

    "save_itinerary_tool Rules:\n"
    "- Only pass fields that are actually changing. Omit everything else — unchanged fields will be preserved automatically.\n"
    "- trip_name: descriptive name based on destinations if not already set.\n"
    "- start_date / end_date: use nearest future dates if not specified.\n"
    "- days: only pass if modifying the schedule. Must include ALL days even if only one changed. "
        "Pass an empty array [] if all days are being cleared. Each stop: {attractionId, startTime (HH:MM), durationMinutes}.\n"
    "- unscheduled: only pass if explicitly adding or removing items. "
        "Pass an empty array [] if all items are being removed. "
        "When scheduling an unscheduled attraction onto a day, pass both the updated days AND updated unscheduled with that attraction removed.\n"
    "- When adding or removing days, always update start_date, end_date, and place together.\n"
    "- place: list of cities the user explicitly mentioned as destinations, each with placeName (str) and placeCountry (str). Do not add nearby cities even if an attraction is technically located there. Do not include placeId — it is resolved automatically.\n"
    "- If there are more attractions than can fit in the scheduled days, place the remaining ones in unscheduled rather than dropping them.\n"
)

# endregion

#############################################################################
# HELPERS
#############################################################################
# region

def ensure_place_exists(city: str, state: Optional[str], country: str) -> Optional[int]:
    """Check if place exists, add it if not, return place_id"""
    print(f"[ensure_place_exists] Checking: city={city}, state={state}, country={country}")
    
    query = supabase.table("place").select("place_id")
    query = query.ilike("place_city", city)
    if state:
        query = query.ilike("place_stateprovince", state)
    query = query.ilike("place_countryregion", country)
    
    result = query.execute()
    
    if result.data:
        place_id = result.data[0]["place_id"]
        print(f"[ensure_place_exists] Found — place_id={place_id}")
        return place_id
    
    print(f"[ensure_place_exists] Not found — adding")
    coords = geocode(f"{city}, {state}, {country}" if state else f"{city}, {country}")
    latitude, longitude = coords if coords else (None, None)
    
    try:
        result = supabase.table("place").insert({
            "place_type": ["city"],
            "place_city": city,
            "place_stateprovince": state,
            "place_countryregion": country,
            "place_latitude": latitude,
            "place_longitude": longitude,
        }).execute()
        place_id = result.data[0]["place_id"]
        print(f"[ensure_place_exists] Added — place_id={place_id}")
        return place_id
    except Exception as e:
        print(f"[ensure_place_exists] Error: {e}")
        return None

def add_attraction(place_id: int, a: dict, embedding: list, canonical_id: int) -> Optional[int]:
    """Insert a new attraction and return its attraction_id. 
    Assumes duplicate check has already been done by the caller."""
    name = a.get("name")
    city = a.get("city")
    state = a.get("state")
    country = a.get("country")
    description = a.get("description", "")

    print(f"[add_attraction] Adding: {name}")

    coords = geocode(f"{name}, {city}, {country}")
    time.sleep(1)
    attraction_lat = coords[0] if coords else a.get("latitude")
    attraction_lng = coords[1] if coords else a.get("longitude")

    place_result = supabase.table("place").select("place_latitude, place_longitude").eq("place_id", place_id).execute()
    place_lat = place_result.data[0].get("place_latitude") if place_result.data else None
    place_lng = place_result.data[0].get("place_longitude") if place_result.data else None

    row = {
        "place_id": place_id,
        "attraction_name": name,
        "attraction_summary": description,
        "attraction_city": city,
        "attraction_stateprovince": state,
        "attraction_countryregion": country,
        "attraction_pricelevel": a.get("price_level"),
        "attraction_vibe": a.get("vibe", []),
        "attraction_latitude": attraction_lat,
        "attraction_longitude": attraction_lng,
        "attraction_distancefromplace": haversine_distance(place_lat, place_lng, attraction_lat, attraction_lng),
        "attraction_popularityscore": a.get("popularity_score"),
        "attraction_rawdata": a.get("raw_data", {}),
        "attraction_lastrefreshed": datetime.now(timezone.utc).isoformat(),
        "attraction_embedding": embedding,
        "attraction_reviewssummary": "Reviews summary not available",
        "attraction_credibilitytier": 3,
        "canonical_id": canonical_id,
    }

    try:
        result = supabase.table("attraction").insert(row).execute()
        attraction_id = result.data[0]["attraction_id"]
        print(f"[add_attraction] Added — attraction_id={attraction_id}")

        try:
            lambda_client.invoke(
                FunctionName=os.getenv("IMAGE_LAMBDA_NAME"),
                InvocationType="Event",
                Payload=json.dumps({"attractions": [{
                    "attraction_id": attraction_id,
                    "place_id": place_id,
                    "name": name,
                    "city": city,
                    "country": country,
                }]})
            )
        except Exception as e:
            print(f"[add_attraction] Image Lambda error: {e}")

        return attraction_id
    except Exception as e:
        if '23505' in str(e):
            print(f"[add_attraction] Duplicate detected — fetching existing record for '{name}', place_id={place_id}")
            existing = supabase.table("attraction").select("attraction_id").eq("attraction_name", name).eq("place_id", place_id).execute()
            if existing.data:
                return existing.data[0]["attraction_id"]
        print(f"[add_attraction] Error: {e}")
        return None

def search_single_attraction(name, country, embedding, match_count=5):
    client = create_client(url, key)
    result = client.rpc("match_attractions", {
        "query_name": name,
        "query_embedding": embedding,
        "query_country": country,
        "match_count": match_count
    }).execute()
    return {"input_name": name, "candidates": result.data or []}

def db_row_to_message(row):
    """Determine message type based on role; convert row to message object"""
    role = row["role"]
    content = row["content"]

    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "tool":
        return ToolMessage(content=content)
    else:
        raise ValueError(f"Unknown role: {role}")

def update_session_summary(session_id: str, recent_messages: list, current_summary: str):
    """Send messages and current summary to LLM, get an updated summary, and store it in the DB"""
    conversation_text = "\n".join(
        f"{type(m).__name__.replace('Message','').lower()}: {m.content}"
        for m in recent_messages
    )

    summarization_prompt = [
        SystemMessage(content=(
            "You are a summarization assistant.\n"
            "Produce a NEW, COMPLETE summary of the conversation.\n\n"
            "Rules:\n"
            "- Use the existing summary only as background context.\n"
            "- Integrate information from ALL recent messages.\n"
            "- Do NOT copy the existing summary.\n"
            "- Do NOT append or label sections.\n"
            "- Focus on the overall conversation, not just the last message.\n"
            "- Output ONLY the final summary text."
            "- Maximum 120 words."
        )),
        HumanMessage(content=f"Existing summary (context only):\n{current_summary}"),
        HumanMessage(content=f"Recent conversation messages:\n{conversation_text}"),
    ]
    summary_result = model.invoke(summarization_prompt)
    new_summary = summary_result.content

    supabase.table("sessions").upsert(
        {
            "session_id": session_id, 
            "summary": new_summary, 
            "updated_at": datetime.now(timezone.utc).isoformat()
        },
        on_conflict="session_id",
    ).execute()

    return new_summary

def embed(text: str) -> list:
    response = openai_client.embeddings.create(
        input=[text], 
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_next_canonical_id() -> int:
    result = supabase.table("attraction").select("canonical_id").order("canonical_id", desc=True).limit(1).execute()
    if result.data and result.data[0].get("canonical_id"):
        return result.data[0]["canonical_id"] + 1
    raise ValueError("Could not determine next canonical_id")

def geocode(location: str) -> tuple[float, float] | None:
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": location, "format": "json", "limit": 1},
        headers={"User-Agent": "TravelApp/1.0"}
    )
    data = response.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    print(f"[geocode] Failed for '{location}'")
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    try:
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return round(2 * R * math.asin(math.sqrt(a)), 3)
    except (TypeError, ValueError):
        return None

# endregion

#############################################################################
# TOOLS
#############################################################################
# region

@tool
def resolve_attractions_tool(attractions: list) -> str:
    """
    Resolves a list of attractions to real database IDs, checking for duplicates
    and creating new records as needed. Call this BEFORE save_itinerary_tool 
    when adding attractions to the itinerary.

    Each attraction must include:
    - name (str)
    - city (str) — must be the nearest large, well-known city. Never use the 
      attraction name, an island name, region, or small obscure town.
      (e.g. Chichen Itza → 'Cancun')
    - state (str, if applicable)
    - country (str) — full country name, never abbreviated
    - description (str)
    - price_level (int, 0=free, 1=cheap, 2=moderate, 3=expensive, 4=luxury)
    - vibe (list of strings, e.g. ['romantic', 'adventurous'])
    - latitude (float)
    - longitude (float)
    - popularity_score (float, 0-100)
    - raw_data (dict with any relevant details such as hours, price_text, website, tips)

    Returns a list of resolved attractions with their real attraction_id values
    to be used directly in save_itinerary_tool.
    """
    print(f"[resolve_attractions_tool] Resolving {len(attractions)} attractions")

    # Step 1 — batch all embeddings in one API call
    valid_attractions = [a for a in attractions if a.get("name") and a.get("city") and a.get("country")]
    texts = [f"{a.get('name')}. {a.get('description', '')}" for a in valid_attractions]
    response = openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = [r.embedding for r in response.data]

    # Step 2 — parallel DB calls, one per attraction
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {
            executor.submit(search_single_attraction, a.get("name"), a.get("country"), emb): a
            for a, emb in zip(valid_attractions, embeddings)
        }
        batch_data = [future.result() for future in as_completed(futures)]

    print(f"[resolve] Vector search raw results:")
    for item in batch_data:
        candidates = [
            f"{c.get('attraction_name')} (id={c.get('attraction_id')}, sim={c.get('similarity', 0):.3f})"
            for c in item.get('candidates', [])
        ]
        print(f"  query='{item.get('input_name')}' → candidates: {candidates}")

    # Step 3 — one LLM call to decide matches
    check_prompt = [
        SystemMessage(content=(
            "You are helping deduplicate a travel attractions database. "
            "For each item, you are given an input_name and a list of candidates with their names, cities, countries, and similarity scores. "
            "Rules:\n"
            "- If ANY candidate has an attraction_name that exactly matches the input_name (case-insensitive), you MUST return match: true for that candidate.\n"
            "- If a candidate name is clearly the same place with minor wording differences (e.g. 'Snorkeling at X' vs 'X'), return match: true.\n"
            "- Use city and country only as a loose sanity check — nearby cities are acceptable.\n"
            "- Only return match: false if NO candidate is plausibly the same physical location.\n"
            "Return JSON only as a list: [{\"name\": \"<input_name value>\", \"match\": true/false, \"attraction_id\": <id or null>}]"
        )),
        HumanMessage(content=f"Attractions to check:\n{json.dumps(batch_data, indent=2)}")
    ]

    check_result = model.invoke(check_prompt)

    try:
        content = check_result.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        match_data = json.loads(content.strip())
        matches = {m["name"]: m.get("attraction_id") for m in match_data if m.get("match")}
        print(f"[resolve] LLM parsed matches: {matches}")
    except Exception as e:
        print(f"[resolve] ⚠️  LLM dedup parse FAILED: {e}")
        print(f"[resolve] Raw content that failed to parse: {check_result.content!r}")
        matches = {}

    # Step 4 — resolve each attraction
    canonical_base = get_next_canonical_id()
    new_attraction_count = 0
    resolved = []

    for a, embedding in zip(valid_attractions, embeddings):
        name = a.get("name")
        city = a.get("city")
        country = a.get("country")

        attraction_id = matches.get(name)

        if attraction_id:
            print(f"[resolve] ✓ MATCHED '{name}' → reusing attraction_id={attraction_id}")
        else:
            print(f"[resolve] ✗ NO MATCH for '{name}' (city={city}, country={country}) — will insert new record")
            place_id = ensure_place_exists(city, a.get("state"), country)
            if not place_id:
                print(f"[resolve] ⚠️  Could not resolve place for '{name}' — skipping")
                continue
            attraction_id = add_attraction(place_id, a, embedding, canonical_base + new_attraction_count)
            if attraction_id:
                print(f"[resolve] + Created new attraction_id={attraction_id} for '{name}'")
                new_attraction_count += 1
            else:
                print(f"[resolve] ⚠️  add_attraction returned None for '{name}'")

        if attraction_id:
            resolved.append({
                "attraction_id": attraction_id,
                "attraction_name": name,
                "attraction_city": city,
            })

    print(f"[resolve_attractions_tool] Resolved {len(resolved)} attractions")
    return json.dumps({"resolved": resolved})


@tool
def save_itinerary_tool(itinerary_id: str, days: Optional[list] = None, trip_name: Optional[str] = None, start_date: Optional[str] = None,
    end_date: Optional[str] = None, notes: Optional[str] = None, unscheduled: Optional[list] = None, place: Optional[list] = None,) -> str:
    """
    Saves updates to the itinerary. Only pass fields that are actually changing — omit the rest.
    Never call this unless the user has explicitly requested a change.
    Creates the itinerary if it doesn't exist, otherwise updates it.

    - itinerary_id: the id of the itinerary to update, found in the itinerary context
    - trip_name: short descriptive name for the trip based on destinations
    - start_date / end_date: trip dates in YYYY-MM-DD format. Use nearest future dates if not specified.
    - days: only pass if modifying the schedule. Must include ALL days even if only one changed. Structure:
        [{ "dayNumber": 1, "stops": [{ "attractionId": 123, "startTime": "09:00", "durationMinutes": 90 }] }]
        - attractionId must be a real integer returned by resolve_attractions_tool, never null or guessed
        - startTime in HH:MM 24-hour format, durationMinutes as integer, stops: [] for empty days
        - When adding or removing days, always update start_date, end_date, and place together
        - When removing a day, move its stops to unscheduled before removing
    - unscheduled: only pass if explicitly adding or removing items. Each item: {attractionId: int, attractionName: str}.
        When scheduling an unscheduled attraction onto a day, pass both updated days AND updated unscheduled with that attraction removed.
    - notes: free-text field for additional trip details or reminders
    - place: list of destination cities the user explicitly mentioned, each with:
        - placeName (str): the city name
        - placeCountry (str): the country name
      Do NOT include nearby cities just because an attraction is located there.
      The placeId will be resolved automatically — do not include it.
    """
    print(f"\n[save_itinerary_tool] Saving itinerary {itinerary_id}")

    updates = {k: v for k, v in {
        "days": days,
        "trip_name": trip_name,
        "start_date": start_date,
        "end_date": end_date,
        "notes": notes,
        "unscheduled": unscheduled,
        "place": place,
    }.items() if v is not None}

    if place:
        resolved_place = []
        for p in place:
            place_name = p.get("placeName")
            place_country = p.get("placeCountry")
            if place_name and place_country:
                place_id = ensure_place_exists(place_name, p.get("placeState"), place_country)
                if place_id:
                    resolved_place.append({"placeId": place_id, "placeName": place_name})
        updates["place"] = resolved_place

    if not updates:
        return json.dumps({"success": False, "reason": "No fields provided to update"})

    try:
        supabase.table("itinerary").upsert(
            {"itinerary_id": itinerary_id, **updates},
            on_conflict="itinerary_id"
        ).execute()
        print(f"[save_itinerary_tool] Saved successfully — fields: {list(updates.keys())}")
        return json.dumps({"success": True, "updated_fields": list(updates.keys())})
    except Exception as e:
        print(f"[save_itinerary_tool] Error: {e}")
        return json.dumps({"success": False, "reason": str(e)})
    

tools = [
    resolve_attractions_tool,
    save_itinerary_tool
]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools, parallel_tool_calls=False)

# endregion

#############################################################################
# GRAPH NODES
#############################################################################
# region

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [model_with_tools.invoke(state["messages"])],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: dict):
    """Execute tool calls requested by the LLM"""
    tool_messages = []
    last_message = state["messages"][-1]
    itinerary_saved = state.get("itinerary_saved", False)
    itinerary = state.get("itinerary")
    itinerary_id = itinerary.get("itinerary_id") if itinerary else None

    for tool_call in last_message.tool_calls:
        print(f"\n[tool_node] Dispatching: {tool_call['name']}")

        tool = tools_by_name.get(tool_call["name"])
        if not tool:
            continue

        if tool_call["name"] == "save_itinerary_tool":
            tool_call["args"]["itinerary_id"] = itinerary_id
            itinerary_saved = True

        observation = tool.invoke(tool_call["args"])

        tool_messages.append(
            ToolMessage(content=observation, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages, "itinerary_saved": itinerary_saved}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop"""
    if state["llm_calls"] >= 30:
        return END

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"

    return END

# endregion

#############################################################################
# STATE GRAPH
#############################################################################
# region

builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", "llm_call", END]
)
builder.add_edge("tool_node", "llm_call")

agent = builder.compile()

# endregion

#############################################################################
# LAMBDA HANDLER
#############################################################################
# region 

def lambda_handler(event, context):
    try:
        # --- Auth ---
        headers = event.get("headers", {})
        if headers.get("x-api-key") != EXPECTED_API_KEY:
            return {"statusCode": 403, "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}, "body": json.dumps({"error": "Forbidden"})}

        # --- Parse body ---
        try:
            body = json.loads(event.get("body", "{}"))
        except json.JSONDecodeError:
            return {"statusCode": 400, "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}, "body": json.dumps({"error": "Invalid JSON body"})}

        user_prompt = body.get("prompt")
        session_id = body.get("session_id", str(uuid.uuid4()))
        itinerary_id = body.get("itinerary_id")
        user_id = body.get("user_id")

        if not user_prompt:
            return {"statusCode": 400, "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}, "body": json.dumps({"error": "Missing prompt"})}

        print(f"[lambda_handler] session_id={session_id}, prompt={user_prompt[:100]}")
        print(f"[lambda_handler] itinerary_id={itinerary_id}")
        print(f"[lambda_handler] user_id={user_id}")

        # --- Session ---
        db_session = (
            supabase
            .table("sessions")
            .select("summary, updated_at")
            .eq("session_id", session_id)
            .execute()
        )

        if not db_session.data or len(db_session.data) == 0:
            supabase.table("sessions").insert({"session_id": session_id}).execute()
            summary = ""
            last_summary_time = None
        else:
            summary = db_session.data[0].get("summary") or ""
            last_summary_time = db_session.data[0].get("updated_at")

        # --- Message history ---
        db_messages = (
            supabase
            .table("messages")
            .select("role, content")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        recent_messages = list(reversed(db_messages.data))
        history_messages = [db_row_to_message(m) for m in recent_messages if m["role"] != "tool"]

        # --- Build messages ---
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            SystemMessage(content=f"Today's date is {datetime.now(timezone.utc).strftime('%B %d, %Y')}. Use this as context when the user mentions travel dates without specifying a year — assume the nearest future date."),
        ]

        if summary:
            messages.append(SystemMessage(content=f"Summary of conversation so far (use as context, prioritize recent messages):\n\n{summary}"))

        # --- Load itinerary ---
        itinerary = None
        if itinerary_id:
            itin_result = (
                supabase
                .table("itinerary")
                .select("*")
                .eq("itinerary_id", itinerary_id)
                .execute()
            )
            itinerary = itin_result.data[0] if itin_result.data else {
                "itinerary_id": itinerary_id,
                "user_id": user_id,
                "days": [],
                "unscheduled": [],
                "place": [],
                "trip_name": None,
                "start_date": None,
                "end_date": None,
                "notes": None,
            }
            messages.append(SystemMessage(content=f"Current itinerary the user wants to edit:\n{json.dumps(itinerary, indent=2)}"))

        messages.extend(history_messages)
        messages.append(HumanMessage(content=user_prompt))

        # --- Run agent ---
        result = agent.invoke({
            "messages": messages, 
            "llm_calls": 0, 
            "itinerary": itinerary,
            "itinerary_saved": False,
        })

        supabase.table("sessions").update({
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("session_id", session_id).execute()

        # --- Final message ---
        final_message = result["messages"][-1]
        output = final_message.content or "I've made the updates! Let me know if you'd like any other changes."

        # --- Persist messages ---
        supabase.table("messages").insert([
            {"session_id": session_id, "role": "user", "content": user_prompt},
            {"session_id": session_id, "role": "assistant", "content": output}
        ]).execute()

        # --- Summarize if needed ---
        query = (
            supabase
            .table("messages")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
        )

        if last_summary_time:
            query = query.gt("created_at", last_summary_time)

        unsummarized = query.execute()
        unsummarized_messages = [db_row_to_message(m) for m in unsummarized.data]

        if len(unsummarized_messages) >= 10:
            update_session_summary(session_id, unsummarized_messages, summary)

        itinerary_saved = result.get("itinerary_saved", False)

        if itinerary_saved and user_id and itinerary_id:
            supabase.table("itinerary").update({"user_id": user_id}).eq("itinerary_id", itinerary_id).execute()

        print(f"[lambda_handler] output={output}")
        print(f"[lambda_handler] session_id={session_id}")
        print(f"[lambda_handler] itinerary_saved={itinerary_saved}")

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"output": output, "session_id": session_id, "refresh_data": itinerary_saved})
        }

    except Exception as e:
        print(f"[lambda_handler] Error: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Internal server error"})
        }
    
# endregion