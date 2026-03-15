from bedrock_agentcore.runtime import BedrockAgentCoreApp
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated, Any, Dict, Optional, List
from typing import Literal
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timezone
import operator, os, json, uuid, requests, math, boto3

load_dotenv()

#############################################################################
# SETUP
#############################################################################
# region

app = BedrockAgentCoreApp()
model = ChatOpenAI(model="gpt-4o", temperature=0)
openai_client = OpenAI()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(url, key)

lambda_client = boto3.client("lambda", region_name=os.getenv("AWS_REGION", "us-east-1"))

# endregion

#############################################################################
# SYSTEM PROMPT
#############################################################################
# region

SYSTEM_PROMPT = (
    "You are an expert travel assistant helping users discover attractions, plan trips, "
    "and explore destinations around the world. Your goal is to provide genuinely helpful, "
    "accurate, and personalized travel guidance.\n\n"

    "You have access to a database of attractions and the following tools:\n"
    "- update_metadata_tool: Sets the current search context.\n"
    "- check_if_place_exists_tool: Checks if a destination exists in the database.\n"
    "- add_place_to_db_tool: Adds a new destination to the database.\n"
    "- search_attractions_tool: Searches for attractions matching the current context.\n"
    "- add_attractions_to_db_tool: Saves a structured list of attractions to the database.\n"
    "- save_itinerary_tool: Saves updates to the itinerary.\n\n"

    "General Rules:\n"
    "- Have a natural conversation to understand the user's destination and preferences before searching.\n"
    "- Once you have a destination and a general sense of what they're looking for, proceed to search.\n"
    "- If a city name is ambiguous, ask the user to clarify the state or country before proceeding.\n\n"

    "Tool Flow — follow this order when ready to search:\n"
    "1. Call update_metadata_tool with the destination and preferences.\n"
    "2. Call check_if_place_exists_tool.\n"
    "3a. If exists=false: call add_place_to_db_tool, then add_attractions_to_db_tool "
    "with suggested attractions BEFORE responding to the user.\n"
    "3b. If exists=true: call search_attractions_tool.\n"
    "4. If search_attractions_tool returns valid_attractions_found=false, call "
    "add_attractions_to_db_tool with attractions NOT in existing_attraction_names."
    "BEFORE responding to the user\n"
    "5. After all tool calls are complete, respond to the user with the attraction suggestions.\n"
    "6. When presenting attractions, show a maximum of 10. If more are returned, select the "
    "most relevant based on the user's preferences.\n\n"

    "Itinerary Editing:\n"
    "- If an itinerary is provided in your context, the user wants to edit it.\n"
    "- When the user requests any change, call save_itinerary_tool with only the fields "
    "that need to be updated.\n"
    "- Always call save_itinerary_tool BEFORE responding to the user.\n"
    "- Always confirm the change to the user after saving.\n"
    "- Never call save_itinerary_tool unless the user has explicitly requested a change.\n"
    "- Never save a stop with a null or missing attractionId.\n\n"

    "Generating an itinerary across multiple destinations:\n"
    "- Loop through each city in the itinerary's place field one at a time.\n"
    "- For each city call update_metadata_tool, check_if_place_exists_tool, and "
    "search_attractions_tool sequentially before moving to the next city.\n"
    "- Only call save_itinerary_tool after all cities have been searched.\n"
)

# endregion

#############################################################################
# TYPE DEFINITIONS
#############################################################################
# region

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    metadata: Dict[str, Any]
    itinerary: Optional[Dict[str, Any]]

class Metadata(TypedDict, total=False):
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    price_level: Optional[int]
    interests: List[str]
    vibe: List[str]
    semantic_query: str

# endregion

#############################################################################
# HELPERS
#############################################################################
# region

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

def update_session_summary(session_id: str, recent_messages: list[AnyMessage], current_summary: str):
    """Send messages and current summary to LLM, get an updated summary, and store it in the DB"""
    conversation_text = "\n".join(
        f"{type(m).__name__.replace('Message','').lower()}: {m.content}"
        for m in recent_messages
    )

    summarization_prompt = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=f"Existing summary (context only):\n{current_summary}"),
        HumanMessage(content=f"Recent conversation messages:\n{conversation_text}"),
    ]

    summary_result = model.invoke(summarization_prompt)
    new_summary = summary_result.content

    supabase.table("sessions").upsert(
        {
            "session_id": session_id,
            "summary": new_summary,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="session_id",
    ).execute()

    return new_summary

def embed(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_next_canonical_id() -> int:
    result = supabase.table("attraction").select("canonical_id").order("canonical_id", desc=True).limit(1).execute()
    if result.data and result.data[0].get("canonical_id"):
        return result.data[0]["canonical_id"] + 1
    raise ValueError("Could not determine next canonical_id — attraction table may be empty or canonical_id is null")

def geocode(location: str) -> tuple[float, float] | None:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    response = requests.get(url, params={"address": location, "key": api_key})
    data = response.json()

    if data["status"] == "OK":
        location = data["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]
    
    print(f"[geocode] Failed for '{location}': {data['status']}")
    return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float | None:
    try:
        R = 6371  # Earth's radius in kilometers
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
def update_metadata_tool(city: Optional[str] = None, state: Optional[str] = None, country: Optional[str] = None, price_level: Optional[int] = None,
    interests: Optional[List[str]] = None, vibe: Optional[List[str]] = None, semantic_query: Optional[str] = None) -> str:
    """
    Updates the current search context. Pass only the fields that need to be set —
    omit fields that haven't changed.

    - city: full city name (e.g. 'Seattle')
    - state: full state or province name (e.g. 'Washington'). Infer from city when unambiguous.
    - country: full country name (e.g. 'United States', 'United Kingdom', 'United Arab Emirates').
      Never abbreviate (not 'USA', 'UK', 'UAE').
    - price_level: 0=free, 1=cheap, 2=moderate, 3=expensive, 4=luxury
    - interests: list of activity types the user is interested in (e.g. ['hiking', 'museums', 'food'])
    - vibe: list of mood or atmosphere descriptors (e.g. ['romantic', 'relaxing', 'adventurous'])
    - semantic_query: a rich natural language phrase combining interests, vibe, and any other
      context into a descriptive search phrase. Always include when interests or vibe are present.
      Good examples:
        'romantic candlelit dinner at an upscale restaurant with an intimate atmosphere'
        'adventurous outdoor activities with stunning mountain views'
        'fun and relaxing anniversary trip for a couple'
    """
    print("\n[update_metadata_tool] Called")

    updated = {k: v for k, v in {
        "city": city,
        "state": state,
        "country": country,
        "price_level": price_level,
        "interests": interests,
        "vibe": vibe,
        "semantic_query": semantic_query,
    }.items() if v is not None}

    print(f"[update_metadata_tool] Result: {json.dumps(updated, indent=2)}")
    return json.dumps(updated)


@tool
def check_if_place_exists_tool(city: Optional[str] = None, state: Optional[str] = None, country: Optional[str] = None) -> str:
    """
    Checks if a destination exists in the database.
    Always call this after update_metadata_tool before searching for attractions.
    Returns exists=true/false and place_id if found.

    - city: preferred for lookup when available
    - state: used as fallback if no city provided
    - country: used as fallback if no city or state provided
    """
    print(f"\n[check_if_place_exists_tool] Checking: city={city}, state={state}, country={country}")

    if not any([city, state, country]):
        return json.dumps({"exists": False, "reason": "No location provided"})

    query = supabase.table("place").select("place_id, place_city, place_stateprovince, place_countryregion")

    if city:
        query = query.ilike("place_city", city)
    elif state:
        query = query.ilike("place_stateprovince", state)
    elif country:
        query = query.ilike("place_countryregion", country)

    result = query.execute()

    if result.data:
        place_id = result.data[0]["place_id"]
        print(f"[check_if_place_exists_tool] Found — place_id={place_id}")
        return json.dumps({"exists": True, "place_id": place_id})

    print(f"[check_if_place_exists_tool] Not found in DB")
    return json.dumps({"exists": False})


@tool
def add_place_to_db_tool(city: Optional[str] = None, state: Optional[str] = None, country: Optional[str] = None) -> str:
    """
    Adds a new destination to the database.
    Only call this when check_if_place_exists_tool returns exists=false.
    Returns the new place_id to use in subsequent calls.
    """
    print(f"\n[add_place_to_db_tool] Adding: city={city}, state={state}, country={country}")

    latitude, longitude = None, None
    coords = geocode(f"{city}, {state}, {country}" if state else f"{city}, {country}")
    if coords:
        latitude, longitude = coords

    row = {
        "place_type": ["city"],
        "place_city": city,
        "place_stateprovince": state,
        "place_countryregion": country,
        "place_latitude": latitude,
        "place_longitude": longitude,
    }

    try:
        result = supabase.table("place").insert(row).execute()
        place_id = result.data[0]["place_id"]
        print(f"[add_place_to_db_tool] Saved successfully — place_id={place_id}")
        return json.dumps({"success": True, "place_id": place_id})
    except Exception as e:
        print(f"[add_place_to_db_tool] Error: {e}")
        return json.dumps({"success": False, "reason": str(e)})


@tool
def search_attractions_tool(metadata: dict) -> str:
    """
    Searches the database for attractions matching the current context.
    Only call this after check_if_place_exists_tool returns exists=true.
    Returns results and valid_attractions_found=true/false.
    If valid_attractions_found=false, returns existing_attraction_names for deduplication —
    pass these to add_attractions_to_db_tool to avoid adding duplicates.
    """
    print(f"\n[search_attractions_tool] Called")

    city = metadata.get("city") or None
    state = metadata.get("state") or None
    country = metadata.get("country") or None
    price_level = metadata.get("price_level")
    interests = metadata.get("interests") or []
    vibe = metadata.get("vibe") or []

    semantic_query = metadata.get("semantic_query") or (
        ", ".join(interests + vibe) if (interests or vibe) else "things to do"
    )

    print(f"[search_attractions_tool] Filters — city={city}, state={state}, country={country}, price_level={price_level}")
    print(f"[search_attractions_tool] Semantic query: \"{semantic_query}\"")

    embedding = embed(semantic_query)

    result = supabase.rpc("match_attractions", {
        "query_embedding": embedding,
        "filter_city": city,
        "filter_state": state,
        "filter_country": country,
        "filter_price_level": price_level,
        "match_count": 100
    }).execute()

    print(f"[search_attractions_tool] {len(result.data)} attractions returned")

    if result.data:
        return json.dumps({"results": result.data, "valid_attractions_found": True})

    print(f"[search_attractions_tool] No results — fetching existing attractions for deduplication")

    existing = supabase.rpc("match_attractions", {
        "query_embedding": embedding,
        "filter_city": city,
        "filter_state": state,
        "filter_country": country,
        "filter_price_level": None,
        "match_count": 200
    }).execute()

    existing_names = [a.get("attraction_name") for a in existing.data] if existing.data else []
    print(f"[search_attractions_tool] {len(existing_names)} existing attractions found for deduplication")

    return json.dumps({
        "results": [],
        "valid_attractions_found": False,
        "existing_attraction_names": existing_names
    })


@tool
def add_attractions_to_db_tool(place_id: str, attractions: list) -> str:
    """
    Saves a list of attractions to the database, linked to a place.
    Only call this when search_attractions_tool returns valid_attractions_found=false,
    or when adding a new place for the first time after add_place_to_db_tool.
    Requires place_id from check_if_place_exists_tool or add_place_to_db_tool.

    Each attraction must include:
    - name (str)
    - description (str)
    - city (str)
    - state (str, if applicable)
    - country (str) — full country name, never abbreviated
    - price_level (int, 0=free, 1=cheap, 2=moderate, 3=expensive, 4=luxury)
    - vibe (list of strings, e.g. ['romantic', 'adventurous', 'family-friendly'])
    - latitude (float)
    - longitude (float)
    - popularity_score (float, 0-100)
    - raw_data (dict with any relevant details such as hours, price_text, website, tips)
    """
    print(f"\n[add_attractions_to_db_tool] Called — place_id={place_id}, {len(attractions)} attractions provided")

    if not attractions:
        return json.dumps({"success": False, "reason": "No attractions provided"})

    next_canonical_id = get_next_canonical_id()

    place_result = supabase.table("place").select("place_latitude, place_longitude").eq("place_id", place_id).execute()
    place_lat = place_result.data[0].get("place_latitude") if place_result.data else None
    place_lng = place_result.data[0].get("place_longitude") if place_result.data else None

    rows = []
    for i, a in enumerate(attractions):
        name = a.get("name")
        city = a.get("city")
        country = a.get("country")

        if not name or not country:
            print(f"[add_attractions_to_db_tool] Skipping incomplete entry: {a}")
            continue

        coords = geocode(f"{name}, {city}, {country}")
        attraction_lat = coords[0] if coords else a.get("latitude")
        attraction_lng = coords[1] if coords else a.get("longitude")

        description = a.get("description", "")
        embedding = embed(f"{name}. {description}")

        rows.append({
            "place_id": place_id,
            "attraction_name": name,
            "attraction_summary": description,
            "attraction_city": city,
            "attraction_stateprovince": a.get("state"),
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
            "canonical_id": next_canonical_id + i,
        })

    if not rows:
        return json.dumps({"success": False, "reason": "No valid attractions to save"})

    try:
        result = supabase.table("attraction").insert(rows).execute()
        print(f"[add_attractions_to_db_tool] Saved {len(rows)} attractions successfully")

        attraction_payloads = [
            {
                "attraction_id": inserted.get("attraction_id"),
                "place_id": place_id,
                "name": saved_row["attraction_name"],
                "city": saved_row.get("attraction_city"),
                "country": saved_row.get("attraction_countryregion"),
            }
            for saved_row, inserted in zip(rows, result.data)
            if inserted.get("attraction_id")
        ]

        lambda_client.invoke(
            FunctionName=os.getenv("IMAGE_LAMBDA_NAME"),
            InvocationType="Event",
            Payload=json.dumps({"attractions": attraction_payloads})
        )
        print(f"[add_attractions_to_db_tool] Image Lambda invoked for {len(attraction_payloads)} attractions")

        return json.dumps({"success": True, "saved": len(rows)})
    except Exception as e:
        print(f"[add_attractions_to_db_tool] Error: {e}")
        return json.dumps({"success": False, "reason": str(e)})
    

@tool
def save_itinerary_tool(itinerary_id: str, days: Optional[list] = None, trip_name: Optional[str] = None, start_date: Optional[str] = None,
    end_date: Optional[str] = None, notes: Optional[str] = None, unscheduled: Optional[list] = None, place: Optional[list] = None,) -> str:
    """
    Saves updates to the itinerary. Pass only the fields that need to be changed — omit the rest.
    Never call this unless the user has explicitly requested a change.

    - itinerary_id: the id of the itinerary to update, found in the itinerary context
    - days: full updated days array. Must include ALL days even if unchanged. Structure:
        [{ "dayNumber": 1, "stops": [{ "attractionId": 123, "startTime": "09:00", "durationMinutes": 90 }] }]
        Rules: attractionId must be a real integer from search_attractions_tool (never null or guessed),
        startTime in HH:MM 24-hour format, durationMinutes as integer, stops: [] for empty days.
        When adding or removing days:
        - Update start_date and end_date to reflect the new trip duration.
        - Update the place array if the new day is in a different city than existing days.
        - Always pass days, start_date, end_date, and place together when the number of days changes.
    - trip_name: short descriptive name for the trip
    - start_date / end_date: trip dates in YYYY-MM-DD format
    - notes: free-text field for additional trip details or reminders
    - unscheduled: attractions not yet assigned to a day, each with attractionId (int) and 
        attractionName (str). If the user adds an attraction without specifying a day or time, 
        add it here instead of the days array. Always preserve existing unscheduled items — 
        pass the full updated list, not just the new item.
    - place: list of place objects, each with placeId (str) and placeName (str)
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

    if not updates:
        return json.dumps({"success": False, "reason": "No fields provided to update"})

    try:
        supabase.table("itinerary").update(updates).eq("itinerary_id", itinerary_id).execute()
        print(f"[save_itinerary_tool] Saved successfully — fields: {list(updates.keys())}")
        return json.dumps({"success": True, "updated_fields": list(updates.keys())})
    except Exception as e:
        print(f"[save_itinerary_tool] Error: {e}")
        return json.dumps({"success": False, "reason": str(e)})
    

tools = [
    update_metadata_tool,
    check_if_place_exists_tool,
    add_place_to_db_tool,
    search_attractions_tool,
    add_attractions_to_db_tool,
    save_itinerary_tool
]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools, parallel_tool_calls=False)

# endregion

#############################################################################
# NODES
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
    new_metadata = state.get("metadata", {})

    for tool_call in last_message.tool_calls:
        print(f"\n[tool_node] Dispatching: {tool_call['name']}")

        tool = tools_by_name.get(tool_call["name"])
        if not tool:
            continue

        if tool_call["name"] == "search_attractions_tool":
            args = {"metadata": new_metadata}
        elif tool_call["name"] == "check_if_place_exists_tool":
            args = {
                "city": new_metadata.get("city"),
                "state": new_metadata.get("state"),
                "country": new_metadata.get("country"),
            }
        elif tool_call["name"] == "add_place_to_db_tool":
            args = {
                "city": new_metadata.get("city"),
                "state": new_metadata.get("state"),
                "country": new_metadata.get("country"),
            }
        else:
            args = tool_call["args"]

        observation = tool.invoke(args)

        if tool_call["name"] == "update_metadata_tool":
            try:
                new_values = json.loads(observation)
                new_metadata = {**new_metadata, **new_values}
            except:
                pass

        tool_messages.append(
            ToolMessage(content=observation, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages, "metadata": new_metadata}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop"""
    if state["llm_calls"] >= 10:
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
# AGENTCORE ENTRYPOINT
#############################################################################
# region

@app.entrypoint
def handler(event: dict):
    user_input = event.get("prompt")
    session_id = event.get("session_id")
    itinerary_id = event.get("itinerary_id")

    if not user_input:
        return {"output": "No prompt provided."}
    if not session_id:
        return {"output": "Missing session_id."}

    db_session = (
        supabase
        .table("sessions")
        .select("summary, updated_at, metadata")
        .eq("session_id", session_id)
        .execute()
    )

    if not db_session.data or len(db_session.data) == 0:
        supabase.table("sessions").insert({"session_id": session_id}).execute()
        summary = ""
        last_summary_time = None
        metadata = {}
    else:
        summary = db_session.data[0].get("summary") or ""
        last_summary_time = db_session.data[0].get("updated_at")
        metadata = db_session.data[0].get("metadata") or {}

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
    history_messages = [
        db_row_to_message(m) for m in recent_messages
        if m["role"] not in ("tool",)
    ]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Current metadata: {json.dumps(metadata)}")
    ]

    if summary:
        messages.append(
            SystemMessage(
                content=(
                    "The following is a summary of the conversation so far. "
                    "Use it as context, but prioritize the most recent messages.\n\n"
                    f"{summary}"
                )
            )
        )

    itinerary = None
    if itinerary_id:
        itin_result = (
            supabase
            .table("itinerary")
            .select("*")
            .eq("itinerary_id", itinerary_id)
            .single()
            .execute()
        )
        itinerary = itin_result.data if itin_result.data else None
        if itinerary:
            messages.append(
                SystemMessage(content=f"Current itinerary the user wants to edit:\n{json.dumps(itinerary, indent=2)}")
            )

    messages.extend(history_messages)
    messages.append(HumanMessage(content=user_input))


    result = agent.invoke({
        "messages": messages,
        "llm_calls": 0,
        "metadata": metadata,
        "itinerary": itinerary
    })

    updated_metadata = result.get("metadata", metadata)
    supabase.table("sessions").update({
        "metadata": updated_metadata
    }).eq("session_id", session_id).execute()

    final_message = result["messages"][-1]

    supabase.table("messages").insert([
        {"session_id": session_id, "role": "user", "content": user_input},
        {"session_id": session_id, "role": "assistant", "content": final_message.content}
    ]).execute()

    query = (
        supabase.table("messages")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
    )

    if last_summary_time:
        query = query.gt("created_at", last_summary_time)

    unsummarized_db_messages = query.execute()
    unsummarized_messages = [db_row_to_message(m) for m in unsummarized_db_messages.data]

    if len(unsummarized_messages) >= 10:
        update_session_summary(session_id, unsummarized_messages, summary)

    return {"output": final_message.content}

# endregion

#############################################################################
# APP INITIALIZER
#############################################################################
# region

if __name__ == "__main__":
    app.run()

# def debug_agent_test():
#     itinerary = "e0a4f983-d9cb-40ee-a1f3-fe3e54f8ba15"

#     test_cases = [
#         ("Itinerary Read TEST", [
#             "Can you add a day to my itinerary and add things to do in Provo Utah to my itinerary?"
#         ])
#     ]

#     for case_name, prompts in test_cases:
#         session_id = str(uuid.uuid4())
#         print(f"\n{'#'*60}")
#         print(f"# {case_name}")
#         print(f"# Session ID: {session_id}")
#         print(f"{'#'*60}")

#         for prompt in prompts:
#             print("\n" + "="*60)
#             print(f"USER: {prompt}")
#             print("="*60)
#             result = handler({"prompt": prompt, "session_id": session_id, "itinerary_id": itinerary})
#             print(f"\nAGENT: {result['output']}")

# if __name__ == "__main__":
#     debug_agent_test()

# endregion