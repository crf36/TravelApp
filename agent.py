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

# Integrate with itinerary generation
    # Create a tool that adds an attraction to the itinerary
    # Create a tool that reads from the existing itinerary

# Also look into adding individual attractions to the DB when a place exists but certain popular attractions do not.

load_dotenv()
DRY_RUN = False

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

    "You have access to a database of attractions and the following tools:\n\n"

    "- update_metadata_tool: Extracts and stores the user's travel preferences and destination "
    "details (city, country, interests, vibe, budget, etc.) from the conversation. Call this "
    "when the user provides a destination or any new preference information.\n\n"

    "- check_if_place_exists_tool: Checks if a destination exists in the database. "
    "Returns exists=true/false and place_id if it exists. Call this after update_metadata_tool "
    "whenever a destination is determined.\n\n"

    "- add_place_to_db_tool: Adds a new destination to the database. Call this only when "
    "check_if_place_exists_tool returns exists=false. Returns the new place_id.\n\n"

    "- search_attractions_tool: Searches the database for attractions matching the user's "
    "destination and preferences. Call this only when check_if_place_exists_tool returns "
    "exists=true. Returns a list of attractions and valid_attractions_found=true/false. "
    "If no results are found, also returns existing_attraction_names for deduplication.\n\n"

    "- add_attractions_to_db_tool: Saves a structured list of attractions to the database. "
    "Call this whenever you suggest attractions from your own knowledge, passing the place_id "
    "from check_if_place_exists_tool or add_place_to_db_tool along with the attractions list.\n\n"

    "Conversation Flow:\n"
    "- Have a natural conversation to understand the user's destination and preferences before searching.\n"
    "- Once you have a destination and a general sense of what they're looking for, proceed to search.\n"
    "- Use your judgment on when you have enough information to be helpful.\n"
    "- If a city name is ambiguous, ask the user to clarify the state or country before proceeding.\n\n"

    "Tool Flow — follow this order when ready to search:\n"
    "1. Call update_metadata_tool to store the destination and preferences.\n"
    "2. Call check_if_place_exists_tool to verify the destination exists in the DB.\n"
    "3a. If exists=false: call add_place_to_db_tool to add it. Then call "
    "add_attractions_to_db_tool with a list of suggested attractions and the place_id "
    "BEFORE responding to the user.\n"
    "3b. If exists=true: call search_attractions_tool to retrieve matching attractions.\n"
    "4. If search_attractions_tool returns valid_attractions_found=false, call "
    "add_attractions_to_db_tool with attractions NOT already in existing_attraction_names "
    "BEFORE responding to the user.\n"
    "5. After all tool calls are complete, respond to the user with the attraction suggestions.\n"
    "6. When presenting attractions, show a maximum of 10. If more are returned, select the "
    "most relevant based on the user's preferences.\n\n"

    "When calling add_attractions_to_db_tool, pass place_id and a structured list where each attraction includes:\n"
    "  - name (str)\n"
    "  - description (str)\n"
    "  - city (str)\n"
    "  - state (str, if applicable)\n"
    "  - country (str)\n"
    "  - price_level (int, 0-4 where 0=free, 1=cheap, 2=moderate, 3=expensive, 4=luxury)\n"
    "  - vibe (list of strings, e.g. ['romantic', 'adventurous', 'family-friendly'])\n"
    "  - latitude (float, estimated coordinates of the attraction)\n"
    "  - longitude (float, estimated coordinates of the attraction)\n"
    "  - popularity_score (float, 0-100 estimate of how popular this attraction is)\n"
    "  - raw_data (dict with any relevant details such as hours, price_text, website, tips)\n\n"
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

class UpdateMetadataInput(TypedDict, total=False):
    metadata: dict
    user_message: str
    conversation: list

class Metadata(TypedDict, total=False):
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    travel_month: Optional[str]
    price_level: Optional[int]
    distance: Optional[int]
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
def update_metadata_tool(data: UpdateMetadataInput) -> str:
    """
    Update the user's travel metadata based on the latest message and conversation context.
    Expects a dict with keys: metadata, user_message, conversation.
    """
    print("\n[update_metadata_tool] Called")

    metadata = data.get("metadata", {})
    user_message = data.get("user_message", "")
    conversation = data.get("conversation", [])

    prompt = f"""
        You are a travel assistant. Update the user's travel metadata based on the latest message.

        Valid metadata fields:
        - city (str)
        - state (str)
        - country (str)
        - travel_month (str)
        - price_level (int, 0-4)
        - distance (int)
        - interests (list of strings)
        - vibe (list of strings)
        - semantic_query (str) — a rich, natural language sentence summarizing what the 
        user is looking for. Combine their interests, vibe, and any other context into 
        a descriptive phrase. Example: "romantic candlelit dinner at an upscale restaurant 
        with an intimate atmosphere". Always generate this when interests or vibe are present.

        Current metadata:
        {json.dumps(metadata, indent=2)}

        Recent conversation:
        {conversation}

        Latest user message:
        "{user_message}"

        Instructions:
        - Return ONLY the full metadata JSON object.
        - Update fields based on this message and the conversation context.
        - Remove or overwrite fields that are no longer relevant.
        - Preserve values that are still consistent.
        - Always return valid JSON only.
        - Do not add fields with null values. Only include fields that have actual values.
        - Only add valid cities, states, or countries ("Not Mexico" is not valid)
        - Always use full country names (e.g. "United States" not "USA" or "US", "United Kingdom" not "UK", "United Arab Emirates" not "UAE").
        - Infer state and country from city when the city is well-known and unambiguous 
        - Only leave state/country blank if the city name is ambiguous across multiple locations.

        Price level mapping — apply strictly:
        - "free", "no cost", "at no cost", "free attractions" → price_level: 0
        - "cheap", "budget", "affordable", "inexpensive" → price_level: 1
        - "moderate", "mid-range", "reasonable" → price_level: 2
        - "expensive", "upscale", "pricey" → price_level: 3
        - "luxury", "very expensive", "high-end" → price_level: 4
    """

    llm_output = model.invoke(prompt)
    raw = llm_output.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        updated = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[update_metadata_tool] Failed to parse LLM output — falling back to existing metadata\nRaw: {raw}")
        updated = metadata

    print(f"[update_metadata_tool] Result: {json.dumps(updated, indent=2)}")
    return json.dumps(updated)

@tool
def check_if_place_exists_tool(city: Optional[str] = None, state: Optional[str] = None, country: Optional[str] = None) -> str:
    """
    Checks if a destination exists in the place table.
    Returns exists=true/false and place_id if found.
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
    Adds a new destination to the place table.
    Call only when check_if_place_exists_tool returns exists=false.
    Provide estimated latitude and longitude. Returns the new place_id.
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

    if DRY_RUN:
        fake_id = str(uuid.uuid4())
        print(f"[add_place_to_db_tool] DRY RUN — would have inserted: {row}")
        print(f"[add_place_to_db_tool] DRY RUN — fake place_id: {fake_id}")
        return json.dumps({"success": True, "place_id": fake_id, "dry_run": True})

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
    Search for attractions using semantic vector search and metadata filters.
    Returns matched attractions and valid_attractions_found=true/false.
    If no results, also returns existing_attraction_names for deduplication.
    """
    print(f"\n[search_attractions_tool] Called")

    interests = metadata.get("interests") or []
    vibe = metadata.get("vibe") or []
    city = metadata.get("city") or None
    state = metadata.get("state") or None
    country = metadata.get("country") or None
    price_level = metadata.get("price_level")

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

    # No filtered results — fetch all for this location for deduplication
    print(f"[search_attractions_tool] No results — fetching all attractions for location (deduplication)")

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
    Saves a list of LLM-suggested attractions to the database, linked to a place.
    Requires place_id from check_if_place_exists_tool or add_place_to_db_tool.
    Each attraction should include: name, description, city, state, country, price_level,
    vibe, latitude, longitude, distance_from_place, popularity_score, raw_data.
    """
    print(f"\n[add_attractions_to_db_tool] Called — place_id={place_id}, {len(attractions)} attractions provided")

    if not attractions:
        return json.dumps({"success": False, "reason": "No attractions provided"})

    next_canonical_id = get_next_canonical_id()

    place_lat = None
    place_lng = None

    if not DRY_RUN:
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

        location_str = f"{name}, {city}, {country}"
        coords = geocode(location_str)
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

    if DRY_RUN:
        print(f"[add_attractions_to_db_tool] DRY RUN — would have inserted {len(rows)} attractions:")
        for r in rows:
            print(f"  - {r['attraction_name']} | {r['attraction_city']}, {r['attraction_countryregion']} | price_level={r['attraction_pricelevel']} | popularity={r['attraction_popularityscore']}")
            print(f"    vibe={r['attraction_vibe']} | lat={r['attraction_latitude']} | lng={r['attraction_longitude']} | dist={r['attraction_distancefromplace']}")
            print(f"    {r['attraction_summary'][:120]}...")
        return json.dumps({"success": True, "saved": len(rows), "dry_run": True})

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

tools = [
    update_metadata_tool,
    check_if_place_exists_tool,
    add_place_to_db_tool,
    search_attractions_tool,
    add_attractions_to_db_tool,
]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

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

    conversation = [
        {"role": type(m).__name__.replace("Message", "").lower(), "content": m.content}
        for m in state["messages"]
        if hasattr(m, "content") and m.content
    ]

    user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    for tool_call in last_message.tool_calls:
        print(f"\n[tool_node] Dispatching: {tool_call['name']}")

        tool = tools_by_name.get(tool_call["name"])
        if not tool:
            continue

        if tool_call["name"] == "update_metadata_tool":
            args = {
                "data": {
                    "metadata": new_metadata,
                    "user_message": user_message,
                    "conversation": conversation
                }
            }
        elif tool_call["name"] == "search_attractions_tool":
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
                new_metadata = json.loads(observation)
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

    messages.extend(history_messages)
    messages.append(HumanMessage(content=user_input))

    result = agent.invoke({
        "messages": messages,
        "llm_calls": 0,
        "metadata": metadata
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
#     test_cases = [
#         ("Orem Utah TEST", [
#             "What are some things to do in Orem, Utah?"
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
#             result = handler({"prompt": prompt, "session_id": session_id})
#             print(f"\nAGENT: {result['output']}")

# if __name__ == "__main__":
#     debug_agent_test()

# endregion