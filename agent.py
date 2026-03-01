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
import operator, os, json, boto3

# Only used when running locally
load_dotenv()

#############################################################################
# SETUP
#############################################################################
# region

app = BedrockAgentCoreApp()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
openai_client = OpenAI()

# Set up Supabase client
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(url, key)

SYSTEM_PROMPT = (
    "You are a helpful travel assistant. Your job is to understand the user's travel goals, "
    "gather any missing details, and provide accurate, concise guidance.\n\n"

    "Core Behavior:\n"
    "- Ask clarifying questions directly only if the destination is missing.\n"
    "- Use tools only when they provide missing or external information.\n"
    "- ONLY recommend attractions that are returned from search_attractions_tool. Never suggest attractions from your own knowledge.\n"
    "- If no attractions are found in the database, say so honestly and do not fabricate alternatives.\n"
    "- If you do not know something and no tool can help, say so.\n"
    "- Never fabricate facts.\n"
    "- Keep responses concise but informative.\n"
    "- Use your judgment when metadata is only partially complete — partial information may still be enough to proceed.\n\n"

    "Metadata Rules:\n"
    "The user's travel metadata may include:\n"
    "- city, state, country (destination)\n"
    "- travel_month or dates\n"
    "- interests (e.g., restaurants, museums, amusement parks, outdoors)\n"
    "- vibe (e.g., romantic, adventurous, relaxing, family-friendly)\n"
    "- price_level (0-4, where 0 is free and 4 is very expensive)\n"
    "- distance (in miles/km from a location)\n\n"

    "The ONLY required field to search is a destination (city, state, or country). "
    "All other fields are optional and will improve results if provided. "
    "Do not ask for missing optional fields before searching — search with what you have "
    "and present results. Only ask clarifying questions if destination is missing.\n\n"

    "If the user changes their destination, dates, or preferences, update the metadata accordingly "
    "and treat previous values as overridden. Do not carry over information that is no longer relevant.\n\n"

    "Tool Usage — follow this order strictly when a destination is provided:\n"
    "1. Call update_metadata_tool to extract and store the destination and any other details.\n"
    "2. Call get_place_tool to check if the destination exists in the database.\n"
    "3. If get_place_tool confirms the place exists, call search_attractions_tool to retrieve results.\n"
    "4. If get_place_tool says the place is not in the DB, inform the user that the location is being "
    "added and results will be available in 10-15 minutes. Do NOT suggest any attractions.\n"
    "5. If search_attractions_tool returns no results, tell the user no results were found. Do NOT suggest attractions from your own knowledge.\n\n"

    "Other Tool Rules:\n"
    "- Only call update_metadata_tool when the user provides NEW information. Do not call it if metadata is already up to date.\n"
    "- After calling any tool, wait for the result before continuing.\n"
    "- If no tool is needed, respond normally in natural language.\n\n"

    "Available tools:\n"
    "- update_metadata_tool: Update the user's travel metadata.\n"
    "- get_place_tool: Check if a city exists in the database. Pass only the city name.\n"    
    "- search_attractions_tool: Retrieve attractions based on metadata.\n"
)

# endregion

#############################################################################
# TYPE DEFINITIONS
#############################################################################
# region

# Define state dictionary
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
        HumanMessage(
            content=f"Existing summary (context only):\n{current_summary}"
        ),
        HumanMessage(
            content=f"Recent conversation messages:\n{conversation_text}"
        ),
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
    """

    llm_output = model.invoke(prompt)
    updated = json.loads(llm_output.content)
    return json.dumps(updated)

@tool
def search_attractions_tool(metadata: dict) -> str:
    """
    Search for attractions using semantic vector search and metadata filters.
    """
    interests = metadata.get("interests") or []
    vibe = metadata.get("vibe") or []
    city = metadata.get("city") or None
    state = metadata.get("state") or None
    country = metadata.get("country") or None
    price_level = metadata.get("price_level") or None

    # Build semantic query string from interests and vibe
    semantic_parts = interests + vibe
    semantic_query = ", ".join(semantic_parts) if semantic_parts else "things to do"

    # Embed the semantic query
    embedding = embed(semantic_query)

    # Call the Supabase RPC function
    result = supabase.rpc("match_attractions", {
        "query_embedding": embedding,
        "filter_city": city,
        "filter_state": state,
        "filter_country": country,
        "filter_price_level": price_level,
        "match_count": 100
    }).execute()

    return json.dumps(result.data)

@tool
def get_place_tool(city: str = None) -> str:
    """Check if a city already exists in the DB."""
    if not city:
        return json.dumps({"exists": False, "reason": "No city provided"})

    result = (
        supabase
        .table("place")
        .select("place_city")
        .eq("place_city", city)
        .execute()
    )

    if result.data:
        return "The place exists in the DB and can be queried for attractions."
    else:
        lambda_client = boto3.client("lambda", region_name=os.getenv("AWS_REGION"))
        lambda_client.invoke(
            FunctionName=os.getenv("SCRAPER_LAMBDA_NAME"),
            InvocationType="Event",
            Payload=json.dumps({"city": city}).encode()
        )
        return "City not found. It's being added and will be available in 10-15 minutes."

# Define Available tools
tools = [update_metadata_tool, search_attractions_tool, get_place_tool]
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
        "messages": [
            model_with_tools.invoke(state["messages"])
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def tool_node(state: dict):
    """Execute tool calls requested by the LLM"""
    tool_messages = []
    last_message = state["messages"][-1]
    new_metadata = state.get("metadata", {})

    # Build conversation context from recent messages
    conversation = [
        {"role": type(m).__name__.replace("Message", "").lower(), "content": m.content}
        for m in state["messages"]
        if hasattr(m, "content") and m.content
    ]

    # Get the latest user message
    user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    for tool_call in last_message.tool_calls:
        tool = tools_by_name.get(tool_call["name"])

        if not tool:
            continue

        # Override args for update_metadata_tool with actual state data
        if tool_call["name"] == "update_metadata_tool":
            args = {
                "data": {
                    "metadata": new_metadata,
                    "user_message": user_message,
                    "conversation": conversation
                }
            }
        elif tool_call["name"] == "search_attractions_tool":
            args = {
                "metadata": new_metadata
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
            ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": tool_messages, "metadata": new_metadata}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop"""
    if state["llm_calls"] >= 5:
        return END
    
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"
    
    # if isinstance(last_message, ToolMessage):
    #     return "llm_call"

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
    # Event is the JSON object passed as the payload
    user_input = event.get("prompt")
    session_id = event.get("session_id")

    if not user_input:
        return {"output": "No prompt provided."}
    if not session_id:
        return {"output": "Missing session_id."}
    
    # Check if session exists
    db_session = (
        supabase
        .table("sessions")
        .select("summary, updated_at, metadata")
        .eq("session_id", session_id)
        .execute()
    )

    # If no session found, create one. Otherwise, get summary, last update time, and metadata
    if not db_session.data or len(db_session.data) == 0:
        supabase.table("sessions").insert({"session_id": session_id}).execute()
        summary = ""
        last_summary_time = None
        metadata = {}
    else:
        summary = db_session.data[0].get("summary") or ""
        last_summary_time = db_session.data[0].get("updated_at")
        metadata = db_session.data[0].get("metadata")
    
    # Query DB for most recent 10 messages, order them, and arrange them as proper messages
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
    history_messages = [db_row_to_message(m) for m in recent_messages]
    
    # Add System Prompt and current metadata to state (messages)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Current metadata: {json.dumps(metadata)}")
    ]

    # If summary was found, add to state
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

    # Add past messages to state and then the user's prompt
    messages.extend(history_messages)
    messages.append(HumanMessage(content=user_input))

    # Call the agent and pass in the state
    result = agent.invoke({
        "messages": messages,
        "llm_calls": 0,
        "metadata": metadata
    })

    # Update metadata in DB
    updated_metadata = result.get("metadata", metadata)
    supabase.table("sessions").update({
        "metadata": updated_metadata
    }).eq("session_id", session_id).execute()
    
    # Get the last message in the agent's response (the output)
    final_message = result["messages"][-1]

    # Store user's prompt and agent's resposne
    supabase.table("messages").insert([
        { "session_id": session_id, "role": "user", "content": user_input },
        { "session_id": session_id, "role": "assistant", "content": final_message.content }
    ]).execute()

    # If there are at least 10 messages since last summary, update the session summary
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
#     print("\n" + "="*50)
#     print("TEST: No metadata — things to do in Sedona, Arizona (not in DB)")
#     print("="*50)
#     run_agent_with_metadata(
#         {},
#         "What are some things to do in Charlotte, North Carolina?"
#     )


# def run_agent_with_metadata(metadata: dict, prompt: str):
#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT),
#         SystemMessage(content=f"Current metadata: {json.dumps(metadata)}"),
#         HumanMessage(content=prompt)
#     ]

#     result = agent.invoke({
#         "messages": messages,
#         "llm_calls": 0,
#         "metadata": metadata
#     })

#     print(f"Metadata used: {json.dumps(metadata)}")
#     print(f"\nResponse:\n{result['messages'][-1].content}")
#     print(f"\nFinal metadata: {json.dumps(result.get('metadata', {}), indent=2)}")
#     print(f"\nFull message trace:")
#     for m in result["messages"]:
#         print(f"  {type(m).__name__}: {m.content[:200]}")


# if __name__ == "__main__":
#     debug_agent_test()

# endregion