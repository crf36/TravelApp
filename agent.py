from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timezone
import operator, os

# Only used when running locally
load_dotenv()

#############################################################################
# SETUP
#############################################################################
# region

app = BedrockAgentCoreApp()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Set up Supabase client
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(url, key)

# Define state dictionary
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

SYSTEM_PROMPT = (
    "You are a helpful travel assistant.\n"
    "Your job is to answer customer travel questions clearly and accurately.\n\n"
    "Rules:\n"
    "- Ask clarifying questions if the request is ambiguous.\n"
    "- Use tools only when they provide missing or external information.\n"
    "- If you do not know something and no tool can help, say so.\n"
    "- Do not fabricate facts.\n"
    "- Keep responses concise but informative."
)
# endregion

#############################################################################
# TOOLS
#############################################################################
# region

@tool
def webscrape() -> str:
    """Scrape the web for relevant information."""
    return "Webscraped data"

tools = [webscrape]
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

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        tool_messages.append(
            ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": tool_messages}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop"""
    if state["llm_calls"] >= 5:
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
    ["tool_node", END]
)

agent = builder.compile()
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
# endregion

#############################################################################
# AGENTCORE JSON ENTRYPOINT
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
        .select("summary, updated_at")
        .eq("session_id", session_id)
        .execute()
    )

    # If no session found, create one. Otherwise, get summary and last update time
    if not db_session.data or len(db_session.data) == 0:
        supabase.table("sessions").insert({"session_id": session_id}).execute()
        summary = ""
        last_summary_time = None
    else:
        summary = db_session.data[0].get("summary") or ""
        last_summary_time = db_session.data[0].get("updated_at")
    
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
    
    # Add System Prompt to state (messages)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

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
        "llm_calls": 0
    })
    
    # Get the last message in the agent's response (the output)
    final_message = result["messages"][-1]

    # Store user's prompt and agent's resposne
    supabase.table("messages").insert([
        { "session_id": session_id, "role": "user", "content": user_input },
        { "session_id": session_id, "role": "assistant", "content": final_message.content }
    ]).execute()

    # If there are at least 10 messages since last summary, update the session summary
    unsummarized_db_messages = (
        supabase.table("messages")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .gt("created_at", last_summary_time)
        .order("created_at", desc=False)
        .execute()
    )
    unsummarized_messages = [db_row_to_message(m) for m in unsummarized_db_messages.data]
    if len(unsummarized_messages) >= 10:
        update_session_summary(session_id, unsummarized_messages, summary)

    return {"output": final_message.content}
# endregion

#############################################################################
# APP INITIALIZER
#############################################################################

if __name__ == "__main__":
    app.run()

# For testing locally
# if __name__ == "__main__":
#     test_event = {
#         "session_id": "57076c76-ad4c-4124-8a80-f4c151366844",
#         "prompt": "Thank you for all your help so far!"
#     }

#     response = handler(test_event)
#     print(response)
