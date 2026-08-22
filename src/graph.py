from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import llm
from database import checkpointer
from store import _THREAD_METADATA, _THREAD_RETRIEVERS
from tools import tools


class ChatState(TypedDict):
    # Appends incoming messages to conversation history
    messages: Annotated[list[BaseMessage], add_messages]


def retrieve_all_threads():
    """Extract all unique thread IDs from SQLite checkpoints."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    """Check if a vector retriever is loaded for this thread."""
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    """Get metadata for the thread's uploaded document."""
    return _THREAD_METADATA.get(str(thread_id), {})


# Bind tool registry to LLM and create tool runner node
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def chat_node(state: ChatState, config=None):
    """Inject thread-aware system prompt and generate model response."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


# Define StateGraph workflow with cyclical tool calling
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# Compile graph with persistent SQLite state checkpointer
chatbot = graph.compile(checkpointer=checkpointer)