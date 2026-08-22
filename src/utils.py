import uuid
import streamlit as st


def generate_thread_id():
    """Generate a unique conversation thread identifier."""
    return str(uuid.uuid4())


def add_thread(thread_id):
    """Add a thread ID to session state if it does not exist."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Start a fresh chat session with a new thread ID and clear messages."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def load_conversation(chatbot_instance, thread_id: str):
    """Fetch stored message history for a specific thread from the checkpointer."""
    state = chatbot_instance.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def init_session_state(retrieve_all_threads_fn):
    """Initialize default Streamlit session state variables on app load."""
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = retrieve_all_threads_fn()

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = generate_thread_id()

    if "ingested_docs" not in st.session_state:
        st.session_state["ingested_docs"] = {}

    add_thread(st.session_state["thread_id"])