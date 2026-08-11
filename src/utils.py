import uuid
from graph.builder import chatbot


def generate_thread_id():
    return uuid.uuid4()


def reset_chat(st):
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st, thread_id)
    st.session_state["message_history"] = []


def add_thread(st, thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def init_session_state(st, retrieve_all_threads_fn):
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = generate_thread_id()

    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = retrieve_all_threads_fn()

    if "ingested_docs" not in st.session_state:
        st.session_state["ingested_docs"] = {}

    add_thread(st, st.session_state["thread_id"])