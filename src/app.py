import streamlit as st
from langchain_core.messages import HumanMessage

from graph.helpers import retrieve_all_threads
from ui.chat import render_chat_interface
from ui.sidebar import render_sidebar
from utils import init_session_state, load_conversation

# 1. State Initialization
init_session_state(st, retrieve_all_threads)

# 2. Extract Active State Options
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]

# 3. Render Views
selected_thread = render_sidebar(st, thread_key, thread_docs, threads)
render_chat_interface(st, thread_key)

# 4. Handle Thread Switch Logic
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()