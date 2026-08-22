import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import chatbot, retrieve_all_threads, thread_document_metadata
from retrieval import ingest_pdf
from utils import init_session_state, load_conversation, reset_chat

# 1. State Initialization
init_session_state(retrieve_all_threads)

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]


def render_sidebar(thread_key, thread_docs, threads):
    selected_thread = None
    st.sidebar.title("OmniRAG Chatbot")

    if st.sidebar.button("New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.sidebar.success(
            f"Using `{latest_doc.get('filename')}` "
            f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
        )
    else:
        st.sidebar.info("No PDF indexed yet.")

    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
        else:
            with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

    st.sidebar.subheader("Past conversations")
    if not threads:
        st.sidebar.write("No past conversations yet.")
    else:
        for t_id in threads:
            if st.sidebar.button(str(t_id), key=f"side-thread-{t_id}"):
                selected_thread = t_id

    return selected_thread


def render_chat_interface(thread_key):
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.text(message["content"])

    user_input = st.chat_input("Ask about your document or use tools")
    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.text(user_input)

        config = {
            "configurable": {"thread_id": thread_key},
            "metadata": {"thread_id": thread_key},
            "run_name": "chat_turn",
        }

        with st.chat_message("assistant"):
            status_holder = {"box": None}

            def ai_only_stream():
                for message_chunk, _ in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …", state="running", expanded=True
                            )

                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

            if status_holder["box"] is not None:
                status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

        doc_meta = thread_document_metadata(thread_key)
        if doc_meta:
            st.caption(
                f"Document indexed: {doc_meta.get('filename')} "
                f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
            )


# 2. Execution Order (Sidebar handling before Main View rendering)
selected_thread = render_sidebar(thread_key, thread_docs, threads)

if selected_thread and selected_thread != thread_key:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(chatbot, selected_thread)
    st.session_state["message_history"] = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages
    ]
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()

render_chat_interface(thread_key)