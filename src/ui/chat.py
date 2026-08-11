from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph.builder import chatbot
from graph.helpers import thread_document_metadata


def render_chat_interface(st, thread_key):
    st.title("Multi Utility Chatbot")

    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.text(message["content"])

    user_input = st.chat_input("Ask about your document or use tools")

    if user_input:
        st.session_state["message_history"].append(
            {"role": "user", "content": user_input}
        )
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
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )

        doc_meta = thread_document_metadata(thread_key)
        if doc_meta:
            st.caption(
                f"Document indexed: {doc_meta.get('filename')} "
                f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
            )

    st.divider()