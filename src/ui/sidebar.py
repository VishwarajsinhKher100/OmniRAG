from retrieval import ingest_pdf
from utils import reset_chat


def render_sidebar(st, thread_key, thread_docs, threads):
    selected_thread = None

    st.sidebar.title("LangGraph PDF Chatbot")
    st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

    if st.sidebar.button("New Chat", use_container_width=True):
        reset_chat(st)
        st.rerun()

    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.sidebar.success(
            f"Using `{latest_doc.get('filename')}` "
            f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
        )
    else:
        st.sidebar.info("No PDF indexed yet.")

    uploaded_pdf = st.sidebar.file_uploader(
        "Upload a PDF for this chat", type=["pdf"]
    )
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.sidebar.info(
                f"`{uploaded_pdf.name}` already processed for this chat."
            )
        else:
            with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(
                    label="✅ PDF indexed", state="complete", expanded=False
                )

    st.sidebar.subheader("Past conversations")
    if not threads:
        st.sidebar.write("No past conversations yet.")
    else:
        for thread_id in threads:
            if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
                selected_thread = thread_id

    return selected_thread