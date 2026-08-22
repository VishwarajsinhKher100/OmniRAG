import os
import tempfile
from typing import Optional
import pymupdf
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import embeddings
from store import _THREAD_METADATA, _THREAD_RETRIEVERS


def get_retriever(thread_id: Optional[str]):
    """Fetch the cached Chroma retriever for a specific thread."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Extract text from a PDF, chunk it, index into ChromaDB, and cache thread retriever."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    # Write raw bytes to a temporary PDF file for PyMuPDF extraction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Extract non-empty pages as LangChain Document objects
        pdf = pymupdf.open(temp_path)
        documents = []

        for page_number, page in enumerate(pdf):
            text = page.get_text("text")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": temp_path, "page": page_number + 1},
                    )
                )
        pdf.close()

        # Split documents into overlapping chunks for embedding
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        # Create thread-isolated Chroma collection and retriever
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"thread_{thread_id}",
        )
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # Store retriever and metadata in memory mapped by thread ID
        meta = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(documents),
            "chunks": len(chunks),
        }
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = meta

        return meta
    finally:
        # Clean up temporary disk file
        try:
            os.remove(temp_path)
        except OSError:
            pass