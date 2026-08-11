from typing import Any, Dict

# Per-thread retriever and metadata cache
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}