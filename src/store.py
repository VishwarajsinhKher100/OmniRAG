from typing import Any, Dict

# In-memory stores mapped by thread ID
_THREAD_RETRIEVERS: Dict[str, Any] = {}  # Dynamic retrievers/indexes per thread
_THREAD_METADATA: Dict[str, dict] = {}   # Session-specific configs and state data