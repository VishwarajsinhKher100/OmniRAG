from typing import Optional
import requests
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from config import ALPHA_VANTAGE_API_KEY
from retrieval import get_retriever
from store import _THREAD_METADATA

search_tool = TavilySearch(
    max_results=3,
    topic="general",         
    search_depth="basic",     
    include_raw_content=False
)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers."""
    try:
        ops = {
            "add": first_num + second_num,
            "sub": first_num - second_num,
            "mul": first_num * second_num,
            "div": first_num / second_num if second_num != 0 else None,
        }
        if operation == "div" and second_num == 0:
            return {"error": "Division by zero is not allowed"}
        if operation not in ops:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": ops[operation],
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    r = requests.get(url, timeout=10)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve relevant information from the uploaded PDF for this chat thread."""
    retriever = get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]