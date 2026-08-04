from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests

load_dotenv()

# model
model = init_chat_model("groq:llama-3.1-8b-instant")

# tool
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def weather(city: str) -> dict:
    """
    Fetch the current weather (temperature, humidity, and wind speed) for a given city.
    Only provide the city name as the input. ALWAYS use this tool for weather requests.
    """

    HEADERS={"User-Agent": "MCP-Crash-Course"}
    geo = requests.get("https://nominatim.openstreetmap.org/search",
                       params={"q": city, "format": "json", "limit": 1},
                       headers=HEADERS).json()
    
    if not geo:
        return "city not found"
    
    lat = geo[0]["lat"]
    lon = geo[0]["lon"]

    weather = requests.get("https://api.open-meteo.com/v1/forecast",
                           params={
                                "latitude": lat,
	                            "longitude": lon,
	                            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"}).json()["current"]
    
    return {
        "City": city.title(),
        "Temperature": f"{weather['temperature_2m']} C",
        "Humidity": f"{weather['relative_humidity_2m']} %",
        "Wind Speed": f"{weather['wind_speed_10m']} km/h"}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. "AAPL", "TSLA")
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=L9VBZEMQ547644DT"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator, weather]
model_with_tools = model.bind_tools(tools)

# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# Helper
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)