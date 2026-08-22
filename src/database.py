import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Connect to SQLite database (thread-safe for multi-threaded/async environments)
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# Persist graph state and conversation history across runs
checkpointer = SqliteSaver(conn=conn)