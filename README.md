# **OmniRAG** : Multi-Utility Agentic RAG Chatbot

An intelligent, stateful assistant built with **LangGraph**, **Groq** (openai/gpt-oss-120b), **ChromaDB**, and **Streamlit**. The system provides multi-thread PDF Retrieval-Augmented Generation (RAG) alongside autonomous tools for web search, real-time stock quotes, and basic calculations, with full observability powered by **LangSmith**.

## **Features**

* **Thread-Isolated PDF RAG**: Upload PDFs bound strictly to active conversation thread IDs using isolated ChromaDB collections.

* **Autonomous Tool Calling**: Automatically delegates tasks to web search, Alpha Vantage stock price API, arithmetic calculator, or document search depending on context.

* **Persistent Conversation Memory**: SQLite checkpointer handles conversation persistence across reloads.

* **Full Observability & Tracing**: Seamless LangSmith integration to monitor agent runs, trace tool calls, analyze latency, and debug state transitions in real time.

* **Interactive Streamlit UI**: Complete user interface with sidebar thread management, PDF uploading, and real-time streaming tool status indicators.

## **Tech Stack**


| Component                    | Technology                      |
|------------------------------|---------------------------------|
| Frontend                     | Streamlit                       |
| Agent Framework              | LangGraph & LangChain           |
| Observability & Evaluation   | LangSmith                       |
| LLM Provider                 | Groq (openai/gpt-oss-120b)      |
| Vector DB                    | ChromaDB (langchain-chroma)     |
| Embedding Model              | HuggingFace (all-MiniLM-L6-v2)  |
| State Persistence            | SQLite (SqliteSaver)            |

## **Setup Instructions**

### 1. Clone the Repository
```bash
git clone https://github.com/VishwarajsinhKher100/OmniRAG.git
```

### 2. Set Up Virtual Environment
```bash
uv venv
```

#### Activate the virtual environment:

```bash
.venv\Scripts\activate     # On Windows
# OR
source .venv/bin/activate  # On Linux/macOS:
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Configure Environment Variables

Create a .env file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here

# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=OmniRAG
```

### 5. Running the Application

Launch the Streamlit web application:

```bash
streamlit run app/frontend.py
```
