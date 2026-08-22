import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load API keys from .env
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Initialize LLM and embedding models
llm = ChatGroq(model="openai/gpt-oss-120b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")