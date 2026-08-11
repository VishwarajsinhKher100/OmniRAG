from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Model Initializations
llm = ChatGroq(model="openai/gpt-oss-120b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")