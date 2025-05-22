import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("vector_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1")