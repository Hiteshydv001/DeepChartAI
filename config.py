# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("PORT", 8000))
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    LLM_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001")  # Add this