"""Application settings and configuration"""
from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    """Application settings"""
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Application
    APP_ENV = os.getenv("APP_ENV", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Embedding Model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_CHECKPOINT_INTERVAL = int(os.getenv("EMBEDDING_CHECKPOINT_INTERVAL", "100"))
    
    # Vector Store Settings
    VECTOR_STORE_BATCH_SIZE = int(os.getenv("VECTOR_STORE_BATCH_SIZE", "100"))
    VECTOR_STORE_COLLECTION_PREFIX = os.getenv("VECTOR_STORE_COLLECTION_PREFIX", "policy_assistant")
    
    # Retrieval Settings
    MAX_CHUNKS_PER_AGENT = int(os.getenv("MAX_CHUNKS_PER_AGENT", "20"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "50"))
    SEARCH_SCORE_THRESHOLD = float(os.getenv("SEARCH_SCORE_THRESHOLD", "0.5"))

settings = Settings()

