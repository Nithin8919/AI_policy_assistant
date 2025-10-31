"""Request models"""
from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str
    mode: str = "normal_qa"
    llm_provider: Optional[str] = "gemini"  # "gemini" or "groq"
    conversation_history: List = []
    top_k: Optional[int] = 10






