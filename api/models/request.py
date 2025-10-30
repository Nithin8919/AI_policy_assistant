"""Request models"""
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    mode: str = "normal"
    conversation_history: list = []




