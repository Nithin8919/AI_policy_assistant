"""Response models"""
from pydantic import BaseModel

class Source(BaseModel):
    title: str
    url: str
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float = 0.0


