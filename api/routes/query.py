"""Query endpoint"""
from fastapi import APIRouter, HTTPException
from api.models.request import QueryRequest
from api.models.response import QueryResponse

router = APIRouter()

@router.post("/query")
async def query_documents(request: QueryRequest):
    """Process a query"""
    # Implementation
    return QueryResponse(answer="", sources=[])




