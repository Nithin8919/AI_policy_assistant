"""Document management endpoint"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/documents")
async def list_documents():
    """List all documents"""
    return {"documents": []}




