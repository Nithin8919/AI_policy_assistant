"""Query endpoint with Multi-LLM integration"""
import time
import logging
from fastapi import APIRouter, HTTPException
from api.models.request import QueryRequest
from api.models.response import QueryResponse, Source

# Import our Multi-LLM pipeline (will be initialized when API keys are available)
import sys
import os
sys.path.append('/Users/nitin/Documents/AI policy Assistant')

try:
    from src.query_processing.qa_pipeline_multi_llm import QAPipeline, MultiLLMAnswerGenerator
    from src.agents.enhanced_router import EnhancedRouter
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    PIPELINE_AVAILABLE = False

router = APIRouter()
logger = logging.getLogger(__name__)

# Global pipeline instance (will be initialized on first request)
qa_pipeline = None
enhanced_router = None

def initialize_pipeline(llm_provider: str = "groq"):
    """Initialize QA pipeline with REAL EnhancedRouter and selected LLM provider"""
    global qa_pipeline, enhanced_router
    
    if qa_pipeline is not None:
        return qa_pipeline
    
    try:
        # Get credentials from environment
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "Missing Qdrant credentials. Set QDRANT_URL and QDRANT_API_KEY in .env"
            )
        
        # Initialize REAL EnhancedRouter with Qdrant
        logger.info(f"Initializing EnhancedRouter with Qdrant at {qdrant_url}")
        enhanced_router = EnhancedRouter(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        
        # Initialize QA Pipeline with real router
        qa_pipeline = QAPipeline(
            router=enhanced_router,
            llm_provider=llm_provider
        )
        logger.info(f"✅ QA Pipeline initialized with REAL retrieval and {llm_provider}")
        return qa_pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {str(e)}")

@router.post("/query")
async def query_documents(request: QueryRequest):
    """Process a query using Multi-LLM pipeline with REAL retrieval"""
    
    if not PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="QA Pipeline not available. Check system setup."
        )
    
    start_time = time.time()
    
    try:
        # Determine LLM provider from request or use default (groq is faster)
        llm_provider = getattr(request, 'llm_provider', 'groq')
        
        # Initialize pipeline with REAL retrieval
        pipeline = initialize_pipeline(llm_provider)
        
        # Process query with REAL document retrieval from Qdrant
        logger.info(f"Processing query with {llm_provider}: {request.query[:50]}...")
        response = pipeline.answer_query(
            query=request.query,
            mode=request.mode,
            top_k=10
        )
        
        # Convert sources to API format
        api_sources = []
        for source_id, source_info in response.citations.get('citation_details', {}).items():
            api_sources.append(Source(
                title=source_info.get('document', 'Unknown'),
                url=f"source://{source_info.get('chunk_id', 'unknown')}",
                excerpt=f"Section {source_info.get('section', 'N/A')} - Year: {source_info.get('year', 'N/A')}"
            ))
        
        processing_time = time.time() - start_time
        
        # Log detailed stats
        logger.info(f"""
Query Result:
  Time: {processing_time:.2f}s
  Confidence: {response.confidence_score}
  Sources: {response.citations.get('unique_sources_cited', 0)}
  Chunks: {response.retrieval_stats.get('chunks_retrieved', 0)}
  Agents: {', '.join(response.retrieval_stats.get('agents_used', []))}
  Provider: {response.llm_stats.get('provider', 'unknown')}
""")
        
        return QueryResponse(
            answer=response.answer,
            sources=api_sources,
            confidence=response.confidence_score
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        
        # Return error response instead of raising exception
        return QueryResponse(
            answer=f"Error processing query: {str(e)}. Please check API key configuration and Qdrant connection.",
            sources=[],
            confidence=0.0
        )

@router.get("/query/test")
async def test_query_endpoint():
    """Test endpoint to verify API functionality"""
    
    # Test without actual LLM call
    test_response = QueryResponse(
        answer="This is a test response. The API is working correctly. Configure API keys to enable full functionality.",
        sources=[
            Source(
                title="Test Document",
                url="test://example",
                excerpt="Test excerpt for API validation"
            )
        ],
        confidence=0.95
    )
    
    return test_response




