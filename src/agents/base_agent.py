"""Enhanced Base Agent with Vector Database Integration"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """Enhanced base class for all agents with vector database capabilities"""
    
    def __init__(self, name: str, doc_type: DocumentType, qdrant_url: str, qdrant_api_key: str):
        self.name = name
        self.doc_type = doc_type
        
        # Initialize vector store and embedder
        self.config = VectorStoreConfig(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        self.vector_store = VectorStore(self.config)
        self.embedder = Embedder()
        
        logger.info(f"Initialized {name} with doc_type: {doc_type.value}")
    
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using vector search"""
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            embedding_result = self.embedder.embed_single(query)
            query_embedding = embedding_result.embedding
            
            # Search in the agent's collection
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                doc_type=self.doc_type,
                limit=top_k
            )
            
            # Convert to standardized format
            results = []
            for result in search_results:
                chunk = {
                    'chunk_id': result.chunk_id,
                    'doc_id': result.doc_id,
                    'text': result.content,
                    'score': result.score,
                    'metadata': result.payload,
                    'agent': self.name,
                    'doc_type': self.doc_type.value
                }
                results.append(chunk)
            
            # Apply agent-specific ranking
            ranked_results = self.rank(results, query)
            
            processing_time = time.time() - start_time
            
            logger.info(f"{self.name} retrieved {len(ranked_results)} results in {processing_time:.3f}s")
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error in {self.name} retrieval: {str(e)}")
            return []
    
    @abstractmethod
    def rank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank results by relevance (agent-specific implementation)"""
        pass
    
    def get_specialized_features(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent-specific features from chunk (optional override)"""
        return {}
    
    def filter_results(self, results: List[Dict[str, Any]], **filters) -> List[Dict[str, Any]]:
        """Apply agent-specific filters (optional override)"""
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and collection info"""
        try:
            collection_name = self.vector_store.get_collection_name(self.doc_type)
            collection_info = self.vector_store.client.get_collection(collection_name)
            
            return {
                'name': self.name,
                'doc_type': self.doc_type.value,
                'collection': collection_name,
                'embeddings_count': getattr(collection_info, 'points_count', 0),
                'status': 'operational'
            }
        except Exception as e:
            return {
                'name': self.name,
                'doc_type': self.doc_type.value,
                'status': 'error',
                'error': str(e)
            }




