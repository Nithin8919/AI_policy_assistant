"""Vector storage using Qdrant with document type collections"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        CollectionInfo, VectorParams, Distance, PointStruct,
        SearchRequest, Filter, FieldCondition, Range, MatchValue
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    # Mock classes for when Qdrant is not available
    class QdrantClient:
        pass
    
    class CollectionInfo:
        pass

from src.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentType(Enum):
    """Document type categories for collections"""
    LEGAL_DOCUMENTS = "legal_documents"  # acts, rules
    GOVERNMENT_ORDERS = "government_orders"
    JUDICIAL_DOCUMENTS = "judicial_documents"
    DATA_REPORTS = "data_reports"
    EXTERNAL_SOURCES = "external_sources"

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    vector_size: int = 384
    distance_metric: str = "Cosine"
    collection_prefix: str = "policy_assistant"

@dataclass
class SearchResult:
    """Search result from vector store"""
    chunk_id: str
    doc_id: str
    score: float
    payload: Dict[str, Any]
    content: str = ""
    
class VectorStore:
    """Qdrant vector store with document type collections"""
    
    # Document type mapping
    DOC_TYPE_MAPPING = {
        "acts": DocumentType.LEGAL_DOCUMENTS,
        "rules": DocumentType.LEGAL_DOCUMENTS,
        "government_orders": DocumentType.GOVERNMENT_ORDERS,
        "go": DocumentType.GOVERNMENT_ORDERS,
        "judicial": DocumentType.JUDICIAL_DOCUMENTS,
        "data_reports": DocumentType.DATA_REPORTS,
        "external_sources": DocumentType.EXTERNAL_SOURCES,
        "budget_finance": DocumentType.DATA_REPORTS,
        "frameworks": DocumentType.EXTERNAL_SOURCES
    }
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize vector store"""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required but not installed. Install with: pip install qdrant-client")
        
        self.config = config
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key
        )
        
        logger.info(f"Connected to Qdrant at {config.qdrant_url}")
        
    def get_collection_name(self, doc_type: DocumentType) -> str:
        """Get collection name for document type"""
        return f"{self.config.collection_prefix}_{doc_type.value}"
    
    def create_collections(self, recreate: bool = False) -> Dict[DocumentType, str]:
        """Create all collections for document types"""
        created_collections = {}
        
        for doc_type in DocumentType:
            collection_name = self.get_collection_name(doc_type)
            
            # Check if collection exists
            try:
                if recreate:
                    logger.info(f"Recreating collection: {collection_name}")
                    self.client.delete_collection(collection_name)
                
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=Distance.COSINE if self.config.distance_metric == "Cosine" else Distance.EUCLIDEAN
                    )
                )
                
                created_collections[doc_type] = collection_name
                logger.info(f"Created collection: {collection_name}")
                
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"Collection already exists: {collection_name}")
                    created_collections[doc_type] = collection_name
                else:
                    logger.error(f"Failed to create collection {collection_name}: {e}")
                    raise
        
        return created_collections
    
    def get_document_type(self, doc_type_str: str) -> DocumentType:
        """Map document type string to enum"""
        doc_type_clean = doc_type_str.lower().strip()
        
        # Direct mapping
        if doc_type_clean in self.DOC_TYPE_MAPPING:
            return self.DOC_TYPE_MAPPING[doc_type_clean]
        
        # Partial matching
        for key, doc_type in self.DOC_TYPE_MAPPING.items():
            if key in doc_type_clean or doc_type_clean in key:
                return doc_type
        
        # Default fallback
        logger.warning(f"Unknown document type '{doc_type_str}', defaulting to external_sources")
        return DocumentType.EXTERNAL_SOURCES
    
    def upsert_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[DocumentType, int]:
        """
        Upsert embeddings to appropriate collections based on document type
        
        Args:
            embeddings: List of embedding records with metadata
            batch_size: Batch size for upsert operations
            
        Returns:
            Dictionary mapping document type to number of embeddings inserted
        """
        # Group embeddings by document type
        grouped_embeddings = {}
        
        for embedding in embeddings:
            doc_type_str = embedding.get('doc_type', 'external_sources')
            doc_type = self.get_document_type(doc_type_str)
            
            if doc_type not in grouped_embeddings:
                grouped_embeddings[doc_type] = []
            
            grouped_embeddings[doc_type].append(embedding)
        
        insertion_counts = {}
        
        # Insert into each collection
        for doc_type, doc_embeddings in grouped_embeddings.items():
            collection_name = self.get_collection_name(doc_type)
            count = self._upsert_to_collection(collection_name, doc_embeddings, batch_size)
            insertion_counts[doc_type] = count
            logger.info(f"Inserted {count} embeddings into {collection_name}")
        
        return insertion_counts
    
    def _upsert_to_collection(
        self,
        collection_name: str,
        embeddings: List[Dict[str, Any]],
        batch_size: int
    ) -> int:
        """Upsert embeddings to a specific collection"""
        total_inserted = 0
        
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            points = []
            
            for j, embedding in enumerate(batch):
                point_id = i + j + 1  # Start from 1
                
                # Extract vector
                vector = embedding.get('embedding', [])
                if not vector:
                    logger.warning(f"Empty embedding for chunk {embedding.get('chunk_id', 'unknown')}")
                    continue
                
                # Prepare payload (exclude embedding vector)
                payload = {k: v for k, v in embedding.items() if k != 'embedding'}
                
                # Ensure required fields
                payload.setdefault('chunk_id', f'chunk_{point_id}')
                payload.setdefault('doc_id', '')
                payload.setdefault('content', '')
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
            
            if points:
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    total_inserted += len(points)
                    
                except Exception as e:
                    logger.error(f"Failed to upsert batch to {collection_name}: {e}")
        
        return total_inserted
    
    def search(
        self,
        query_vector: List[float],
        collection_names: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search across collections
        
        Args:
            query_vector: Query embedding vector
            collection_names: Specific collections to search (default: all)
            limit: Maximum number of results per collection
            score_threshold: Minimum similarity score
            filters: Metadata filters
            
        Returns:
            List of search results sorted by score
        """
        if collection_names is None:
            # Search all collections
            collection_names = [self.get_collection_name(dt) for dt in DocumentType]
        
        all_results = []
        
        for collection_name in collection_names:
            try:
                # Build filter
                qdrant_filter = None
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        if isinstance(value, (list, tuple)):
                            # Multiple values - use should conditions
                            should_conditions = [
                                FieldCondition(key=key, match=MatchValue(value=v))
                                for v in value
                            ]
                            conditions.extend(should_conditions)
                        elif isinstance(value, dict) and 'range' in value:
                            # Range filter
                            range_filter = value['range']
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    range=Range(
                                        gte=range_filter.get('gte'),
                                        gt=range_filter.get('gt'),
                                        lte=range_filter.get('lte'),
                                        lt=range_filter.get('lt')
                                    )
                                )
                            )
                        else:
                            # Exact match
                            conditions.append(
                                FieldCondition(key=key, match=MatchValue(value=value))
                            )
                    
                    if conditions:
                        qdrant_filter = Filter(must=conditions)
                
                # Perform search
                search_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=qdrant_filter
                )
                
                # Convert to SearchResult objects
                for result in search_results:
                    search_result = SearchResult(
                        chunk_id=result.payload.get('chunk_id', ''),
                        doc_id=result.payload.get('doc_id', ''),
                        score=result.score,
                        payload=result.payload,
                        content=result.payload.get('content', '')
                    )
                    all_results.append(search_result)
                
            except Exception as e:
                logger.error(f"Search failed for collection {collection_name}: {e}")
        
        # Sort by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:limit]
    
    def search_by_document_type(
        self,
        query_vector: List[float],
        doc_types: List[str],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search specific document types"""
        collection_names = []
        
        for doc_type_str in doc_types:
            doc_type = self.get_document_type(doc_type_str)
            collection_name = self.get_collection_name(doc_type)
            if collection_name not in collection_names:
                collection_names.append(collection_name)
        
        return self.search(
            query_vector=query_vector,
            collection_names=collection_names,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters
        )
    
    def get_collection_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all collections"""
        info = {}
        
        for doc_type in DocumentType:
            collection_name = self.get_collection_name(doc_type)
            
            try:
                collection_info = self.client.get_collection(collection_name)
                info[collection_name] = {
                    'document_type': doc_type.value,
                    'points_count': collection_info.points_count,
                    'vectors_count': collection_info.vectors_count,
                    'status': collection_info.status
                }
            except Exception as e:
                info[collection_name] = {
                    'document_type': doc_type.value,
                    'error': str(e)
                }
        
        return info
    
    def delete_collection(self, doc_type: DocumentType) -> bool:
        """Delete a collection"""
        collection_name = self.get_collection_name(doc_type)
        
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of Qdrant connection and collections"""
        try:
            # Check client connection
            collections = self.client.get_collections()
            
            health_info = {
                'qdrant_connected': True,
                'total_collections': len(collections.collections),
                'collections': self.get_collection_info()
            }
            
            return health_info
            
        except Exception as e:
            return {
                'qdrant_connected': False,
                'error': str(e)
            }

def create_vector_store_from_config(config_dict: Dict[str, Any]) -> VectorStore:
    """Create vector store from configuration dictionary"""
    config = VectorStoreConfig(**config_dict)
    return VectorStore(config)