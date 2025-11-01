"""
Retrieval Module - SOTA Hybrid Retrieval System

Components:
- HybridRetriever: Combines vector search + BM25 for best precision
- Reranker: Cross-encoder reranking + MMR for diversity
- BridgeTableLookup: Relationship-aware retrieval (GO supersession, etc.)
- KeywordFilter: Metadata filtering and keyword boosting

Usage:
    from src.retrieval import HybridRetriever, Reranker, BridgeTableLookup
    
    # Create hybrid retriever
    retriever = HybridRetriever(vector_store, corpus_documents)
    
    # Search with hybrid approach
    results = retriever.search(query, query_vector, collections)
    
    # Rerank for precision
    reranker = Reranker()
    reranked = reranker.rerank(query, results)
    
    # Enhance with bridge table
    bridge = BridgeTableLookup()
    context = bridge.enhance_query_with_context(query, entities)
"""

from .hybrid_retriever import (
    HybridRetriever,
    BM25Scorer,
    SearchResult,
    create_hybrid_retriever
)

from .reranker import (
    Reranker,
    RankedResult,
    create_reranker
)

from .bridge_lookup import (
    BridgeTableLookup,
    create_bridge_lookup
)

from .keyword_filter import (
    KeywordFilter,
    create_keyword_filter,
    filter_results
)

__all__ = [
    # Hybrid retrieval
    'HybridRetriever',
    'BM25Scorer',
    'SearchResult',
    'create_hybrid_retriever',
    
    # Reranking
    'Reranker',
    'RankedResult',
    'create_reranker',
    
    # Bridge table
    'BridgeTableLookup',
    'create_bridge_lookup',
    
    # Filtering
    'KeywordFilter',
    'create_keyword_filter',
    'filter_results',
]
