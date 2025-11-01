"""
Hybrid Retrieval: Combine vector search + keyword/BM25 search for better precision
SOTA approach: Vector captures semantics, BM25 captures exact matches
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result"""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'vector', 'keyword', or 'hybrid'


class BM25Scorer:
    """BM25 (Best Matching 25) scoring for keyword search"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0.75 is standard)
        """
        self.k1 = k1
        self.b = b
        self.avg_doc_length = 0
        self.doc_freqs = {}  # Document frequencies for each term
        self.idf = {}  # Inverse document frequency
        self.doc_lengths = {}  # Document lengths
        
    def fit(self, documents: List[Dict[str, Any]]):
        """Build IDF scores from document corpus"""
        N = len(documents)
        term_doc_count = Counter()
        
        # Calculate document frequencies
        for doc in documents:
            chunk_id = doc.get('chunk_id', '')
            content = doc.get('content', '') or doc.get('text', '')
            tokens = self._tokenize(content)
            
            # Store document length
            self.doc_lengths[chunk_id] = len(tokens)
            
            # Count unique terms in this document
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_count[token] += 1
        
        # Calculate average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
        # Calculate IDF for each term
        for term, doc_count in term_doc_count.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((N - doc_count + 0.5) / (doc_count + 0.5) + 1)
        
        logger.info(f"BM25 fitted on {N} documents, avg_length={self.avg_doc_length:.1f}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase + split on non-alphanumeric"""
        # Remove special characters but keep numbers and basic punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def score(self, query: str, document: Dict[str, Any]) -> float:
        """Calculate BM25 score for a document given a query"""
        chunk_id = document.get('chunk_id', '')
        content = document.get('content', '') or document.get('text', '')
        
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(content)
        
        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)
        doc_length = self.doc_lengths.get(chunk_id, len(doc_tokens))
        
        score = 0.0
        
        for term in query_tokens:
            if term not in self.idf:
                continue  # Term not in corpus
            
            # Term frequency in document
            tf = doc_term_freqs.get(term, 0)
            
            # BM25 formula
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, documents: List[Dict[str, Any]], limit: int = 10) -> List[SearchResult]:
        """Search documents using BM25"""
        scored_docs = []
        
        for doc in documents:
            score = self.score(query, doc)
            if score > 0:
                scored_docs.append({
                    'doc': doc,
                    'score': score
                })
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Convert to SearchResult objects
        results = []
        for item in scored_docs[:limit]:
            doc = item['doc']
            results.append(SearchResult(
                chunk_id=doc.get('chunk_id', ''),
                doc_id=doc.get('doc_id', ''),
                content=doc.get('content', '') or doc.get('text', ''),
                metadata=doc.get('metadata', {}),
                score=item['score'],
                source='keyword'
            ))
        
        return results


class HybridRetriever:
    """
    Combines vector search (semantic) + BM25 (lexical) for best of both worlds
    
    Use cases:
    - Vector: "teacher qualifications" matches "educator credentials"
    - BM25: "Section 12(1)(c)" matches exactly "Section 12(1)(c)"
    """
    
    def __init__(self, vector_store, corpus_documents: Optional[List[Dict]] = None):
        """
        Args:
            vector_store: VectorStore instance for semantic search
            corpus_documents: Full corpus for BM25 fitting (optional, can fit later)
        """
        self.vector_store = vector_store
        self.bm25 = BM25Scorer()
        
        if corpus_documents:
            self.fit_bm25(corpus_documents)
        
        logger.info("HybridRetriever initialized")
    
    def fit_bm25(self, documents: List[Dict[str, Any]]):
        """Fit BM25 on document corpus"""
        self.bm25.fit(documents)
        logger.info(f"BM25 fitted on {len(documents)} documents")
    
    def search(
        self,
        query: str,
        query_vector: List[float],
        collection_names: List[str],
        limit: int = 10,
        alpha: float = 0.7,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector and keyword approaches
        
        Args:
            query: Text query
            query_vector: Query embedding vector
            collection_names: Collections to search
            limit: Number of results to return
            alpha: Weight for vector search (1-alpha for keyword)
            vector_weight: Weight for combining vector scores
            keyword_weight: Weight for combining keyword scores
            filters: Metadata filters
            
        Returns:
            List of SearchResult objects, ranked by hybrid score
        """
        # Step 1: Vector search
        vector_results = self.vector_store.search(
            query_vector=query_vector,
            collection_names=collection_names,
            limit=limit * 2,  # Get more for merging
            filters=filters
        )
        
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # Convert to documents for BM25
        documents = []
        for result in vector_results:
            documents.append({
                'chunk_id': result.chunk_id,
                'doc_id': result.doc_id,
                'content': result.content,
                'text': result.content,
                'metadata': result.payload
            })
        
        # Step 2: BM25 search on the same documents
        if documents and hasattr(self.bm25, 'idf') and self.bm25.idf:
            # BM25 already fitted, search directly
            keyword_results = self.bm25.search(query, documents, limit=limit * 2)
        else:
            # Fit BM25 on these documents first
            if documents:
                self.bm25.fit(documents)
                keyword_results = self.bm25.search(query, documents, limit=limit * 2)
            else:
                keyword_results = []
        
        logger.info(f"Keyword search returned {len(keyword_results)} results")
        
        # Step 3: Combine scores using RRF (Reciprocal Rank Fusion)
        combined_scores = self._reciprocal_rank_fusion(
            vector_results, 
            keyword_results,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        )
        
        # Step 4: Re-rank and return top results
        final_results = sorted(
            combined_scores,
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:limit]
        
        # Convert to SearchResult objects
        results = []
        for item in final_results:
            doc = item['doc']
            results.append(SearchResult(
                chunk_id=doc.chunk_id if hasattr(doc, 'chunk_id') else doc.get('chunk_id', ''),
                doc_id=doc.doc_id if hasattr(doc, 'doc_id') else doc.get('doc_id', ''),
                content=doc.content if hasattr(doc, 'content') else doc.get('content', ''),
                metadata=doc.payload if hasattr(doc, 'payload') else doc.get('metadata', {}),
                score=item['hybrid_score'],
                source='hybrid'
            ))
        
        logger.info(f"Hybrid search returning {len(results)} results")
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List,
        keyword_results: List,
        k: int = 60,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) for combining ranked lists
        
        RRF score = Σ (weight / (k + rank))
        
        This is more robust than score normalization
        """
        scores = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result.chunk_id
            rrf_score = vector_weight / (k + rank)
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    'doc': result,
                    'vector_score': result.score,
                    'keyword_score': 0.0,
                    'vector_rrf': rrf_score,
                    'keyword_rrf': 0.0,
                    'hybrid_score': rrf_score
                }
            else:
                scores[chunk_id]['vector_rrf'] = rrf_score
                scores[chunk_id]['hybrid_score'] += rrf_score
        
        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            chunk_id = result.chunk_id
            rrf_score = keyword_weight / (k + rank)
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    'doc': result,
                    'vector_score': 0.0,
                    'keyword_score': result.score,
                    'vector_rrf': 0.0,
                    'keyword_rrf': rrf_score,
                    'hybrid_score': rrf_score
                }
            else:
                scores[chunk_id]['keyword_rrf'] = rrf_score
                scores[chunk_id]['hybrid_score'] += rrf_score
        
        return list(scores.values())
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]


# Convenience function
def create_hybrid_retriever(vector_store, corpus_jsonl_path: Optional[str] = None):
    """
    Create and fit a hybrid retriever
    
    Args:
        vector_store: VectorStore instance
        corpus_jsonl_path: Path to JSONL file with all chunks (for BM25 fitting)
    
    Returns:
        HybridRetriever instance
    """
    documents = None
    
    if corpus_jsonl_path:
        import json
        from pathlib import Path
        
        documents = []
        corpus_path = Path(corpus_jsonl_path)
        
        if corpus_path.exists():
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk = json.loads(line.strip())
                    documents.append({
                        'chunk_id': chunk.get('chunk_id', ''),
                        'doc_id': chunk.get('doc_id', ''),
                        'content': chunk.get('text', ''),
                        'text': chunk.get('text', ''),
                        'metadata': chunk.get('metadata', {})
                    })
            
            logger.info(f"Loaded {len(documents)} documents for BM25 fitting")
    
    return HybridRetriever(vector_store, documents)
