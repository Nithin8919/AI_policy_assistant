"""
Result Reranking for Precision and Diversity
Uses cross-encoder models for accurate relevance scoring
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import sentence transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available, falling back to simpler reranking")


@dataclass
class RankedResult:
    """Result with reranking score"""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float
    final_score: float
    source: str


class Reranker:
    """
    Rerank search results for better precision
    
    Strategies:
    1. Cross-encoder reranking (if available)
    2. MMR (Maximal Marginal Relevance) for diversity
    3. Metadata boosting (recency, doc type priority)
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cross_encoder: bool = True
    ):
        """
        Initialize reranker
        
        Args:
            model_name: Cross-encoder model name
            use_cross_encoder: Whether to use cross-encoder (requires GPU for speed)
        """
        self.cross_encoder = None
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE
        
        if self.use_cross_encoder:
            try:
                self.cross_encoder = CrossEncoder(model_name)
                logger.info(f"Loaded cross-encoder: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}, falling back to simple reranking")
                self.use_cross_encoder = False
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
        diversity_weight: float = 0.3,
        metadata_boost: bool = True
    ) -> List[RankedResult]:
        """
        Rerank results with multiple strategies
        
        Args:
            query: Original query
            results: Search results to rerank
            top_k: Number of results to return
            diversity_weight: Weight for diversity (0-1, higher = more diverse)
            metadata_boost: Whether to boost based on metadata
        
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        # Step 1: Cross-encoder reranking (if available)
        if self.use_cross_encoder and self.cross_encoder:
            reranked = self._cross_encoder_rerank(query, results)
        else:
            # Fallback: use original scores
            reranked = [
                {
                    'result': r,
                    'original_score': r.get('score', 0.0),
                    'rerank_score': r.get('score', 0.0)
                }
                for r in results
            ]
        
        # Step 2: Apply metadata boosting
        if metadata_boost:
            reranked = self._apply_metadata_boost(reranked)
        
        # Step 3: Apply MMR for diversity
        if diversity_weight > 0:
            reranked = self._maximal_marginal_relevance(
                reranked, 
                top_k=top_k,
                lambda_param=1.0 - diversity_weight
            )
        else:
            # Just sort by rerank score
            reranked.sort(key=lambda x: x['final_score'], reverse=True)
            reranked = reranked[:top_k]
        
        # Convert to RankedResult objects
        ranked_results = []
        for item in reranked:
            result = item['result']
            ranked_results.append(RankedResult(
                chunk_id=result.get('chunk_id', ''),
                doc_id=result.get('doc_id', ''),
                content=result.get('content', '') or result.get('text', ''),
                metadata=result.get('metadata', {}),
                original_score=item['original_score'],
                rerank_score=item['rerank_score'],
                final_score=item['final_score'],
                source=result.get('source', 'unknown')
            ))
        
        logger.info(f"Reranked {len(results)} results to top {len(ranked_results)}")
        
        return ranked_results
    
    def _cross_encoder_rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank using cross-encoder model
        
        Cross-encoders are much more accurate than bi-encoders for reranking
        because they can attend to both query and document together
        """
        # Prepare query-document pairs
        pairs = []
        for result in results:
            content = result.get('content', '') or result.get('text', '')
            # Truncate long documents for speed
            content = content[:512] if len(content) > 512 else content
            pairs.append([query, content])
        
        # Get cross-encoder scores
        try:
            scores = self.cross_encoder.predict(pairs)
            
            reranked = []
            for result, score in zip(results, scores):
                reranked.append({
                    'result': result,
                    'original_score': result.get('score', 0.0),
                    'rerank_score': float(score),
                    'final_score': float(score)  # Will be adjusted by metadata boost
                })
            
            return reranked
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Fallback to original scores
            return [
                {
                    'result': r,
                    'original_score': r.get('score', 0.0),
                    'rerank_score': r.get('score', 0.0),
                    'final_score': r.get('score', 0.0)
                }
                for r in results
            ]
    
    def _apply_metadata_boost(self, reranked: List[Dict]) -> List[Dict]:
        """
        Boost scores based on metadata signals
        
        Boosts:
        - Recency: Newer documents get higher scores
        - Document type priority: Legal > GO > Schemes
        - Authority: Official sources > others
        """
        for item in reranked:
            result = item['result']
            metadata = result.get('metadata', {})
            boost_factor = 1.0
            
            # Recency boost (more recent = higher)
            year = metadata.get('year')
            if year and isinstance(year, (int, float)):
                if year >= 2023:
                    boost_factor *= 1.2
                elif year >= 2020:
                    boost_factor *= 1.1
                elif year < 2015:
                    boost_factor *= 0.9
            
            # Document type priority boost
            doc_type = metadata.get('doc_type', '').lower()
            if doc_type in ['act', 'rule']:
                boost_factor *= 1.15  # Legal documents are authoritative
            elif doc_type == 'government_order':
                boost_factor *= 1.10  # GOs are implementation
            elif doc_type == 'framework':
                boost_factor *= 0.95  # Frameworks are less authoritative
            
            # Priority boost (from metadata)
            priority = metadata.get('priority', '').lower()
            if priority == 'high':
                boost_factor *= 1.1
            elif priority == 'low':
                boost_factor *= 0.95
            
            # Apply boost
            item['final_score'] = item['rerank_score'] * boost_factor
        
        return reranked
    
    def _maximal_marginal_relevance(
        self,
        reranked: List[Dict],
        top_k: int,
        lambda_param: float = 0.7
    ) -> List[Dict]:
        """
        MMR: Balance relevance and diversity
        
        MMR = λ * relevance - (1-λ) * max_similarity_to_selected
        
        This ensures we don't return many similar documents
        """
        if not reranked:
            return []
        
        selected = []
        remaining = list(reranked)
        
        # Sort by relevance for first pick
        remaining.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Select first document (most relevant)
        selected.append(remaining.pop(0))
        
        # Select remaining documents using MMR
        while remaining and len(selected) < top_k:
            mmr_scores = []
            
            for item in remaining:
                # Relevance score (already computed)
                relevance = item['final_score']
                
                # Max similarity to already selected documents
                max_similarity = self._max_similarity_to_selected(
                    item['result'],
                    [s['result'] for s in selected]
                )
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((item, mmr))
            
            # Select document with highest MMR
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_item, best_mmr = mmr_scores[0]
            
            # Update final score with MMR
            best_item['final_score'] = best_mmr
            selected.append(best_item)
            remaining.remove(best_item)
        
        return selected
    
    def _max_similarity_to_selected(
        self,
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate maximum similarity between candidate and selected documents
        
        Simple implementation using word overlap (Jaccard similarity)
        For production, could use actual embeddings
        """
        if not selected:
            return 0.0
        
        candidate_text = candidate.get('content', '') or candidate.get('text', '')
        candidate_words = set(candidate_text.lower().split())
        
        max_sim = 0.0
        for selected_doc in selected:
            selected_text = selected_doc.get('content', '') or selected_doc.get('text', '')
            selected_words = set(selected_text.lower().split())
            
            if not candidate_words or not selected_words:
                continue
            
            # Jaccard similarity
            intersection = len(candidate_words & selected_words)
            union = len(candidate_words | selected_words)
            
            if union > 0:
                similarity = intersection / union
                max_sim = max(max_sim, similarity)
        
        return max_sim


# Convenience function
def create_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    use_cross_encoder: bool = False  # Disabled by default (slow without GPU)
) -> Reranker:
    """Create a Reranker instance"""
    return Reranker(model_name, use_cross_encoder)
