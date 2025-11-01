"""
Metadata-based filtering for precise document selection
"""
from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class KeywordFilter:
    """
    Advanced filtering using metadata and keywords
    
    Use cases:
    - Filter by document type (only GOs, only Acts)
    - Filter by year range (2020-2023)
    - Filter by district, scheme, section number
    - Keyword boosting (exact matches get higher scores)
    """
    
    def __init__(self):
        """Initialize keyword filter"""
        self.filters = []
    
    def filter_by_metadata(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter results based on metadata criteria
        
        Args:
            results: Search results to filter
            filters: Dictionary of metadata filters
                Examples:
                - {'doc_type': 'government_order'}
                - {'year': {'gte': 2020, 'lte': 2023}}
                - {'priority': 'high'}
        
        Returns:
            Filtered results
        """
        if not filters:
            return results
        
        filtered = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Check all filter conditions
            match = True
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Range filter (e.g., year >= 2020)
                    result_value = metadata.get(key)
                    if result_value is None:
                        match = False
                        break
                    
                    if 'gte' in value and result_value < value['gte']:
                        match = False
                        break
                    if 'gt' in value and result_value <= value['gt']:
                        match = False
                        break
                    if 'lte' in value and result_value > value['lte']:
                        match = False
                        break
                    if 'lt' in value and result_value >= value['lt']:
                        match = False
                        break
                    
                elif isinstance(value, list):
                    # Multiple allowed values (OR logic)
                    if metadata.get(key) not in value:
                        match = False
                        break
                
                else:
                    # Exact match
                    if metadata.get(key) != value:
                        match = False
                        break
            
            if match:
                filtered.append(result)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered)} based on metadata")
        
        return filtered
    
    def boost_exact_keyword_matches(
        self,
        query: str,
        results: List[Dict[str, Any]],
        boost_factor: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Boost results that contain exact keyword matches
        
        Args:
            query: Query string
            results: Search results
            boost_factor: Multiplier for exact matches
        
        Returns:
            Results with boosted scores
        """
        # Extract keywords from query
        keywords = self._extract_important_keywords(query)
        
        if not keywords:
            return results
        
        boosted_results = []
        
        for result in results:
            content = result.get('content', '') or result.get('text', '')
            content_lower = content.lower()
            
            # Count exact keyword matches
            matches = sum(1 for kw in keywords if kw in content_lower)
            
            if matches > 0:
                # Boost score based on number of matches
                boost = 1.0 + (boost_factor - 1.0) * (matches / len(keywords))
                result = result.copy()
                result['score'] = result.get('score', 0.0) * boost
                result['keyword_matches'] = matches
            
            boosted_results.append(result)
        
        # Re-sort by updated scores
        boosted_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        logger.info(f"Boosted {sum(1 for r in boosted_results if r.get('keyword_matches', 0) > 0)} results with keyword matches")
        
        return boosted_results
    
    def _extract_important_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        
        Important keywords:
        - GO numbers (GO.Ms.No.54)
        - Section numbers (Section 12(1)(c))
        - Scheme names (Nadu-Nedu, Amma Vodi)
        - Acronyms (RTE, SMC)
        """
        keywords = []
        query_lower = query.lower()
        
        # GO numbers
        go_patterns = [
            r'g\.?o\.?\s*(?:ms\.?|rt\.?)?\s*no\.?\s*\d+',
            r'government order\s+(?:ms\.?|rt\.?)?\s*no\.?\s*\d+'
        ]
        for pattern in go_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            keywords.extend(matches)
        
        # Section numbers
        section_patterns = [
            r'section\s+\d+(?:\(\d+\))?(?:\([a-z]\))?',
            r'article\s+\d+[a-z]?'
        ]
        for pattern in section_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            keywords.extend(matches)
        
        # Scheme names (common AP schemes)
        scheme_names = [
            'nadu-nedu', 'nadu nedu', 'nadunedu',
            'amma vodi', 'ammavodi', 'amma-vodi',
            'jagananna', 'gorumudda'
        ]
        for scheme in scheme_names:
            if scheme in query_lower:
                keywords.append(scheme)
        
        # Acronyms (3-5 capital letters)
        acronyms = re.findall(r'\b[A-Z]{3,5}\b', query)
        keywords.extend([a.lower() for a in acronyms])
        
        return list(set(keywords))  # Remove duplicates
    
    def filter_by_recency(
        self,
        results: List[Dict[str, Any]],
        max_age_years: Optional[int] = None,
        prefer_recent: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Filter or boost results based on recency
        
        Args:
            results: Search results
            max_age_years: Maximum age in years (filters out older docs)
            prefer_recent: Boost recent documents
        
        Returns:
            Filtered/boosted results
        """
        import datetime
        current_year = datetime.datetime.now().year
        
        filtered = []
        
        for result in results:
            metadata = result.get('metadata', {})
            year = metadata.get('year')
            
            if not year or not isinstance(year, (int, float)):
                filtered.append(result)
                continue
            
            # Filter by max age
            if max_age_years and (current_year - year) > max_age_years:
                continue
            
            # Boost recent documents
            if prefer_recent:
                result = result.copy()
                age = current_year - year
                
                if age <= 1:
                    boost = 1.3  # Very recent
                elif age <= 3:
                    boost = 1.15
                elif age <= 5:
                    boost = 1.05
                else:
                    boost = 1.0
                
                result['score'] = result.get('score', 0.0) * boost
            
            filtered.append(result)
        
        # Re-sort if boosted
        if prefer_recent:
            filtered.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        logger.info(f"Recency filter: {len(results)} → {len(filtered)} results")
        
        return filtered
    
    def deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        similarity_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate results
        
        Args:
            results: Search results
            similarity_threshold: Threshold for considering documents as duplicates
        
        Returns:
            Deduplicated results
        """
        if len(results) <= 1:
            return results
        
        deduplicated = []
        seen_content_hashes = set()
        
        for result in results:
            content = result.get('content', '') or result.get('text', '')
            
            # Simple hash-based deduplication
            # For production, could use MinHash or similar
            content_hash = hash(content[:500])  # First 500 chars
            
            if content_hash not in seen_content_hashes:
                deduplicated.append(result)
                seen_content_hashes.add(content_hash)
        
        logger.info(f"Deduplication: {len(results)} → {len(deduplicated)} results")
        
        return deduplicated


# Convenience function
def create_keyword_filter() -> KeywordFilter:
    """Create a KeywordFilter instance"""
    return KeywordFilter()


def filter_results(
    results: List[Dict[str, Any]],
    query: str,
    metadata_filters: Optional[Dict] = None,
    boost_exact_matches: bool = True,
    prefer_recent: bool = True,
    deduplicate: bool = True
) -> List[Dict[str, Any]]:
    """
    Apply all filtering and boosting in one call
    
    Args:
        results: Search results
        query: Original query
        metadata_filters: Metadata filtering criteria
        boost_exact_matches: Boost exact keyword matches
        prefer_recent: Boost recent documents
        deduplicate: Remove near-duplicates
    
    Returns:
        Filtered and boosted results
    """
    kf = KeywordFilter()
    
    # Apply metadata filters
    if metadata_filters:
        results = kf.filter_by_metadata(results, metadata_filters)
    
    # Boost exact matches
    if boost_exact_matches:
        results = kf.boost_exact_keyword_matches(query, results)
    
    # Apply recency preference
    if prefer_recent:
        results = kf.filter_by_recency(results, prefer_recent=True)
    
    # Deduplicate
    if deduplicate:
        results = kf.deduplicate_results(results)
    
    return results
