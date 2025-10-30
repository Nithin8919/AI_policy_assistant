"""Enhanced GO Agent - Government Orders & Policy Implementation Specialist"""
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent
from src.embeddings.vector_store import DocumentType

class GOAgent(BaseAgent):
    """Agent specialized in Government Orders and Policy Implementation"""
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        super().__init__("GO Agent", DocumentType.GOVERNMENT_ORDERS, qdrant_url, qdrant_api_key)
        
        # GO-specific patterns and keywords
        self.go_patterns = [
            r'G\.?O\.?\s*(?:Ms|MS|Rt)\.?\s*(?:No\.?)?\s*(\d+)',
            r'Circular\s+(?:No\.?)?\s*(\d+)',
            r'Notification\s+(?:No\.?)?\s*([A-Z0-9\-/]+)'
        ]
        
        self.go_keywords = {
            'high_priority': ['nadu nedu', 'amma vodi', 'jagananna', 'rte implementation', 'teacher transfer'],
            'medium_priority': ['implementation', 'guidelines', 'procedures', 'eligibility', 'scheme'],
            'administrative': ['order', 'circular', 'notification', 'memorandum', 'proceedings']
        }
        
        # Load supersession data if available
        self.supersession_chains = self._load_supersession_data()
    
    def _load_supersession_data(self) -> Dict[str, List[str]]:
        """Load GO supersession chains"""
        try:
            import csv
            from pathlib import Path
            
            supersession_file = Path("data/processed_verticals/go_supersession_chains.csv")
            supersession_data = {}
            
            if supersession_file.exists():
                with open(supersession_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        go_number = row['go_number']
                        supersedes = row['supersedes']
                        if supersedes:  # Not empty
                            if go_number not in supersession_data:
                                supersession_data[go_number] = []
                            supersession_data[go_number].append(supersedes)
            
            return supersession_data
        except Exception as e:
            logger.warning(f"Could not load supersession data: {e}")
            return {}
    
    def rank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank GO results with specialized GO ranking"""
        
        if not results:
            return results
        
        query_lower = query.lower()
        
        # Calculate specialized scores for each result
        for result in results:
            go_score = self._calculate_go_relevance_score(result, query_lower)
            # Combine with original vector similarity score
            result['go_score'] = go_score
            result['combined_score'] = (result['score'] * 0.7) + (go_score * 0.3)
        
        # Sort by combined score
        ranked_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
        
        return ranked_results
    
    def _calculate_go_relevance_score(self, result: Dict[str, Any], query_lower: str) -> float:
        """Calculate GO-specific relevance score"""
        
        text = result.get('text', '').lower()
        doc_id = result.get('doc_id', '').lower()
        score = 0.0
        
        # 1. GO number matching (highest priority)
        go_refs_in_query = self._extract_go_references(query_lower)
        go_refs_in_text = self._extract_go_references(text)
        go_refs_in_doc_id = self._extract_go_references(doc_id)
        
        all_go_refs = go_refs_in_text + go_refs_in_doc_id
        
        if go_refs_in_query and all_go_refs:
            matches = set(go_refs_in_query) & set(all_go_refs)
            if matches:
                score += 0.4  # High boost for exact GO number matches
        
        # 2. High-priority scheme/program matching
        for keyword in self.go_keywords['high_priority']:
            if keyword in query_lower and keyword in text:
                score += 0.2
        
        # 3. Implementation keywords
        for keyword in self.go_keywords['medium_priority']:
            if keyword in query_lower and keyword in text:
                score += 0.1
        
        # 4. Administrative document type bonus
        for keyword in self.go_keywords['administrative']:
            if keyword in query_lower and keyword in text:
                score += 0.05
        
        # 5. Supersession chain relevance
        go_numbers = self._extract_go_references(text + doc_id)
        for go_num in go_numbers:
            if go_num in self.supersession_chains:
                score += 0.1  # Boost for GOs that supersede others
        
        # 6. Recent document bonus (if date can be extracted)
        date_score = self._calculate_recency_score(text, doc_id)
        score += date_score
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_go_references(self, text: str) -> List[str]:
        """Extract GO references from text"""
        
        references = []
        
        for pattern in self.go_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'g.o.' in pattern.lower() or 'go' in pattern.lower():
                    references.append(f"G.O.MS.No.{match}")
                else:
                    references.append(match)
        
        return references
    
    def _calculate_recency_score(self, text: str, doc_id: str) -> float:
        """Calculate recency score based on dates found in text/doc_id"""
        
        # Look for years in text and doc_id
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, text + doc_id)
        
        if not years:
            return 0.0
        
        # Get the most recent year
        latest_year = max(int(year) for year in years)
        current_year = datetime.now().year
        
        # Give higher score to more recent documents
        year_diff = current_year - latest_year
        
        if year_diff <= 1:
            return 0.15  # Very recent
        elif year_diff <= 3:
            return 0.1   # Recent
        elif year_diff <= 5:
            return 0.05  # Somewhat recent
        else:
            return 0.0   # Old
    
    def get_specialized_features(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract GO-specific features from chunk"""
        
        text = chunk.get('text', '')
        doc_id = chunk.get('doc_id', '')
        features = {}
        
        # Extract GO references
        features['go_references'] = self._extract_go_references(text + doc_id)
        
        # Identify GO type
        if 'circular' in doc_id.lower() or 'circular' in text.lower():
            features['go_type'] = 'Circular'
        elif 'notification' in doc_id.lower() or 'notification' in text.lower():
            features['go_type'] = 'Notification'
        elif 'proceedings' in doc_id.lower() or 'proceedings' in text.lower():
            features['go_type'] = 'Proceedings'
        else:
            features['go_type'] = 'Government Order'
        
        # Identify schemes mentioned
        schemes = []
        for scheme in self.go_keywords['high_priority']:
            if scheme in text.lower():
                schemes.append(scheme.title())
        features['schemes_mentioned'] = schemes
        
        # Check for supersession
        supersession_keywords = ['supersede', 'replace', 'substitute', 'cancel', 'revoke']
        features['mentions_supersession'] = any(
            keyword in text.lower() for keyword in supersession_keywords
        )
        
        # Extract implementation status keywords
        status_keywords = ['implemented', 'under implementation', 'approved', 'sanctioned', 'extended']
        found_status = [keyword for keyword in status_keywords if keyword in text.lower()]
        features['implementation_status'] = found_status
        
        return features
    
    def filter_results(self, results: List[Dict[str, Any]], **filters) -> List[Dict[str, Any]]:
        """Apply GO-specific filters"""
        
        filtered_results = results
        
        # Filter by GO number if specified
        if 'go_number' in filters:
            go_num = filters['go_number'].lower()
            filtered_results = [
                r for r in filtered_results 
                if go_num in (r.get('text', '') + r.get('doc_id', '')).lower()
            ]
        
        # Filter by scheme if specified
        if 'scheme' in filters:
            scheme = filters['scheme'].lower()
            filtered_results = [
                r for r in filtered_results 
                if scheme in r.get('text', '').lower()
            ]
        
        # Filter by GO type if specified
        if 'go_type' in filters:
            go_type = filters['go_type'].lower()
            filtered_results = [
                r for r in filtered_results 
                if go_type in r.get('doc_id', '').lower()
            ]
        
        # Filter by year if specified
        if 'year' in filters:
            year = str(filters['year'])
            filtered_results = [
                r for r in filtered_results 
                if year in (r.get('text', '') + r.get('doc_id', ''))
            ]
        
        return filtered_results
    
    def get_supersession_chain(self, go_number: str) -> Dict[str, Any]:
        """Get supersession chain for a GO number"""
        
        chain_info = {
            'go_number': go_number,
            'supersedes': self.supersession_chains.get(go_number, []),
            'superseded_by': []
        }
        
        # Find GOs that supersede this one
        for go, superseded_list in self.supersession_chains.items():
            if go_number in superseded_list:
                chain_info['superseded_by'].append(go)
        
        return chain_info
    
    def explain_go_context(self, chunk: Dict[str, Any]) -> str:
        """Provide GO context explanation for a chunk"""
        
        features = self.get_specialized_features(chunk)
        explanations = []
        
        # GO type context
        go_type = features.get('go_type', 'Government Order')
        explanations.append(f"Type: {go_type}")
        
        # GO references context
        go_refs = features.get('go_references', [])
        if go_refs:
            explanations.append(f"GO Numbers: {', '.join(go_refs[:2])}")
        
        # Schemes context
        schemes = features.get('schemes_mentioned', [])
        if schemes:
            explanations.append(f"Schemes: {', '.join(schemes[:2])}")
        
        # Implementation status
        status = features.get('implementation_status', [])
        if status:
            explanations.append(f"Status: {', '.join(status[:2])}")
        
        # Supersession context
        if features.get('mentions_supersession'):
            explanations.append("Contains supersession information")
        
        return " | ".join(explanations)




