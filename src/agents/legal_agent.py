"""Enhanced Legal Agent - Acts, Rules & Constitutional Framework Specialist"""
import re
from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent
from src.embeddings.vector_store import DocumentType

class LegalAgent(BaseAgent):
    """Agent specialized in Acts, Rules, and Legal Framework"""
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        super().__init__("Legal Agent", DocumentType.LEGAL_DOCUMENTS, qdrant_url, qdrant_api_key)
        
        # Legal-specific patterns and keywords
        self.section_patterns = [
            r'Section\s+(\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?)',
            r'Article\s+(\d+[A-Z]?)',
            r'Rule\s+(\d+[A-Z]?)',
            r'Chapter\s+([IVX]+|\d+)'
        ]
        
        self.legal_keywords = {
            'high_priority': ['rte', 'right to education', 'constitution', 'fundamental right', 'directive principle'],
            'medium_priority': ['act', 'rule', 'regulation', 'amendment', 'provision'],
            'legal_entities': ['section', 'article', 'chapter', 'schedule', 'part']
        }
    
    def rank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank legal results with specialized legal ranking"""
        
        if not results:
            return results
        
        query_lower = query.lower()
        
        # Calculate specialized scores for each result
        for result in results:
            legal_score = self._calculate_legal_relevance_score(result, query_lower)
            # Combine with original vector similarity score
            result['legal_score'] = legal_score
            result['combined_score'] = (result['score'] * 0.7) + (legal_score * 0.3)
        
        # Sort by combined score
        ranked_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
        
        return ranked_results
    
    def _calculate_legal_relevance_score(self, result: Dict[str, Any], query_lower: str) -> float:
        """Calculate legal-specific relevance score"""
        
        text = result.get('text', '').lower()
        score = 0.0
        
        # 1. Legal reference matching (highest priority)
        legal_refs_in_query = self._extract_legal_references(query_lower)
        legal_refs_in_text = self._extract_legal_references(text)
        
        if legal_refs_in_query and legal_refs_in_text:
            matches = set(legal_refs_in_query) & set(legal_refs_in_text)
            if matches:
                score += 0.4  # High boost for exact legal reference matches
        
        # 2. High-priority legal keyword matching
        for keyword in self.legal_keywords['high_priority']:
            if keyword in query_lower and keyword in text:
                score += 0.2
        
        # 3. Medium-priority legal keyword matching
        for keyword in self.legal_keywords['medium_priority']:
            if keyword in query_lower and keyword in text:
                score += 0.1
        
        # 4. Legal entity structure bonus
        for keyword in self.legal_keywords['legal_entities']:
            if keyword in query_lower and keyword in text:
                score += 0.05
        
        # 5. Document type specific bonuses
        doc_id = result.get('doc_id', '').lower()
        if 'rte' in query_lower and 'rte' in doc_id:
            score += 0.15
        elif 'constitution' in query_lower and 'constitution' in doc_id:
            score += 0.15
        elif 'education act' in query_lower and 'education' in doc_id and 'act' in doc_id:
            score += 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_legal_references(self, text: str) -> List[str]:
        """Extract legal references (sections, articles, etc.) from text"""
        
        references = []
        
        for pattern in self.section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend([f"Section {match}" for match in matches])
        
        return references
    
    def get_specialized_features(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract legal-specific features from chunk"""
        
        text = chunk.get('text', '')
        features = {}
        
        # Extract legal references
        features['legal_references'] = self._extract_legal_references(text)
        
        # Identify legal document type
        doc_id = chunk.get('doc_id', '').lower()
        if 'rte' in doc_id:
            features['legal_document_type'] = 'Right to Education Act'
        elif 'constitution' in doc_id:
            features['legal_document_type'] = 'Constitution'
        elif 'education act' in doc_id:
            features['legal_document_type'] = 'Education Act'
        else:
            features['legal_document_type'] = 'General Legal Document'
        
        # Check for amendments
        if re.search(r'amend|insert|substitute|omit', text, re.IGNORECASE):
            features['contains_amendments'] = True
        
        # Check for definitions
        if re.search(r'"[^"]+"\s+means', text, re.IGNORECASE):
            features['contains_definitions'] = True
        
        return features
    
    def filter_results(self, results: List[Dict[str, Any]], **filters) -> List[Dict[str, Any]]:
        """Apply legal-specific filters"""
        
        filtered_results = results
        
        # Filter by legal reference if specified
        if 'legal_reference' in filters:
            ref = filters['legal_reference'].lower()
            filtered_results = [
                r for r in filtered_results 
                if ref in r.get('text', '').lower()
            ]
        
        # Filter by document type if specified
        if 'document_type' in filters:
            doc_type = filters['document_type'].lower()
            filtered_results = [
                r for r in filtered_results 
                if doc_type in r.get('doc_id', '').lower()
            ]
        
        # Filter by amendment status if specified
        if 'has_amendments' in filters:
            if filters['has_amendments']:
                filtered_results = [
                    r for r in filtered_results 
                    if re.search(r'amend|insert|substitute|omit', r.get('text', ''), re.IGNORECASE)
                ]
        
        return filtered_results
    
    def explain_legal_context(self, chunk: Dict[str, Any]) -> str:
        """Provide legal context explanation for a chunk"""
        
        features = self.get_specialized_features(chunk)
        explanations = []
        
        # Document type context
        doc_type = features.get('legal_document_type', 'Legal Document')
        explanations.append(f"Source: {doc_type}")
        
        # Legal references context
        legal_refs = features.get('legal_references', [])
        if legal_refs:
            explanations.append(f"References: {', '.join(legal_refs[:3])}")
        
        # Amendment context
        if features.get('contains_amendments'):
            explanations.append("Contains amendments to existing provisions")
        
        # Definition context
        if features.get('contains_definitions'):
            explanations.append("Contains legal definitions")
        
        return " | ".join(explanations)






