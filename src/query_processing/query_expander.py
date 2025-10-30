"""
Query Expansion Module

Generates query variations through:
- Synonym expansion
- Legal term relatives (e.g., "Act" -> ["legislation", "statute"])
- Metric rollups (district -> state aggregation)
- Scheme aliases
- Temporal variations
"""

import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryExpansion:
    """Expanded query variation with metadata"""
    text: str
    expansion_type: str
    weight: float  # Relevance weight for ranking
    source_terms: List[str]  # Original terms that were expanded


class QueryExpander:
    """
    Generates semantically similar query variations
    for improved recall in retrieval.
    """
    
    def __init__(self):
        """Initialize query expander with expansion dictionaries"""
        self.synonym_dict = self._build_synonym_dict()
        self.legal_terms = self._build_legal_terms()
        self.metric_hierarchy = self._build_metric_hierarchy()
        self.scheme_aliases = self._build_scheme_aliases()
        
        logger.info("QueryExpander initialized")
    
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """Build comprehensive synonym dictionary"""
        return {
            # Education terms
            "student": ["pupil", "learner", "child"],
            "teacher": ["educator", "instructor", "faculty"],
            "school": ["institution", "educational institution"],
            "education": ["learning", "schooling", "instruction"],
            
            # Administrative terms
            "district": ["mandal", "region", "area"],
            "state": ["province", "territory"],
            "government": ["administration", "authority"],
            
            # Policy terms
            "policy": ["guideline", "framework", "directive"],
            "scheme": ["programme", "program", "yojana", "project"],
            "implementation": ["execution", "deployment", "rollout"],
            
            # Data terms
            "enrollment": ["enrolment", "admission", "registration"],
            "dropout": ["attrition", "wastage"],
            "performance": ["achievement", "outcome", "result"],
            
            # Financial terms
            "budget": ["allocation", "funding", "expenditure"],
            "cost": ["expense", "spending"],
            
            # Quality terms
            "quality": ["standard", "excellence"],
            "improvement": ["enhancement", "development", "progress"],
        }
    
    def _build_legal_terms(self) -> Dict[str, List[str]]:
        """Build legal term relatives"""
        return {
            "act": ["legislation", "statute", "law", "enactment"],
            "section": ["provision", "clause", "article"],
            "rule": ["regulation", "guideline", "directive"],
            "amendment": ["modification", "change", "revision"],
            "rights": ["entitlement", "privilege", "provision"],
            "obligation": ["duty", "responsibility", "requirement"],
            "mandate": ["requirement", "directive", "order"],
            "judgment": ["ruling", "decision", "verdict", "order"],
            "petition": ["appeal", "writ", "application"],
        }
    
    def _build_metric_hierarchy(self) -> Dict[str, Dict]:
        """Build metric aggregation hierarchy"""
        return {
            "enrollment": {
                "rollup": ["total enrollment", "aggregate enrollment"],
                "breakdown": ["primary enrollment", "secondary enrollment", "male enrollment", "female enrollment"]
            },
            "ptr": {
                "rollup": ["average ptr", "overall ptr"],
                "breakdown": ["district ptr", "school-level ptr"]
            },
            "dropout": {
                "rollup": ["overall dropout rate", "total dropout"],
                "breakdown": ["grade-wise dropout", "gender-wise dropout"]
            },
            "budget": {
                "rollup": ["total budget", "overall allocation"],
                "breakdown": ["scheme-wise budget", "district-wise allocation"]
            }
        }
    
    def _build_scheme_aliases(self) -> Dict[str, List[str]]:
        """Build scheme name variations"""
        return {
            "nadu-nedu": ["nadu nedu", "nadunedu", "school infrastructure program"],
            "amma vodi": ["ammavodi", "jagananna amma vodi", "mother support scheme"],
            "gorumudda": ["goru mudda", "jagananna gorumudda", "mid day meal", "mdm"],
            "vidya deevena": ["vidyadeevena", "fee reimbursement", "education subsidy"],
            "ssa": ["sarva shiksha abhiyan", "universal education program"],
            "rmsa": ["rashtriya madhyamik shiksha abhiyan", "secondary education program"],
        }
    
    def expand(self, query: str, 
               max_expansions: int = 5,
               include_synonyms: bool = True,
               include_legal: bool = True,
               include_schemes: bool = True) -> List[QueryExpansion]:
        """
        Generate query expansions.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expansions to return
            include_synonyms: Include synonym expansions
            include_legal: Include legal term expansions
            include_schemes: Include scheme alias expansions
            
        Returns:
            List of query expansions with weights
        """
        expansions = []
        query_lower = query.lower()
        
        # Original query (highest weight)
        expansions.append(QueryExpansion(
            text=query,
            expansion_type="original",
            weight=1.0,
            source_terms=[]
        ))
        
        # Synonym expansion
        if include_synonyms:
            expansions.extend(self._expand_synonyms(query, query_lower))
        
        # Legal term expansion
        if include_legal:
            expansions.extend(self._expand_legal_terms(query, query_lower))
        
        # Scheme expansion
        if include_schemes:
            expansions.extend(self._expand_schemes(query, query_lower))
        
        # Metric expansion
        expansions.extend(self._expand_metrics(query, query_lower))
        
        # Sort by weight and limit
        expansions.sort(key=lambda x: x.weight, reverse=True)
        
        # Deduplicate
        seen = set()
        unique_expansions = []
        for exp in expansions:
            text_normalized = exp.text.lower().strip()
            if text_normalized not in seen:
                seen.add(text_normalized)
                unique_expansions.append(exp)
        
        logger.debug(f"Generated {len(unique_expansions)} query expansions")
        return unique_expansions[:max_expansions]
    
    def _expand_synonyms(self, query: str, query_lower: str) -> List[QueryExpansion]:
        """Generate synonym-based expansions"""
        expansions = []
        words = query.split()
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.synonym_dict:
                for synonym in self.synonym_dict[word_lower][:2]:  # Limit to 2 synonyms per word
                    # Replace word with synonym (preserve case)
                    expanded = query.replace(word, synonym)
                    if expanded != query:
                        expansions.append(QueryExpansion(
                            text=expanded,
                            expansion_type="synonym",
                            weight=0.8,
                            source_terms=[word]
                        ))
        
        return expansions
    
    def _expand_legal_terms(self, query: str, query_lower: str) -> List[QueryExpansion]:
        """Generate legal term expansions"""
        expansions = []
        
        for term, relatives in self.legal_terms.items():
            if term in query_lower:
                for relative in relatives[:2]:  # Limit relatives
                    expanded = re.sub(
                        r'\b' + re.escape(term) + r'\b',
                        relative,
                        query_lower
                    )
                    if expanded != query_lower:
                        expansions.append(QueryExpansion(
                            text=expanded,
                            expansion_type="legal_term",
                            weight=0.75,
                            source_terms=[term]
                        ))
        
        return expansions
    
    def _expand_schemes(self, query: str, query_lower: str) -> List[QueryExpansion]:
        """Generate scheme name expansions"""
        expansions = []
        
        for scheme, aliases in self.scheme_aliases.items():
            if scheme in query_lower:
                for alias in aliases[:2]:
                    expanded = query_lower.replace(scheme, alias)
                    if expanded != query_lower:
                        expansions.append(QueryExpansion(
                            text=expanded,
                            expansion_type="scheme_alias",
                            weight=0.9,  # High weight for scheme aliases
                            source_terms=[scheme]
                        ))
        
        return expansions
    
    def _expand_metrics(self, query: str, query_lower: str) -> List[QueryExpansion]:
        """Generate metric-based expansions (rollups/breakdowns)"""
        expansions = []
        
        for metric, hierarchy in self.metric_hierarchy.items():
            if metric in query_lower:
                # Add rollup terms
                for rollup in hierarchy.get("rollup", []):
                    expanded = query_lower.replace(metric, rollup)
                    expansions.append(QueryExpansion(
                        text=expanded,
                        expansion_type="metric_rollup",
                        weight=0.7,
                        source_terms=[metric]
                    ))
                
                # Add breakdown terms (lower weight)
                for breakdown in hierarchy.get("breakdown", [])[:2]:
                    expanded = query_lower.replace(metric, breakdown)
                    expansions.append(QueryExpansion(
                        text=expanded,
                        expansion_type="metric_breakdown",
                        weight=0.6,
                        source_terms=[metric]
                    ))
        
        return expansions
    
    def expand_with_context(self, query: str, context: Dict[str, any]) -> List[QueryExpansion]:
        """
        Context-aware expansion using entities and intent.
        
        Args:
            query: Original query
            context: Dictionary with entities, intent, etc.
            
        Returns:
            List of contextual query expansions
        """
        expansions = self.expand(query)
        
        # Add context-specific expansions
        if context.get("entities"):
            entities = context["entities"]
            
            # Expand with district context
            if entities.get("districts"):
                for district in entities["districts"][:2]:
                    expanded = f"{query} in {district}"
                    expansions.append(QueryExpansion(
                        text=expanded,
                        expansion_type="context_district",
                        weight=0.85,
                        source_terms=[district]
                    ))
            
            # Expand with temporal context
            if entities.get("dates"):
                for date in entities["dates"][:1]:
                    expanded = f"{query} for {date}"
                    expansions.append(QueryExpansion(
                        text=expanded,
                        expansion_type="context_temporal",
                        weight=0.85,
                        source_terms=[date]
                    ))
        
        return expansions


# Convenience functions for backwards compatibility
def expand_with_synonyms(query: str, synonym_dict: dict = None) -> list:
    """Generate query variations using synonyms (backwards compatible)"""
    expander = QueryExpander()
    if synonym_dict:
        expander.synonym_dict = synonym_dict
    
    expansions = expander.expand(query, include_legal=False, include_schemes=False)
    return [exp.text for exp in expansions]


def expand_legal_terms(query: str) -> str:
    """Expand legal terminology (backwards compatible)"""
    expander = QueryExpander()
    expansions = expander._expand_legal_terms(query, query.lower())
    return expansions[0].text if expansions else query


# Main API function
def process_query_expansion(query: str, 
                           context: Dict[str, any] = None,
                           max_expansions: int = 5) -> List[Dict[str, any]]:
    """
    Process query through expansion pipeline.
    
    Args:
        query: Original query
        context: Optional context (entities, intent)
        max_expansions: Maximum expansions to return
        
    Returns:
        List of query expansions with metadata
    """
    expander = QueryExpander()
    
    if context:
        expansions = expander.expand_with_context(query, context)
    else:
        expansions = expander.expand(query, max_expansions=max_expansions)
    
    # Convert to dict format
    return [
        {
            "text": exp.text,
            "type": exp.expansion_type,
            "weight": exp.weight,
            "source_terms": exp.source_terms
        }
        for exp in expansions
    ]
