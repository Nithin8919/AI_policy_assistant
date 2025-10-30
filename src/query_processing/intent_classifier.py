"""
Query Intent Classification Module

Multi-label intent classification supporting:
- Primary intent: factual_lookup, data_query, procedural, legal_interpretation, etc.
- Secondary intents: comparison, temporal, explanation
- Confidence scoring
- Context awareness
"""

import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Structured intent with confidence"""
    name: str
    confidence: float
    triggers: List[str]  # Keywords/patterns that triggered this intent
    metadata: Dict[str, any]


class QueryIntentClassifier:
    """
    Multi-label intent classifier using keyword patterns,
    entity presence, and structural cues.
    """
    
    def __init__(self):
        """Initialize intent classifier with intent definitions"""
        self.intent_definitions = self._build_intent_definitions()
        logger.info("QueryIntentClassifier initialized")
    
    def _build_intent_definitions(self) -> Dict[str, Dict]:
        """
        Build comprehensive intent definitions with patterns and metadata.
        """
        return {
            # PRIMARY INTENTS
            "factual_lookup": {
                "description": "Looking up specific facts or information",
                "patterns": [
                    r'\b(?:what|which|who|when|where)\s+(?:is|are|was|were)\b',
                    r'\b(?:show|display|get|find|tell)\s+(?:me)?\s+(?:the)?\b',
                    r'\b(?:details?|information|info)\s+(?:about|on|of)\b',
                ],
                "keywords": ["what is", "who is", "tell me", "show me", "details", "information"],
                "requires_entities": False,
                "vertical_hints": [],
            },
            
            "data_query": {
                "description": "Querying statistical data or metrics",
                "patterns": [
                    r'\b(?:how many|how much|total|count|number)\b',
                    r'\b(?:statistics|stats|data|figures|numbers)\b',
                    r'\b(?:enrollment|ptr|ger|ner|dropout|budget)\b',
                ],
                "keywords": ["how many", "statistics", "enrollment", "ptr", "data", "metrics", "budget"],
                "requires_entities": ["metrics", "dates"],
                "vertical_hints": ["Data_Reports"],
            },
            
            "procedural": {
                "description": "Asking about processes or procedures",
                "patterns": [
                    r'\b(?:how to|how do|how can)\b',
                    r'\b(?:process|procedure|steps|method|way)\b',
                    r'\b(?:apply|submit|register|enroll)\b',
                ],
                "keywords": ["how to", "process", "procedure", "steps", "apply", "how do"],
                "requires_entities": False,
                "vertical_hints": ["Government_Orders", "Policy"],
            },
            
            "legal_interpretation": {
                "description": "Asking about legal matters or interpretations",
                "patterns": [
                    r'\b(?:according to|under|as per)\s+(?:section|rule|act)\b',
                    r'\b(?:legal|law|legislation|regulation|provision)\b',
                    r'\b(?:rights?|obligations?|duties|mandates?)\b',
                ],
                "keywords": ["according to", "legal", "act", "section", "rule", "rights", "law"],
                "requires_entities": ["legal_references"],
                "vertical_hints": ["Legal", "Judicial"],
            },
            
            "policy_inquiry": {
                "description": "Asking about policies or guidelines",
                "patterns": [
                    r'\b(?:policy|guideline|framework|nep|npe)\b',
                    r'\b(?:recommendation|objective|goal|vision)\b',
                    r'\b(?:what does the policy|policy says?)\b',
                ],
                "keywords": ["policy", "nep", "guideline", "framework", "recommendation"],
                "requires_entities": False,
                "vertical_hints": ["Policy", "National"],
            },
            
            "scheme_inquiry": {
                "description": "Asking about government schemes",
                "patterns": [
                    r'\b(?:scheme|programme|program|yojana)\b',
                    r'\b(?:nadu-nedu|amma vodi|gorumudda|vidya deevena)\b',
                    r'\b(?:eligibility|benefits?|coverage|implementation)\b',
                ],
                "keywords": ["scheme", "nadu-nedu", "amma vodi", "eligibility", "benefits"],
                "requires_entities": ["schemes"],
                "vertical_hints": ["Schemes", "Government_Orders"],
            },
            
            "case_law_query": {
                "description": "Asking about court cases or judgments",
                "patterns": [
                    r'\b(?:case|judgment|ruling|verdict|decision)\b',
                    r'\b(?:court|high court|supreme court|tribunal)\b',
                    r'\b(?:petition|writ|appeal)\b',
                ],
                "keywords": ["case", "judgment", "court", "petition", "ruling"],
                "requires_entities": False,
                "vertical_hints": ["Judicial"],
            },
            
            "go_inquiry": {
                "description": "Asking about Government Orders",
                "patterns": [
                    r'\bG\.O\.(?:Ms|Rt|MS|RT)\b',
                    r'\b(?:government order|go number|circular|notification)\b',
                    r'\b(?:superseded|supersedes|amended)\b',
                ],
                "keywords": ["government order", "go", "circular", "notification", "superseded"],
                "requires_entities": ["go_numbers"],
                "vertical_hints": ["Government_Orders"],
            },
            
            "comparison": {
                "description": "Comparing entities or metrics",
                "patterns": [
                    r'\b(?:compare|comparison|versus|vs|difference between)\b',
                    r'\b(?:higher|lower|more|less|better|worse)\s+than\b',
                    r'\b(?:which\s+(?:district|school|scheme))\b',
                ],
                "keywords": ["compare", "versus", "difference", "higher than", "which district"],
                "requires_entities": False,
                "vertical_hints": [],
            },
            
            "temporal_query": {
                "description": "Time-based or trend queries",
                "patterns": [
                    r'\b(?:trend|over time|change|growth|decline)\b',
                    r'\b(?:year-over-year|yoy|historical|past)\b',
                    r'\b(?:from|between|during)\s+\d{4}\b',
                ],
                "keywords": ["trend", "over time", "historical", "change", "growth"],
                "requires_entities": ["dates"],
                "vertical_hints": ["Data_Reports"],
            },
            
            "explanation": {
                "description": "Asking for explanation or reasoning",
                "patterns": [
                    r'\b(?:why|reason|because|cause|explain)\b',
                    r'\b(?:rationale|justification|purpose|objective)\b',
                ],
                "keywords": ["why", "reason", "explain", "rationale", "purpose"],
                "requires_entities": False,
                "vertical_hints": [],
            },
            
            "listing": {
                "description": "Requesting a list of items",
                "patterns": [
                    r'\b(?:list|show all|get all|enumerate)\b',
                    r'\b(?:all|every)\s+(?:districts?|schemes?|schools?)\b',
                ],
                "keywords": ["list", "show all", "all districts", "all schemes"],
                "requires_entities": False,
                "vertical_hints": [],
            },
            
            "definition": {
                "description": "Asking for definition or meaning",
                "patterns": [
                    r'\b(?:what is|what are|what does|define|meaning of)\b',
                    r'\b(?:definition|means?|refers? to)\b',
                ],
                "keywords": ["what is", "define", "meaning", "definition"],
                "requires_entities": False,
                "vertical_hints": [],
            },
        }
    
    def classify(self, query: str, entities: Dict[str, List] = None) -> Dict[str, any]:
        """
        Classify query intent with multi-label support.
        
        Args:
            query: Query string
            entities: Optional pre-extracted entities for context
            
        Returns:
            Dictionary with primary intent, secondary intents, and metadata
        """
        query_lower = query.lower()
        
        # Score all intents
        intent_scores = []
        
        for intent_name, intent_def in self.intent_definitions.items():
            score, triggers = self._score_intent(query_lower, intent_def, entities)
            
            if score > 0:
                intent_scores.append((
                    intent_name,
                    score,
                    triggers,
                    intent_def
                ))
        
        # Sort by score
        intent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Determine primary and secondary intents
        primary_intent = None
        secondary_intents = []
        
        if intent_scores:
            primary_name, primary_score, primary_triggers, primary_def = intent_scores[0]
            primary_intent = Intent(
                name=primary_name,
                confidence=min(primary_score, 1.0),
                triggers=primary_triggers,
                metadata={
                    "description": primary_def["description"],
                    "vertical_hints": primary_def["vertical_hints"]
                }
            )
            
            # Add secondary intents (score > 0.3 and not primary)
            for intent_name, score, triggers, intent_def in intent_scores[1:]:
                if score >= 0.3:
                    secondary_intents.append(Intent(
                        name=intent_name,
                        confidence=min(score, 1.0),
                        triggers=triggers,
                        metadata={
                            "description": intent_def["description"],
                            "vertical_hints": intent_def["vertical_hints"]
                        }
                    ))
        
        # If no intent detected, default to factual_lookup
        if not primary_intent:
            primary_intent = Intent(
                name="factual_lookup",
                confidence=0.5,
                triggers=["default"],
                metadata={
                    "description": "General factual lookup",
                    "vertical_hints": []
                }
            )
        
        result = {
            "primary": asdict(primary_intent),
            "secondary": [asdict(intent) for intent in secondary_intents],
            "query_type": self._determine_query_type(query_lower),
            "complexity": self._estimate_complexity(query_lower, entities),
            "suggested_verticals": self._suggest_verticals(primary_intent, secondary_intents, entities)
        }
        
        logger.debug(f"Classified intent: primary={primary_intent.name}, secondary={[i.name for i in secondary_intents]}")
        return result
    
    def _score_intent(self, query_lower: str, intent_def: Dict, 
                     entities: Dict[str, List] = None) -> Tuple[float, List[str]]:
        """
        Score an intent based on patterns, keywords, and entities.
        
        Returns:
            Tuple of (score, triggers)
        """
        score = 0.0
        triggers = []
        
        # Pattern matching (highest weight)
        for pattern in intent_def["patterns"]:
            if re.search(pattern, query_lower):
                score += 0.4
                triggers.append(f"pattern:{pattern[:30]}")
        
        # Keyword matching
        for keyword in intent_def["keywords"]:
            if keyword in query_lower:
                score += 0.2
                triggers.append(f"keyword:{keyword}")
        
        # Entity presence check
        if entities and intent_def.get("requires_entities"):
            required = intent_def["requires_entities"]
            if isinstance(required, list):
                for entity_type in required:
                    if entities.get(entity_type):
                        score += 0.3
                        triggers.append(f"entity:{entity_type}")
                        break
        
        return score, triggers
    
    def _determine_query_type(self, query_lower: str) -> str:
        """Determine high-level query type"""
        question_words = ["what", "who", "when", "where", "why", "how", "which"]
        
        if any(query_lower.startswith(word) for word in question_words):
            return "question"
        elif any(word in query_lower for word in ["show", "list", "get", "find"]):
            return "request"
        elif "?" in query_lower:
            return "question"
        else:
            return "statement"
    
    def _estimate_complexity(self, query_lower: str, entities: Dict[str, List] = None) -> str:
        """Estimate query complexity"""
        # Count entities
        entity_count = 0
        if entities:
            entity_count = sum(len(v) for v in entities.values())
        
        # Count clauses (rough estimate)
        clause_count = query_lower.count(" and ") + query_lower.count(" or ") + 1
        
        # Word count
        word_count = len(query_lower.split())
        
        # Calculate complexity
        if word_count < 5 and entity_count < 2 and clause_count == 1:
            return "simple"
        elif word_count < 15 and entity_count < 4 and clause_count <= 2:
            return "moderate"
        else:
            return "complex"
    
    def _suggest_verticals(self, primary: Intent, secondary: List[Intent], 
                          entities: Dict[str, List] = None) -> List[str]:
        """Suggest relevant verticals based on intents and entities"""
        verticals = set()
        
        # Add from primary intent
        verticals.update(primary.metadata.get("vertical_hints", []))
        
        # Add from secondary intents
        for intent in secondary:
            verticals.update(intent.metadata.get("vertical_hints", []))
        
        # Add based on entities
        if entities:
            if entities.get("legal_references") or entities.get("acts"):
                verticals.add("Legal")
            if entities.get("go_numbers"):
                verticals.add("Government_Orders")
            if entities.get("schemes"):
                verticals.add("Schemes")
            if entities.get("metrics"):
                verticals.add("Data_Reports")
        
        # Default to all if empty
        if not verticals:
            verticals = {"Legal", "Government_Orders", "Judicial", "Data_Reports"}
        
        return sorted(list(verticals))


# Convenience function for backwards compatibility
def classify_intent(query: str) -> dict:
    """Classify query intent (backwards compatible)"""
    classifier = QueryIntentClassifier()
    result = classifier.classify(query)
    
    # Simpler format
    return {
        "primary": result["primary"]["name"],
        "secondary": [intent["name"] for intent in result["secondary"]],
        "confidence": result["primary"]["confidence"]
    }


# Main API function
def process_intent_classification(query: str, entities: Dict[str, List] = None) -> Dict[str, any]:
    """
    Process query through intent classification pipeline.
    
    Args:
        query: Query string
        entities: Optional pre-extracted entities
        
    Returns:
        Intent classification result
    """
    classifier = QueryIntentClassifier()
    return classifier.classify(query, entities)
