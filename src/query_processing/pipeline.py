"""Orchestrate query processing stages"""
from .normalizer import normalize_query
from .entity_extractor import extract_entities
from .intent_classifier import classify_intent
from .query_expander import expand_with_synonyms

def process_query(query: str, dictionaries: dict, conversation_history: list = None) -> dict:
    """Process query through all stages"""
    result = {
        "original": query,
        "normalized": None,
        "entities": None,
        "intent": None,
        "expanded": None
    }
    
    # Stage 1A: Normalization
    result["normalized"] = normalize_query(query, dictionaries)
    
    # Stage 1B: Entity extraction
    result["entities"] = extract_entities(query, dictionaries)
    
    # Stage 1C: Intent classification
    result["intent"] = classify_intent(query)
    
    # Stage 1D: Query expansion
    result["expanded"] = expand_with_synonyms(query, dictionaries.get("synonyms", {}))
    
    return result


