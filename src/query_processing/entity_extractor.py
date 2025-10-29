"""Named entity extraction from queries"""
import json

def extract_entities(query: str, dictionaries: dict) -> dict:
    """Extract entities from query"""
    entities = {
        "districts": [],
        "schemes": [],
        "metrics": [],
        "dates": []
    }
    
    gazetteer = dictionaries.get("gazetteer", {})
    
    # Extract districts
    for district in gazetteer.get("districts", []):
        if district.lower() in query.lower():
            entities["districts"].append(district)
    
    # Extract schemes
    for scheme in gazetteer.get("schemes", []):
        if scheme.lower() in query.lower():
            entities["schemes"].append(scheme)
    
    return entities


