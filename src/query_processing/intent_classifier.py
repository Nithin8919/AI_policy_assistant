"""Query intent classification"""

def classify_intent(query: str) -> dict:
    """Classify query intent"""
    intents = {
        "primary": "unknown",
        "secondary": []
    }
    
    query_lower = query.lower()
    
    # Classify based on keywords
    if any(word in query_lower for word in ["what", "explain", "define"]):
        intents["primary"] = "definition"
    elif any(word in query_lower for word in ["how", "procedure", "process"]):
        intents["primary"] = "procedure"
    elif any(word in query_lower for word in ["list", "show", "get all"]):
        intents["primary"] = "listing"
    elif any(word in query_lower for word in ["why", "reason", "because"]):
        intents["primary"] = "explanation"
    
    return intents


