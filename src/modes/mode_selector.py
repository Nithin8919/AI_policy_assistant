"""Auto-detect or user override mode selection"""

def select_mode(query: str, user_override: str = None) -> str:
    """Select appropriate mode"""
    if user_override:
        return user_override
    
    query_lower = query.lower()
    
    # Auto-detect based on query characteristics
    if any(word in query_lower for word in ["explore", "brainstorm", "suggest"]):
        return "brainstorming"
    elif any(word in query_lower for word in ["deep", "comprehensive", "detailed"]):
        return "pro"
    else:
        return "normal"






