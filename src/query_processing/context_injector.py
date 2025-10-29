"""Inject conversation context into queries"""

def inject_context(query: str, conversation_history: list) -> str:
    """Add context from previous messages"""
    if not conversation_history:
        return query
    
    context = "Previous conversation: " + " ".join(conversation_history[-3:])
    return f"{context}\n\nCurrent query: {query}"


