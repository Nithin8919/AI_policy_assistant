"""Input validation"""

def validate_query(query: str) -> bool:
    """Validate query input"""
    if not query or len(query.strip()) == 0:
        return False
    if len(query) > 1000:
        return False
    return True

def validate_mode(mode: str) -> bool:
    """Validate mode selection"""
    valid_modes = ["normal", "brainstorming", "pro"]
    return mode in valid_modes






