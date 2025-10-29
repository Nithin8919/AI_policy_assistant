"""Validate external sources"""

def check_quality(url: str, content: str) -> bool:
    """Check quality of external source"""
    # Basic quality checks
    if len(content) < 100:
        return False
    if not content.strip():
        return False
    return True


