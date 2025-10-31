"""Assess answer quality"""

def score_confidence(answer: str, sources: list) -> float:
    """Score confidence in answer (0-1)"""
    # Simple heuristic
    if not sources:
        return 0.0
    if len(sources) >= 3:
        return 0.9
    if len(sources) == 2:
        return 0.7
    return 0.5






