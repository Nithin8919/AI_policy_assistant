"""Precision, recall, citation accuracy"""

def calculate_precision(predicted: list, actual: list) -> float:
    """Calculate precision"""
    if not predicted:
        return 0.0
    correct = len(set(predicted) & set(actual))
    return correct / len(predicted)

def calculate_recall(predicted: list, actual: list) -> float:
    """Calculate recall"""
    if not actual:
        return 1.0
    correct = len(set(predicted) & set(actual))
    return correct / len(actual)


