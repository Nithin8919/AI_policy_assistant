"""Metadata-based filtering"""
from typing import List, Dict

def filter_by_metadata(results: List[Dict], filters: Dict) -> List[Dict]:
    """Filter results based on metadata"""
    filtered = []
    for result in results:
        if all(result.get(key) == value for key, value in filters.items()):
            filtered.append(result)
    return filtered






