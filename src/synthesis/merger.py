"""Deduplicate and organize by section"""

def merge_results(results: list) -> list:
    """Merge and deduplicate results"""
    seen = set()
    merged = []
    for result in results:
        key = result.get("id")
        if key and key not in seen:
            seen.add(key)
            merged.append(result)
    return merged

def organize_by_section(results: list) -> dict:
    """Organize results by document section"""
    sections = {}
    for result in results:
        section = result.get("section", "general")
        if section not in sections:
            sections[section] = []
        sections[section].append(result)
    return sections




