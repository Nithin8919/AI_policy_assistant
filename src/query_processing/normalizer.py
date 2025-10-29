"""Query normalization - spelling, acronym expansion"""

def normalize_spelling(text: str) -> str:
    """Correct common spelling errors"""
    # Implementation for spell checking
    return text

def expand_acronyms(text: str, acronyms_dict: dict) -> str:
    """Expand acronyms in text"""
    for acronym, expansions in acronyms_dict.items():
        text = text.replace(acronym, f"{acronym} ({expansions[0]})")
    return text

def normalize_query(query: str, dictionaries: dict) -> str:
    """Main normalization function"""
    normalized = normalize_spelling(query)
    normalized = expand_acronyms(normalized, dictionaries.get("acronyms", {}))
    return normalized


