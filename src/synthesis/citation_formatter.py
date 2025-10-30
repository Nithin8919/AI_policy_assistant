"""Consistent citation style"""

def format_citation(source: dict) -> str:
    """Format citation in consistent style"""
    title = source.get("title", "")
    date = source.get("date", "")
    return f"{title} ({date})"

def format_citations(sources: list) -> str:
    """Format multiple citations"""
    return "\n".join([format_citation(source) for source in sources])




