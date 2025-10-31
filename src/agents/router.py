"""Query routing to appropriate agents"""
from .legal_agent import LegalAgent
from .go_agent import GOAgent
from .judicial_agent import JudicialAgent
from .data_agent import DataAgent
from .internet_agent import InternetAgent

class Router:
    """Route queries to appropriate agents"""
    
    def __init__(self):
        self.agents = {
            "legal": LegalAgent(),
            "go": GOAgent(),
            "judicial": JudicialAgent(),
            "data": DataAgent(),
            "internet": InternetAgent()
        }
    
    def select_agents(self, query: str, intent: dict) -> list:
        """Select appropriate agents based on query and intent"""
        selected = []
        
        # Simple keyword-based routing
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["act", "rule", "law"]):
            selected.append(self.agents["legal"])
        
        if any(word in query_lower for word in ["go", "government order", "order"]):
            selected.append(self.agents["go"])
        
        if any(word in query_lower for word in ["case", "judgment", "court"]):
            selected.append(self.agents["judicial"])
        
        if any(word in query_lower for word in ["ratio", "metric", "statistics", "data"]):
            selected.append(self.agents["data"])
        
        return selected or [self.agents["data"]]  # Default to data agent






