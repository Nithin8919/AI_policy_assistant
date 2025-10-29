"""Base agent class"""
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, collection: str):
        self.name = name
        self.collection = collection
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> list:
        """Retrieve relevant documents"""
        pass
    
    @abstractmethod
    def rank(self, results: list, query: str) -> list:
        """Rank results by relevance"""
        pass


