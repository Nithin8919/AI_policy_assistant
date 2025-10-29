"""Efficient batch processing for embeddings"""
from typing import List
from .embedder import Embedder

class BatchProcessor:
    """Process embeddings in batches"""
    
    def __init__(self, batch_size: int = 32):
        self.embedder = Embedder()
        self.batch_size = batch_size
    
    def process(self, texts: List[str]) -> List[List[float]]:
        """Process texts in batches"""
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.embedder.embed_batch(batch)
            results.extend(embeddings)
        return results

