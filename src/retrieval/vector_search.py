"""Semantic similarity search"""
from typing import List
import numpy as np

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity"""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    return np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np))

def search_similar(query_vector: List[float], document_vectors: List[List[float]], top_k: int = 10) -> List[dict]:
    """Search for similar documents"""
    similarities = []
    for i, doc_vec in enumerate(document_vectors):
        sim = cosine_similarity(query_vector, doc_vec)
        similarities.append({"index": i, "score": sim})
    
    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities[:top_k]




