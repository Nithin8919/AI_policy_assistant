"""Update embeddings incrementally"""
import sys
sys.path.append('src')

from src.embeddings.pipeline import update_embeddings

def main():
    """Update embeddings for new documents"""
    update_embeddings()

if __name__ == "__main__":
    main()






