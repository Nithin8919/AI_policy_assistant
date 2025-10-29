"""Run full data processing pipeline"""
import sys
sys.path.append('src')

from src.ingestion.pipeline import process_document
from pathlib import Path

def main():
    """Main ingestion pipeline"""
    data_dir = Path("data/raw")
    output_dir = Path("data/processed")
    
    # Process all documents
    for doc in data_dir.rglob("*.pdf"):
        print(f"Processing {doc}")
        process_document(str(doc), str(output_dir))

if __name__ == "__main__":
    main()


