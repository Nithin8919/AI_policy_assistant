"""Generate corpus metadata CSV"""
import json
import csv
from pathlib import Path

def main():
    """Export corpus index"""
    data_dir = Path("data/raw")
    output_file = Path("outputs/corpus_index.csv")
    
    all_docs = []
    for metadata_file in data_dir.rglob("metadata.json"):
        with open(metadata_file, 'r') as f:
            docs = json.load(f)
            if isinstance(docs, dict) and "documents" in docs:
                all_docs.extend(docs["documents"])
    
    # Write CSV
    if all_docs:
        fieldnames = all_docs[0].keys()
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_docs)

if __name__ == "__main__":
    main()






