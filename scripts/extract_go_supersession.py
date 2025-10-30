#!/usr/bin/env python3
"""
Extract GO supersession relationships from processed GOs
"""
import json
import re
from pathlib import Path
import csv
import sys

def extract_go_supersession(processed_dir: str):
    """Extract supersession chains from processed GOs"""
    
    go_dir = Path(processed_dir) / "Government_Orders"
    supersessions = []
    
    if not go_dir.exists():
        print(f"❌ Government_Orders directory not found: {go_dir}")
        return []
    
    # Look for metadata files in the metadata subdirectory
    metadata_dir = go_dir / "metadata"
    if not metadata_dir.exists():
        print(f"❌ Metadata directory not found: {metadata_dir}")
        return []
    
    metadata_files = list(metadata_dir.glob("*_metadata.json"))
    
    print(f"🔍 Searching for supersession relationships in {len(metadata_files)} GO metadata files...")
    
    for json_file in metadata_files:
        print(f"📄 Processing: {json_file.name}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            doc_id = data.get("doc_id", "")
            title = data.get("title", "")
            file_name = data.get("file_name", "")
            
            # Extract GO number from title or filename
            primary_go = None
            go_patterns = [
                r'G\.?O\.?\s*(?:Ms|MS|Rt)\.?\s*(?:No\.?)?\s*(\d+)',
                r'MS\s*(\d+)',
                r'GO\s*(\d+)'
            ]
            
            for pattern in go_patterns:
                match = re.search(pattern, title + " " + file_name, re.IGNORECASE)
                if match:
                    primary_go = f"G.O.MS.No.{match.group(1)}"
                    break
            
            if not primary_go:
                print(f"  ⚠️ No GO number found in {json_file.name}")
                continue
            
            print(f"  ✅ Found GO: {primary_go}")
            
            # Read chunks from the all_chunks.jsonl file
            chunks_file = go_dir / "chunks" / "all_chunks.jsonl"
            if not chunks_file.exists():
                print(f"  ⚠️ Chunks file not found: {chunks_file}")
                continue
            
            # Read and filter chunks for this document
            full_text = ""
            with open(chunks_file, 'r') as f:
                for line in f:
                    chunk_data = json.loads(line.strip())
                    if chunk_data.get("doc_id") == doc_id:
                        full_text += chunk_data.get("text", "") + " "
            
            # Search for supersession patterns in the full text
            supersession_patterns = [
                r'supersed(?:e|es|ing)\s+G\.?O\.?\s*(?:Ms|MS|Rt)\.?\s*(?:No\.?)?\s*(\d+)',
                r'(?:replaces?|substitute[sd]?)\s+G\.?O\.?\s*(?:Ms|MS|Rt)\.?\s*(?:No\.?)?\s*(\d+)',
                r'(?:in place of|instead of)\s+G\.?O\.?\s*(?:Ms|MS|Rt)\.?\s*(?:No\.?)?\s*(\d+)',
                r'(?:cancels?|revokes?)\s+G\.?O\.?\s*(?:Ms|MS|Rt)\.?\s*(?:No\.?)?\s*(\d+)'
            ]
            
            for pattern in supersession_patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    superseded_go = f"G.O.MS.No.{match}"
                    supersessions.append({
                        "go_number": primary_go,
                        "supersedes": superseded_go,
                        "document": doc_id,
                        "source_file": file_name,
                        "status": "active"
                    })
                    print(f"  🔗 Found supersession: {primary_go} supersedes {superseded_go}")
        
        except Exception as e:
            print(f"  ❌ Error processing {json_file.name}: {str(e)}")
            continue
    
    # Remove duplicates
    unique_supersessions = []
    seen = set()
    for item in supersessions:
        key = (item["go_number"], item["supersedes"])
        if key not in seen:
            unique_supersessions.append(item)
            seen.add(key)
    
    # Save to CSV
    output_file = Path(processed_dir) / "go_supersession_chains.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["go_number", "supersedes", "document", "source_file", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_supersessions)
    
    print(f"\n✅ Extracted {len(unique_supersessions)} unique supersession relationships")
    print(f"📄 Saved to: {output_file}")
    
    # Print summary
    if unique_supersessions:
        print(f"\n📊 SUPERSESSION SUMMARY:")
        for item in unique_supersessions:
            print(f"  • {item['go_number']} supersedes {item['supersedes']}")
    
    return unique_supersessions

def main():
    if len(sys.argv) > 1:
        processed_dir = sys.argv[1]
    else:
        processed_dir = "data/processed_verticals"
    
    print("🔗 GO SUPERSESSION CHAIN EXTRACTOR")
    print("=" * 50)
    print(f"📁 Processing directory: {processed_dir}")
    
    extract_go_supersession(processed_dir)

if __name__ == "__main__":
    main()