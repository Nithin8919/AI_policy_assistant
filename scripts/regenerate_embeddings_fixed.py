#!/usr/bin/env python3
"""
Regenerate embeddings with proper doc_type routing
This script consolidates chunks from all verticals and generates embeddings properly distributed across collections
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

def consolidate_chunks_from_verticals():
    """Consolidate all chunks from Legal, Government_Orders, and Schemes"""
    processed_verticals_dir = Path("data/processed_verticals")
    verticals = ["Legal", "Government_Orders", "Schemes"]
    
    all_chunks = []
    chunk_counts = {}
    
    for vertical in verticals:
        chunks_file = processed_verticals_dir / vertical / "chunks" / "all_chunks.jsonl"
        
        if not chunks_file.exists():
            logger.warning(f"Chunks file not found for {vertical}: {chunks_file}")
            continue
        
        logger.info(f"Loading chunks from {vertical}...")
        vertical_chunks = []
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line.strip())
                    vertical_chunks.append(chunk)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line {line_num} in {vertical}: {e}")
                    continue
        
        chunk_counts[vertical] = len(vertical_chunks)
        all_chunks.extend(vertical_chunks)
        logger.info(f"Loaded {len(vertical_chunks)} chunks from {vertical}")
    
    logger.info(f"\nTotal chunks consolidated: {len(all_chunks)}")
    logger.info(f"Breakdown: {chunk_counts}")
    
    return all_chunks, chunk_counts

def save_consolidated_chunks(chunks, output_file):
    """Save consolidated chunks to JSONL file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving consolidated chunks to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    logger.info(f"Saved {len(chunks)} chunks to {output_file}")

def analyze_doc_types(chunks):
    """Analyze doc_type distribution in chunks"""
    doc_type_counts = {}
    
    for chunk in chunks:
        # Get doc_type from metadata
        doc_type = chunk.get('metadata', {}).get('doc_type', 'unknown')
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
    
    logger.info("\nDoc Type Distribution:")
    for doc_type, count in sorted(doc_type_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {doc_type}: {count} chunks")
    
    return doc_type_counts

def main():
    logger.info("="*80)
    logger.info("REGENERATING EMBEDDINGS WITH PROPER DOC_TYPE ROUTING")
    logger.info("="*80)
    
    # Step 1: Consolidate all chunks
    logger.info("\nStep 1: Consolidating chunks from all verticals...")
    all_chunks, vertical_counts = consolidate_chunks_from_verticals()
    
    if not all_chunks:
        logger.error("No chunks found! Please run the ingestion pipeline first.")
        sys.exit(1)
    
    # Step 2: Analyze doc_type distribution
    logger.info("\nStep 2: Analyzing doc_type distribution...")
    doc_type_counts = analyze_doc_types(all_chunks)
    
    # Step 3: Save consolidated chunks
    output_file = "data/processed/chunks/all_chunks_consolidated.jsonl"
    logger.info(f"\nStep 3: Saving consolidated chunks...")
    save_consolidated_chunks(all_chunks, output_file)
    
    logger.info("\n" + "="*80)
    logger.info("CONSOLIDATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info(f"Output file: {output_file}")
    logger.info("\nNext steps:")
    logger.info(f"  1. Run: python scripts/generate_embeddings.py --chunks-file {output_file} --recreate-collections")
    logger.info(f"  2. This will properly distribute embeddings across collections")
    logger.info(f"  3. Expected distribution:")
    for doc_type, count in sorted(doc_type_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"      - {doc_type}: {count} chunks")
    logger.info("\n" + "="*80)

if __name__ == "__main__":
    main()

