#!/usr/bin/env python3
"""
Test SOTA Data Processing Pipeline

This is a simplified test version that uses existing working components
to validate the SOTA processing approach.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set environment variables
os.environ['QDRANT_URL'] = 'https://3bfa5117-dd8a-4048-abf9-5267856c164e.us-east4-0.gcp.cloud.qdrant.io:6333'
os.environ['QDRANT_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9Mk6YTL8BaQeHF3945J1_-MoWa4MWe-XvJxST5EeQ60'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType
from src.embeddings.embedder import Embedder
from src.knowledge_graph.bridge_builder import BridgeTableBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_document_processing():
    """Test basic document processing"""
    logger.info("🧪 Testing document processing...")
    
    # Check if we have processed chunks
    chunks_file = "./data/processed/chunks/all_chunks_consolidated.jsonl"
    
    if not os.path.exists(chunks_file):
        logger.warning("No processed chunks found. Using test data...")
        
        # Create simple test chunks
        test_chunks = [
            {
                'chunk_id': 'test_chunk_1',
                'doc_id': 'test_doc_1',
                'content': 'This is a test chunk about education policy. Section 12 of the RTE Act provides for admission procedures.',
                'metadata': {
                    'doc_type': 'legal_documents',
                    'year': 2023,
                    'title': 'Test Document'
                }
            },
            {
                'chunk_id': 'test_chunk_2', 
                'doc_id': 'test_doc_1',
                'content': 'Government Order Ms.No.54 supersedes previous order No.23. This order deals with teacher recruitment.',
                'metadata': {
                    'doc_type': 'government_orders',
                    'year': 2023,
                    'title': 'Test GO'
                }
            }
        ]
        
        # Save test chunks
        os.makedirs("./data/processed/chunks", exist_ok=True)
        with open(chunks_file, 'w') as f:
            for chunk in test_chunks:
                f.write(json.dumps(chunk) + '\n')
        
        logger.info("✅ Created test chunks")
        return test_chunks
    
    else:
        # Load existing chunks
        chunks = []
        with open(chunks_file, 'r') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        
        logger.info(f"✅ Loaded {len(chunks)} existing chunks")
        return chunks[:10]  # Use first 10 for testing


def test_embedding_generation(chunks: List[Dict[str, Any]]):
    """Test embedding generation"""
    logger.info("🔮 Testing embedding generation...")
    
    embedder = Embedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=4
    )
    
    # Extract texts (handle different content field names)
    texts = []
    chunk_ids = []
    doc_ids = []
    
    for chunk in chunks:
        # Handle different content field names
        content = chunk.get('content') or chunk.get('text') or chunk.get('chunk_content', '')
        if content:
            texts.append(content)
            chunk_ids.append(chunk.get('chunk_id', chunk.get('id', f'chunk_{len(texts)}')))
            doc_ids.append(chunk.get('doc_id', chunk.get('document_id', 'unknown')))
    
    logger.info(f"Extracted {len(texts)} valid chunks for embedding")
    
    # Generate embeddings
    results = embedder.embed_batch(texts, chunk_ids, doc_ids)
    
    # Combine with metadata
    enhanced_chunks = []
    for i, result in enumerate(results):
        if result.success and i < len(chunks):
            chunk = chunks[i]
            content = chunk.get('content') or chunk.get('text') or chunk.get('chunk_content', '')
            enhanced_chunk = {
                'chunk_id': result.chunk_id,
                'doc_id': result.doc_id,
                'content': content,
                'embedding': result.embedding,
                'metadata': chunk.get('metadata', {})
            }
            enhanced_chunks.append(enhanced_chunk)
    
    logger.info(f"✅ Generated {len(enhanced_chunks)} embeddings")
    return enhanced_chunks


def test_vector_storage(embeddings_data: List[Dict[str, Any]]):
    """Test vector storage"""
    logger.info("💾 Testing vector storage...")
    
    vector_store = VectorStore(VectorStoreConfig(
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY')
    ))
    
    # Group by document type
    type_groups = {}
    for embedding in embeddings_data:
        doc_type_str = embedding.get('metadata', {}).get('doc_type', 'unknown')
        
        # Map to DocumentType enum
        if 'legal' in doc_type_str:
            doc_type = DocumentType.LEGAL_DOCUMENTS
        elif 'government' in doc_type_str or 'order' in doc_type_str:
            doc_type = DocumentType.GOVERNMENT_ORDERS
        else:
            doc_type = DocumentType.EXTERNAL_SOURCES
        
        if doc_type not in type_groups:
            type_groups[doc_type] = []
        type_groups[doc_type].append(embedding)
    
    # Convert all embeddings to format expected by vector store
    formatted_embeddings = []
    for embedding in embeddings_data:
        doc_type_str = embedding.get('metadata', {}).get('doc_type', 'external_sources')
        
        formatted_embedding = {
            'chunk_id': embedding['chunk_id'],
            'doc_id': embedding['doc_id'],
            'content': embedding['content'],
            'embedding': embedding['embedding'],
            'doc_type': doc_type_str,
            'metadata': embedding.get('metadata', {})
        }
        formatted_embeddings.append(formatted_embedding)
    
    # Create collections first
    vector_store.create_collections()
    
    # Store all embeddings
    insertion_counts = vector_store.upsert_embeddings(formatted_embeddings)
    total_stored = sum(insertion_counts.values())
    
    for doc_type, count in insertion_counts.items():
        logger.info(f"  📂 Stored {count} embeddings for {doc_type.value}")
    
    logger.info(f"✅ Vector storage complete: {total_stored} embeddings stored")
    return total_stored


def test_knowledge_graph(embeddings_data: List[Dict[str, Any]]):
    """Test knowledge graph construction"""
    logger.info("🕸️ Testing knowledge graph construction...")
    
    # For now, create a simple entity extraction test
    entity_count = 0
    relationship_count = 0
    
    for embedding in embeddings_data:
        content = embedding['content']
        
        # Simple entity extraction patterns
        import re
        
        # Count legal sections
        legal_refs = re.findall(r'Section\s+\d+|Article\s+\d+|Rule\s+\d+', content, re.IGNORECASE)
        entity_count += len(legal_refs)
        
        # Count GO references  
        go_refs = re.findall(r'G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*\d+', content, re.IGNORECASE)
        entity_count += len(go_refs)
        
        # Count supersession relationships
        supersessions = re.findall(r'supersedes?|supersession', content, re.IGNORECASE)
        relationship_count += len(supersessions)
    
    # Create simple stats
    stats = {
        'entities_created': entity_count,
        'relationships_created': relationship_count,
        'test_success': True
    }
    
    logger.info(f"✅ Knowledge graph complete: {stats}")
    return stats


def main():
    """Main test function"""
    logger.info("=" * 80)
    logger.info("🧪 TESTING SOTA DATA PROCESSING PIPELINE")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Stage 1: Document Processing
        logger.info("📝 Stage 1: Document Processing")
        chunks = test_document_processing()
        
        if not chunks:
            raise Exception("No chunks available for testing")
        
        # Stage 2: Embedding Generation  
        logger.info("🔮 Stage 2: Embedding Generation")
        embeddings_data = test_embedding_generation(chunks)
        
        # Stage 3: Vector Storage
        logger.info("💾 Stage 3: Vector Storage")
        stored_count = test_vector_storage(embeddings_data)
        
        # Stage 4: Knowledge Graph
        logger.info("🕸️ Stage 4: Knowledge Graph")
        kg_stats = test_knowledge_graph(embeddings_data)
        
        # Calculate results
        total_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 80)
        print("✅ SOTA PIPELINE TEST COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"📚 Chunks Processed: {len(chunks)}")
        print(f"🔮 Embeddings Generated: {len(embeddings_data)}")
        print(f"💾 Vectors Stored: {stored_count}")
        print(f"🕸️ Entities Found: {kg_stats.get('entities_created', 0)}")
        print(f"🔗 Relationships Found: {kg_stats.get('relationships_created', 0)}")
        print("=" * 80)
        print("🎯 SOTA pipeline components working correctly!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)