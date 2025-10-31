#!/usr/bin/env python3
"""Generate embeddings from processed chunks and store in Qdrant vector database"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import Embedder, EmbeddingStats
from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_chunks_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all chunks from JSONL file"""
    chunks = []
    
    logger.info(f"Loading chunks from {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks

def group_chunks_by_document_type(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group chunks by document type"""
    grouped = {}
    
    for chunk in chunks:
        # Try top-level doc_type first, then check metadata
        doc_type = chunk.get('doc_type')
        if not doc_type:
            doc_type = chunk.get('metadata', {}).get('doc_type', 'external_sources')
        
        if doc_type not in grouped:
            grouped[doc_type] = []
        grouped[doc_type].append(chunk)
    
    # Log statistics
    logger.info("Chunks grouped by document type:")
    for doc_type, doc_chunks in grouped.items():
        logger.info(f"  {doc_type}: {len(doc_chunks)} chunks")
    
    return grouped

def generate_and_store_embeddings(
    jsonl_path: str,
    vector_store_config: VectorStoreConfig,
    embedder_config: Dict[str, Any],
    batch_size: int = 32,
    checkpoint_interval: int = 100,
    resume: bool = True,
    recreate_collections: bool = False
) -> Dict[str, Any]:
    """
    Generate embeddings and store in vector database
    
    Returns:
        Dictionary with generation statistics
    """
    start_time = time.time()
    
    # Initialize embedder
    logger.info("Initializing embedder...")
    embedder = Embedder(
        model_name=embedder_config.get('model_name', settings.EMBEDDING_MODEL),
        batch_size=batch_size,
        checkpoint_dir=embedder_config.get('checkpoint_dir', 'data/embeddings'),
        checkpoint_interval=checkpoint_interval
    )
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = VectorStore(vector_store_config)
    
    # Create collections
    logger.info("Creating collections...")
    created_collections = vector_store.create_collections(recreate=recreate_collections)
    
    # Load chunks
    chunks = load_chunks_from_jsonl(jsonl_path)
    if not chunks:
        logger.error("No chunks found to process")
        return {'error': 'No chunks found'}
    
    # Group by document type
    grouped_chunks = group_chunks_by_document_type(chunks)
    
    # Statistics tracking
    total_stats = {
        'total_chunks': len(chunks),
        'successful_embeddings': 0,
        'failed_embeddings': 0,
        'total_time': 0,
        'collections_created': len(created_collections),
        'embeddings_by_type': {},
        'embedding_dimension': embedder.get_embedding_dimension()
    }
    
    # Process each document type
    for doc_type, doc_chunks in grouped_chunks.items():
        logger.info(f"Processing {len(doc_chunks)} chunks for document type: {doc_type}")
        
        type_start_time = time.time()
        
        # Generate embeddings for this document type
        embedding_results, embedding_stats = embedder.generate_embeddings_from_jsonl_chunks(doc_chunks)
        
        # Prepare embeddings for vector store
        embeddings_for_store = []
        for chunk, result in zip(doc_chunks, embedding_results):
            if result.success:
                # Ensure doc_type is at top level for vector store
                embedding_record = {
                    'chunk_id': result.chunk_id,
                    'doc_id': result.doc_id,
                    'embedding': result.embedding,
                    'doc_type': doc_type,  # Explicit doc_type for vector store routing
                    'content': chunk.get('text', ''),  # Make sure content is available
                    **chunk  # Include all original chunk metadata
                }
                # Also ensure metadata has doc_type
                if 'metadata' in embedding_record:
                    embedding_record['metadata']['doc_type'] = doc_type
                embeddings_for_store.append(embedding_record)
        
        # Store in vector database
        if embeddings_for_store:
            logger.info(f"Storing {len(embeddings_for_store)} embeddings for {doc_type}")
            insertion_counts = vector_store.upsert_embeddings(embeddings_for_store, batch_size=100)
            
            # Update statistics
            type_time = time.time() - type_start_time
            total_stats['embeddings_by_type'][doc_type] = {
                'chunks_processed': len(doc_chunks),
                'successful_embeddings': embedding_stats.successful_embeddings,
                'failed_embeddings': embedding_stats.failed_embeddings,
                'processing_time': type_time,
                'insertion_counts': {k.value: v for k, v in insertion_counts.items()}
            }
            
            total_stats['successful_embeddings'] += embedding_stats.successful_embeddings
            total_stats['failed_embeddings'] += embedding_stats.failed_embeddings
        else:
            logger.warning(f"No successful embeddings for document type: {doc_type}")
    
    # Final statistics
    total_stats['total_time'] = time.time() - start_time
    total_stats['avg_time_per_chunk'] = total_stats['total_time'] / total_stats['total_chunks'] if total_stats['total_chunks'] > 0 else 0
    
    # Save statistics
    stats_file = Path('data/embeddings/generation_stats.json')
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_file, 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    logger.info(f"Generation complete. Statistics saved to {stats_file}")
    
    return total_stats

def generate_embeddings_from_jsonl_chunks(self, chunks: List[Dict[str, Any]]) -> tuple:
    """Generate embeddings from list of chunk dictionaries"""
    # Create temporary JSONL file
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
        temp_file = f.name
    
    try:
        # Use existing method
        return self.generate_embeddings_from_jsonl(temp_file, resume=False)
    finally:
        # Clean up temp file
        Path(temp_file).unlink(missing_ok=True)

# Monkey patch the method
Embedder.generate_embeddings_from_jsonl_chunks = generate_embeddings_from_jsonl_chunks

def test_vector_store_connection(config: VectorStoreConfig) -> bool:
    """Test connection to vector store"""
    try:
        vector_store = VectorStore(config)
        health = vector_store.health_check()
        
        if health.get('qdrant_connected'):
            logger.info("✓ Vector store connection successful")
            logger.info(f"  Collections: {health.get('total_collections', 0)}")
            return True
        else:
            logger.error(f"✗ Vector store connection failed: {health.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Vector store connection failed: {e}")
        return False

def validate_embeddings(config: VectorStoreConfig, sample_size: int = 5) -> bool:
    """Validate that embeddings were created successfully"""
    try:
        vector_store = VectorStore(config)
        
        # Test search with a dummy vector
        dummy_vector = [0.1] * config.vector_size
        
        for doc_type in DocumentType:
            collection_name = vector_store.get_collection_name(doc_type)
            
            try:
                results = vector_store.search(
                    query_vector=dummy_vector,
                    collection_names=[collection_name],
                    limit=sample_size
                )
                
                if results:
                    logger.info(f"✓ {collection_name}: {len(results)} embeddings found")
                else:
                    logger.warning(f"⚠ {collection_name}: No embeddings found")
                    
            except Exception as e:
                logger.error(f"✗ {collection_name}: Search failed - {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate embeddings from processed chunks")
    
    parser.add_argument(
        '--chunks-file',
        default='data/processed/chunks/all_chunks.jsonl',
        help='Path to JSONL file containing chunks'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation'
    )
    
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=100,
        help='Save checkpoint every N batches'
    )
    
    parser.add_argument(
        '--recreate-collections',
        action='store_true',
        help='Recreate vector store collections'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoint'
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test vector store connection only'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing embeddings'
    )
    
    parser.add_argument(
        '--qdrant-url',
        default=settings.QDRANT_URL,
        help='Qdrant server URL'
    )
    
    parser.add_argument(
        '--qdrant-api-key',
        default=settings.QDRANT_API_KEY,
        help='Qdrant API key'
    )
    
    args = parser.parse_args()
    
    # Setup vector store configuration
    vector_store_config = VectorStoreConfig(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        vector_size=384,  # all-MiniLM-L6-v2 dimension
        distance_metric="Cosine"
    )
    
    # Test connection only
    if args.test_connection:
        success = test_vector_store_connection(vector_store_config)
        sys.exit(0 if success else 1)
    
    # Validate embeddings only
    if args.validate:
        success = validate_embeddings(vector_store_config)
        sys.exit(0 if success else 1)
    
    # Check if chunks file exists
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        logger.info("Please run the ingestion pipeline first to generate chunks")
        sys.exit(1)
    
    # Embedder configuration
    embedder_config = {
        'model_name': settings.EMBEDDING_MODEL,
        'checkpoint_dir': 'data/embeddings'
    }
    
    logger.info("Starting embedding generation pipeline...")
    logger.info(f"Chunks file: {chunks_file}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Checkpoint interval: {args.checkpoint_interval}")
    logger.info(f"Resume: {not args.no_resume}")
    logger.info(f"Recreate collections: {args.recreate_collections}")
    
    # Generate and store embeddings
    try:
        stats = generate_and_store_embeddings(
            jsonl_path=str(chunks_file),
            vector_store_config=vector_store_config,
            embedder_config=embedder_config,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            resume=not args.no_resume,
            recreate_collections=args.recreate_collections
        )
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EMBEDDING GENERATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Successful embeddings: {stats['successful_embeddings']}")
        logger.info(f"Failed embeddings: {stats['failed_embeddings']}")
        logger.info(f"Success rate: {stats['successful_embeddings']/stats['total_chunks']*100:.1f}%")
        logger.info(f"Total time: {stats['total_time']:.2f} seconds")
        logger.info(f"Average time per chunk: {stats['avg_time_per_chunk']:.3f} seconds")
        logger.info(f"Collections created: {stats['collections_created']}")
        logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
        
        logger.info("\nEmbeddings by document type:")
        for doc_type, type_stats in stats['embeddings_by_type'].items():
            logger.info(f"  {doc_type}: {type_stats['successful_embeddings']} embeddings")
        
        # Validate results
        logger.info("\nValidating embeddings...")
        validate_embeddings(vector_store_config)
        
        logger.info("\n✓ Embedding generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()