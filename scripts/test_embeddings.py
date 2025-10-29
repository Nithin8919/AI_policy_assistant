#!/usr/bin/env python3
"""Test and validate embedding system"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_embedder():
    """Test the embedder functionality"""
    logger.info("Testing Embedder...")
    
    try:
        # Initialize embedder
        embedder = Embedder(
            model_name=settings.EMBEDDING_MODEL,
            batch_size=4,
            checkpoint_dir="data/embeddings/test"
        )
        
        # Test single embedding
        test_text = "This is a test document about education policy in Andhra Pradesh."
        result = embedder.embed_single(test_text, "test_chunk_1", "test_doc_1")
        
        if result.success:
            logger.info(f"✓ Single embedding: {len(result.embedding)} dimensions")
            logger.info(f"  Processing time: {result.processing_time:.3f} seconds")
        else:
            logger.error(f"✗ Single embedding failed: {result.error_message}")
            return False
        
        # Test batch embedding
        test_texts = [
            "Education policy framework for primary schools.",
            "Government order regarding teacher recruitment.",
            "Judicial ruling on education rights.",
            "Budget allocation for school infrastructure."
        ]
        
        results = embedder.embed_batch(
            texts=test_texts,
            chunk_ids=[f"test_chunk_{i}" for i in range(len(test_texts))],
            doc_ids=[f"test_doc_{i}" for i in range(len(test_texts))]
        )
        
        successful_results = [r for r in results if r.success]
        if len(successful_results) == len(test_texts):
            logger.info(f"✓ Batch embedding: {len(successful_results)} successful")
        else:
            logger.error(f"✗ Batch embedding: {len(successful_results)}/{len(test_texts)} successful")
            return False
        
        # Test dimension consistency
        embedding_dim = embedder.get_embedding_dimension()
        if embedding_dim == 384:  # Expected for all-MiniLM-L6-v2
            logger.info(f"✓ Embedding dimension: {embedding_dim}")
        else:
            logger.warning(f"⚠ Unexpected embedding dimension: {embedding_dim}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Embedder test failed: {e}")
        return False

def test_vector_store():
    """Test the vector store functionality"""
    logger.info("Testing Vector Store...")
    
    try:
        # Initialize vector store
        config = VectorStoreConfig(
            qdrant_url=settings.QDRANT_URL,
            qdrant_api_key=settings.QDRANT_API_KEY,
            vector_size=384,
            collection_prefix="test_policy_assistant"
        )
        
        vector_store = VectorStore(config)
        
        # Test health check
        health = vector_store.health_check()
        if health.get('qdrant_connected'):
            logger.info("✓ Qdrant connection successful")
        else:
            logger.error(f"✗ Qdrant connection failed: {health.get('error')}")
            return False
        
        # Test collection creation
        collections = vector_store.create_collections(recreate=True)
        if len(collections) == len(DocumentType):
            logger.info(f"✓ Created {len(collections)} collections")
        else:
            logger.error(f"✗ Collection creation failed: {len(collections)}/{len(DocumentType)}")
            return False
        
        # Test embedding insertion
        test_embeddings = [
            {
                'chunk_id': 'test_chunk_1',
                'doc_id': 'test_doc_1',
                'doc_type': 'acts',
                'content': 'Test education act content',
                'embedding': [0.1] * 384,
                'section_id': 'section_1',
                'year': 2023,
                'priority': 'high'
            },
            {
                'chunk_id': 'test_chunk_2',
                'doc_id': 'test_doc_2',
                'doc_type': 'government_orders',
                'content': 'Test government order content',
                'embedding': [0.2] * 384,
                'section_id': 'section_1',
                'year': 2024,
                'priority': 'medium'
            }
        ]
        
        insertion_counts = vector_store.upsert_embeddings(test_embeddings)
        total_inserted = sum(insertion_counts.values())
        
        if total_inserted == len(test_embeddings):
            logger.info(f"✓ Inserted {total_inserted} test embeddings")
        else:
            logger.error(f"✗ Insertion failed: {total_inserted}/{len(test_embeddings)}")
            return False
        
        # Test search
        query_vector = [0.15] * 384
        search_results = vector_store.search(
            query_vector=query_vector,
            limit=5
        )
        
        if search_results:
            logger.info(f"✓ Search returned {len(search_results)} results")
            for result in search_results[:2]:
                logger.info(f"  - {result.chunk_id}: score={result.score:.3f}")
        else:
            logger.warning("⚠ Search returned no results")
        
        # Test filtered search
        filtered_results = vector_store.search(
            query_vector=query_vector,
            limit=5,
            filters={'doc_type': 'acts'}
        )
        
        if filtered_results:
            logger.info(f"✓ Filtered search returned {len(filtered_results)} results")
        
        # Test document type search
        type_results = vector_store.search_by_document_type(
            query_vector=query_vector,
            doc_types=['acts', 'government_orders'],
            limit=5
        )
        
        if type_results:
            logger.info(f"✓ Document type search returned {len(type_results)} results")
        
        # Clean up test collections
        for doc_type in DocumentType:
            vector_store.delete_collection(doc_type)
        
        logger.info("✓ Cleaned up test collections")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Vector store test failed: {e}")
        return False

def test_integration():
    """Test end-to-end integration"""
    logger.info("Testing Integration...")
    
    try:
        # Create test chunks
        test_chunks = [
            {
                'chunk_id': 'integration_test_1',
                'doc_id': 'test_doc_integration',
                'doc_type': 'acts',
                'content': 'The Right to Education Act ensures free and compulsory education for children.',
                'section_id': 'section_1',
                'year': 2009,
                'priority': 'critical'
            },
            {
                'chunk_id': 'integration_test_2',
                'doc_id': 'test_doc_integration',
                'doc_type': 'government_orders',
                'content': 'Government order for implementation of Nadu-Nedu programme in schools.',
                'section_id': 'section_2',
                'year': 2020,
                'priority': 'high'
            }
        ]
        
        # Initialize embedder
        embedder = Embedder(
            model_name=settings.EMBEDDING_MODEL,
            batch_size=2
        )
        
        # Generate embeddings
        texts = [chunk['content'] for chunk in test_chunks]
        chunk_ids = [chunk['chunk_id'] for chunk in test_chunks]
        doc_ids = [chunk['doc_id'] for chunk in test_chunks]
        
        embedding_results = embedder.embed_batch(texts, chunk_ids, doc_ids)
        
        if all(r.success for r in embedding_results):
            logger.info(f"✓ Generated {len(embedding_results)} embeddings")
        else:
            logger.error("✗ Embedding generation failed")
            return False
        
        # Prepare for vector store
        embeddings_for_store = []
        for chunk, result in zip(test_chunks, embedding_results):
            if result.success:
                embedding_record = {
                    **chunk,
                    'embedding': result.embedding
                }
                embeddings_for_store.append(embedding_record)
        
        # Initialize vector store
        config = VectorStoreConfig(
            qdrant_url=settings.QDRANT_URL,
            qdrant_api_key=settings.QDRANT_API_KEY,
            vector_size=384,
            collection_prefix="integration_test"
        )
        
        vector_store = VectorStore(config)
        vector_store.create_collections(recreate=True)
        
        # Store embeddings
        insertion_counts = vector_store.upsert_embeddings(embeddings_for_store)
        total_inserted = sum(insertion_counts.values())
        
        if total_inserted == len(embeddings_for_store):
            logger.info(f"✓ Stored {total_inserted} embeddings")
        else:
            logger.error(f"✗ Storage failed: {total_inserted}/{len(embeddings_for_store)}")
            return False
        
        # Test semantic search
        query_text = "education rights for children"
        query_result = embedder.embed_single(query_text)
        
        if query_result.success:
            search_results = vector_store.search(
                query_vector=query_result.embedding,
                limit=2
            )
            
            if search_results:
                logger.info(f"✓ Semantic search for '{query_text}':")
                for result in search_results:
                    logger.info(f"  - {result.chunk_id}: score={result.score:.3f}")
                    logger.info(f"    Content: {result.content[:60]}...")
            else:
                logger.warning("⚠ Semantic search returned no results")
        
        # Clean up
        for doc_type in DocumentType:
            vector_store.delete_collection(doc_type)
        
        logger.info("✓ Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False

def performance_test():
    """Test performance with larger batches"""
    logger.info("Testing Performance...")
    
    try:
        # Generate larger test dataset
        test_texts = [
            f"This is test document number {i} about education policy and governance."
            for i in range(100)
        ]
        
        # Test embedder performance
        embedder = Embedder(
            model_name=settings.EMBEDDING_MODEL,
            batch_size=32
        )
        
        start_time = time.time()
        results = embedder.embed_batch(test_texts)
        embedding_time = time.time() - start_time
        
        successful_results = [r for r in results if r.success]
        
        if len(successful_results) == len(test_texts):
            logger.info(f"✓ Performance test: {len(successful_results)} embeddings")
            logger.info(f"  Total time: {embedding_time:.2f} seconds")
            logger.info(f"  Time per embedding: {embedding_time/len(test_texts):.3f} seconds")
            logger.info(f"  Throughput: {len(test_texts)/embedding_time:.1f} embeddings/second")
        else:
            logger.error(f"✗ Performance test failed: {len(successful_results)}/{len(test_texts)}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        return False

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Test embedding system")
    
    parser.add_argument(
        '--test',
        choices=['embedder', 'vector_store', 'integration', 'performance', 'all'],
        default='all',
        help='Which test to run'
    )
    
    parser.add_argument(
        '--qdrant-url',
        default=settings.QDRANT_URL,
        help='Qdrant server URL'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting embedding system tests...")
    
    test_results = {}
    
    if args.test in ['embedder', 'all']:
        test_results['embedder'] = test_embedder()
    
    if args.test in ['vector_store', 'all']:
        test_results['vector_store'] = test_vector_store()
    
    if args.test in ['integration', 'all']:
        test_results['integration'] = test_integration()
    
    if args.test in ['performance', 'all']:
        test_results['performance'] = performance_test()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name.upper()}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()