#!/usr/bin/env python3
"""Test search functionality of the embedding system"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def search_test_queries():
    """Test with predefined queries relevant to AP education policy"""
    return [
        {
            "query": "Right to Education Act implementation",
            "expected_types": ["acts", "government_orders"],
            "description": "Should find RTE Act and related government orders"
        },
        {
            "query": "teacher recruitment and DSC notification",
            "expected_types": ["government_orders", "acts"],
            "description": "Should find teacher recruitment policies and notifications"
        },
        {
            "query": "Nadu Nedu school infrastructure program",
            "expected_types": ["government_orders"],
            "description": "Should find Nadu-Nedu program documents"
        },
        {
            "query": "Jagananna Amma Vodi scholarship scheme",
            "expected_types": ["government_orders"],
            "description": "Should find Amma Vodi scheme documents"
        },
        {
            "query": "budget allocation for education",
            "expected_types": ["data_reports", "government_orders"],
            "description": "Should find budget documents and financial orders"
        },
        {
            "query": "NCTE teacher education norms",
            "expected_types": ["external_sources", "acts"],
            "description": "Should find NCTE documents and education norms"
        },
        {
            "query": "school safety guidelines",
            "expected_types": ["judicial_documents", "government_orders"],
            "description": "Should find safety guidelines and judicial orders"
        },
        {
            "query": "private school regulation",
            "expected_types": ["acts", "judicial_documents"],
            "description": "Should find private school acts and court cases"
        }
    ]

def run_search_tests(
    vector_store: VectorStore,
    embedder: Embedder,
    test_queries: List[Dict[str, Any]],
    top_k: int = 5
) -> Dict[str, Any]:
    """Run search tests and return results"""
    
    results = {
        "total_queries": len(test_queries),
        "successful_searches": 0,
        "failed_searches": 0,
        "query_results": []
    }
    
    for i, test_query in enumerate(test_queries, 1):
        query_text = test_query["query"]
        expected_types = test_query["expected_types"]
        description = test_query["description"]
        
        logger.info(f"\n[{i}/{len(test_queries)}] Testing: {query_text}")
        logger.info(f"Expected types: {expected_types}")
        logger.info(f"Description: {description}")
        
        try:
            # Generate query embedding
            query_result = embedder.embed_single(query_text)
            
            if not query_result.success:
                logger.error(f"  ✗ Failed to generate query embedding: {query_result.error_message}")
                results["failed_searches"] += 1
                continue
            
            # Search all collections
            search_results = vector_store.search(
                query_vector=query_result.embedding,
                limit=top_k,
                score_threshold=0.3  # Minimum similarity threshold
            )
            
            if not search_results:
                logger.warning(f"  ⚠ No results found for query")
                results["failed_searches"] += 1
                query_result_data = {
                    "query": query_text,
                    "expected_types": expected_types,
                    "results_count": 0,
                    "top_results": [],
                    "type_distribution": {},
                    "avg_score": 0.0,
                    "success": False
                }
            else:
                logger.info(f"  ✓ Found {len(search_results)} results")
                
                # Analyze results
                type_distribution = {}
                total_score = 0
                
                for result in search_results:
                    doc_type = result.payload.get('doc_type', 'unknown')
                    type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
                    total_score += result.score
                
                avg_score = total_score / len(search_results)
                
                # Display top results
                logger.info(f"  Top {min(3, len(search_results))} results:")
                top_results_data = []
                
                for j, result in enumerate(search_results[:3]):
                    content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                    logger.info(f"    {j+1}. {result.chunk_id} (score: {result.score:.3f})")
                    logger.info(f"       Type: {result.payload.get('doc_type', 'unknown')}")
                    logger.info(f"       Doc: {result.doc_id}")
                    logger.info(f"       Content: {content_preview}")
                    
                    top_results_data.append({
                        "chunk_id": result.chunk_id,
                        "doc_id": result.doc_id,
                        "score": result.score,
                        "doc_type": result.payload.get('doc_type', 'unknown'),
                        "content_preview": content_preview
                    })
                
                # Check if expected types are found
                found_expected_types = any(doc_type in type_distribution for doc_type in expected_types)
                
                logger.info(f"  Type distribution: {type_distribution}")
                logger.info(f"  Average score: {avg_score:.3f}")
                logger.info(f"  Expected types found: {found_expected_types}")
                
                results["successful_searches"] += 1
                
                query_result_data = {
                    "query": query_text,
                    "expected_types": expected_types,
                    "results_count": len(search_results),
                    "top_results": top_results_data,
                    "type_distribution": type_distribution,
                    "avg_score": avg_score,
                    "expected_types_found": found_expected_types,
                    "success": True
                }
            
            results["query_results"].append(query_result_data)
            
        except Exception as e:
            logger.error(f"  ✗ Search failed: {e}")
            results["failed_searches"] += 1
            
            results["query_results"].append({
                "query": query_text,
                "expected_types": expected_types,
                "error": str(e),
                "success": False
            })
    
    return results

def test_filtered_search(vector_store: VectorStore, embedder: Embedder):
    """Test search with metadata filters"""
    logger.info("\nTesting filtered search...")
    
    test_cases = [
        {
            "query": "education policy",
            "filters": {"doc_type": "acts"},
            "description": "Search only in legal documents"
        },
        {
            "query": "government order",
            "filters": {"doc_type": "government_orders"},
            "description": "Search only in government orders"
        },
        {
            "query": "budget allocation",
            "filters": {"year": {"range": {"gte": 2020}}},
            "description": "Search documents from 2020 onwards"
        },
        {
            "query": "critical policy",
            "filters": {"priority": "critical"},
            "description": "Search only critical priority documents"
        }
    ]
    
    for test_case in test_cases:
        query_text = test_case["query"]
        filters = test_case["filters"]
        description = test_case["description"]
        
        logger.info(f"\nFiltered search: {description}")
        logger.info(f"Query: {query_text}")
        logger.info(f"Filters: {filters}")
        
        try:
            # Generate query embedding
            query_result = embedder.embed_single(query_text)
            
            if query_result.success:
                # Search with filters
                search_results = vector_store.search(
                    query_vector=query_result.embedding,
                    limit=5,
                    filters=filters
                )
                
                if search_results:
                    logger.info(f"  ✓ Found {len(search_results)} filtered results")
                    for result in search_results[:2]:
                        logger.info(f"    - {result.chunk_id}: {result.score:.3f}")
                        logger.info(f"      Type: {result.payload.get('doc_type')}")
                        logger.info(f"      Year: {result.payload.get('year')}")
                        logger.info(f"      Priority: {result.payload.get('priority')}")
                else:
                    logger.warning(f"  ⚠ No filtered results found")
            else:
                logger.error(f"  ✗ Failed to generate query embedding")
                
        except Exception as e:
            logger.error(f"  ✗ Filtered search failed: {e}")

def test_document_type_search(vector_store: VectorStore, embedder: Embedder):
    """Test search by specific document types"""
    logger.info("\nTesting document type search...")
    
    test_cases = [
        {
            "query": "education rights",
            "doc_types": ["acts", "judicial_documents"],
            "description": "Search legal documents and court cases"
        },
        {
            "query": "teacher recruitment",
            "doc_types": ["government_orders"],
            "description": "Search government orders only"
        },
        {
            "query": "school statistics",
            "doc_types": ["data_reports"],
            "description": "Search data reports only"
        }
    ]
    
    for test_case in test_cases:
        query_text = test_case["query"]
        doc_types = test_case["doc_types"]
        description = test_case["description"]
        
        logger.info(f"\nDocument type search: {description}")
        logger.info(f"Query: {query_text}")
        logger.info(f"Document types: {doc_types}")
        
        try:
            # Generate query embedding
            query_result = embedder.embed_single(query_text)
            
            if query_result.success:
                # Search specific document types
                search_results = vector_store.search_by_document_type(
                    query_vector=query_result.embedding,
                    doc_types=doc_types,
                    limit=5
                )
                
                if search_results:
                    logger.info(f"  ✓ Found {len(search_results)} results")
                    for result in search_results[:2]:
                        logger.info(f"    - {result.chunk_id}: {result.score:.3f}")
                        logger.info(f"      Type: {result.payload.get('doc_type')}")
                else:
                    logger.warning(f"  ⚠ No results found")
            else:
                logger.error(f"  ✗ Failed to generate query embedding")
                
        except Exception as e:
            logger.error(f"  ✗ Document type search failed: {e}")

def save_search_results(results: Dict[str, Any], output_file: str):
    """Save search test results to file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Search results saved to {output_path}")

def main():
    """Main search test runner"""
    parser = argparse.ArgumentParser(description="Test search functionality")
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to retrieve'
    )
    
    parser.add_argument(
        '--output',
        default='data/evaluation/search_test_results.json',
        help='Output file for results'
    )
    
    parser.add_argument(
        '--test-type',
        choices=['basic', 'filtered', 'document_type', 'all'],
        default='all',
        help='Type of search test to run'
    )
    
    parser.add_argument(
        '--qdrant-url',
        default=settings.QDRANT_URL,
        help='Qdrant server URL'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting search functionality tests...")
    
    try:
        # Initialize embedder
        logger.info("Initializing embedder...")
        embedder = Embedder(model_name=settings.EMBEDDING_MODEL)
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        config = VectorStoreConfig(
            qdrant_url=args.qdrant_url,
            qdrant_api_key=settings.QDRANT_API_KEY,
            vector_size=384
        )
        vector_store = VectorStore(config)
        
        # Check vector store health
        health = vector_store.health_check()
        if not health.get('qdrant_connected'):
            logger.error(f"Vector store connection failed: {health.get('error')}")
            sys.exit(1)
        
        logger.info("Vector store connection successful")
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        total_embeddings = 0
        
        logger.info("Collection status:")
        for collection_name, info in collection_info.items():
            if 'error' not in info:
                count = info.get('points_count', 0)
                total_embeddings += count
                logger.info(f"  {collection_name}: {count} embeddings")
            else:
                logger.warning(f"  {collection_name}: {info['error']}")
        
        if total_embeddings == 0:
            logger.error("No embeddings found in vector store. Please run generate_embeddings.py first.")
            sys.exit(1)
        
        logger.info(f"Total embeddings available: {total_embeddings}")
        
        # Run tests
        if args.test_type in ['basic', 'all']:
            # Run basic search tests
            test_queries = search_test_queries()
            results = run_search_tests(vector_store, embedder, test_queries, args.top_k)
            
            # Save results
            if args.output:
                save_search_results(results, args.output)
            
            # Print summary
            logger.info(f"\n{'='*50}")
            logger.info("SEARCH TEST SUMMARY")
            logger.info(f"{'='*50}")
            logger.info(f"Total queries: {results['total_queries']}")
            logger.info(f"Successful searches: {results['successful_searches']}")
            logger.info(f"Failed searches: {results['failed_searches']}")
            logger.info(f"Success rate: {results['successful_searches']/results['total_queries']*100:.1f}%")
        
        if args.test_type in ['filtered', 'all']:
            test_filtered_search(vector_store, embedder)
        
        if args.test_type in ['document_type', 'all']:
            test_document_type_search(vector_store, embedder)
        
        logger.info("\n✓ Search tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Search test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()