#!/usr/bin/env python3
"""Test script to verify retrieval system works after fixes"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ['QDRANT_URL'] = 'https://3bfa5117-dd8a-4048-abf9-5267856c164e.us-east4-0.gcp.cloud.qdrant.io:6333'
os.environ['QDRANT_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9Mk6YTL8BaQeHF3945J1_-MoWa4MWe-XvJxST5EeQ60'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.agents.enhanced_router import EnhancedRouter
import time

def test_retrieval():
    """Test the enhanced router retrieval"""
    print("=== TESTING ENHANCED ROUTER RETRIEVAL ===")
    
    try:
        # Initialize router
        router = EnhancedRouter(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY')
        )
        print("✅ Router initialized successfully")
        
        # Test queries
        test_queries = [
            "What is Section 12(1)(c) of RTE Act?",
            "Nadu-Nedu scheme implementation guidelines",
            "Teacher transfer rules in AP",
            "Dropout rate statistics in government schools",
            "Primary education funding allocation"
        ]
        
        print(f"\n=== TESTING {len(test_queries)} QUERIES ===")
        
        total_start = time.time()
        all_passed = True
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            start_time = time.time()
            
            try:
                result = router.route_query(query, top_k=5)
                
                processing_time = time.time() - start_time
                total_chunks = sum(len(r.chunks) for r in result.retrieval_results)
                
                print(f"✅ Processing time: {processing_time:.3f}s")
                print(f"✅ Agents selected: {len(result.selected_agents)}")
                print(f"✅ Successful retrievals: {len(result.retrieval_results)}")
                print(f"✅ Total chunks: {total_chunks}")
                
                if total_chunks == 0:
                    print("❌ FAILED: No chunks retrieved!")
                    all_passed = False
                else:
                    # Show sample results
                    for agent_result in result.retrieval_results[:2]:  # Show first 2 agents
                        print(f"  📁 {agent_result.agent_name}: {len(agent_result.chunks)} chunks")
                        if agent_result.chunks:
                            chunk = agent_result.chunks[0]
                            title = chunk['metadata'].get('title', 'Unknown')[:50]
                            print(f"    - {title}... (score: {chunk['score']:.3f})")
                
                # Validate response time is reasonable (not suspicious 0.37s)
                if processing_time < 0.5:
                    print(f"⚠️  WARNING: Very fast response time ({processing_time:.3f}s) - might not be doing real search")
                
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                all_passed = False
        
        total_time = time.time() - total_start
        print(f"\n=== SUMMARY ===")
        print(f"Total test time: {total_time:.3f}s")
        print(f"All tests passed: {'✅ YES' if all_passed else '❌ NO'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_retrieval()
    sys.exit(0 if success else 1)