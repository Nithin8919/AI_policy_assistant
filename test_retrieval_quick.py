#!/usr/bin/env python3
"""Quick retrieval test to verify embeddings work"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.agents.enhanced_router import EnhancedRouter

def test_retrieval():
    print("="*80)
    print("TESTING RETRIEVAL WITH NEW EMBEDDINGS")
    print("="*80)
    
    # Initialize router
    print("\n1. Initializing router...")
    router = EnhancedRouter(
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY')
    )
    print("✅ Router initialized")
    
    # Test queries
    test_queries = [
        "What is Section 12(1)(c) of RTE Act?",
        "What are the details of Nadu-Nedu scheme?",
        "What are teacher qualifications in Andhra Pradesh?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {query}")
        print(f"{'='*80}")
        
        try:
            result = router.route_query(query, top_k=5)
            
            total_chunks = sum(len(r.chunks) for r in result.retrieval_results)
            
            print(f"\n✅ Query processed successfully!")
            print(f"   Complexity: {result.complexity.value}")
            print(f"   Agents used: {len(result.selected_agents)}")
            print(f"   Total chunks retrieved: {total_chunks}")
            
            for agent_result in result.retrieval_results:
                print(f"\n   Agent: {agent_result.agent_name}")
                print(f"   Collection: {agent_result.doc_type}")
                print(f"   Chunks: {agent_result.total_results}")
                
                if agent_result.chunks:
                    top_chunk = agent_result.chunks[0]
                    print(f"   Top match score: {top_chunk.get('score', 0):.3f}")
                    doc_title = top_chunk.get('metadata', {}).get('title', 'Unknown')
                    print(f"   Top doc: {doc_title[:60]}...")
                    print(f"   Preview: {top_chunk.get('text', '')[:100]}...")
            
            if total_chunks == 0:
                print("\n❌ WARNING: No chunks retrieved!")
            else:
                print(f"\n✅ SUCCESS: Retrieved {total_chunks} relevant chunks")
                
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("RETRIEVAL TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_retrieval()

