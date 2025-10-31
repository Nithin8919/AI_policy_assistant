#!/usr/bin/env python3
"""End-to-end test of the complete QA pipeline"""
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from src.agents.enhanced_router import EnhancedRouter
from src.query_processing.qa_pipeline_multi_llm import QAPipeline

def test_full_pipeline():
    print("="*80)
    print("END-TO-END QA PIPELINE TEST")
    print("="*80)
    
    # Initialize
    print("\n1. Initializing router...")
    router = EnhancedRouter(
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY')
    )
    print("✅ Router initialized")
    
    print("\n2. Initializing QA pipeline with Groq...")
    pipeline = QAPipeline(router=router, llm_provider='groq')
    print("✅ Pipeline initialized")
    
    # Test query
    query = "What is Section 12(1)(c) of RTE Act?"
    
    print(f"\n3. Processing query: '{query}'")
    print("-" * 80)
    
    try:
        response = pipeline.answer_query(query, mode="normal_qa", top_k=10)
        
        print(f"\n✅ Query processed successfully!")
        print(f"\nProcessing time: {response.processing_time:.2f}s")
        print(f"Confidence: {response.confidence_score:.0%}")
        
        print(f"\n📊 Retrieval Stats:")
        print(f"   Chunks retrieved: {response.retrieval_stats.get('chunks_retrieved', 0)}")
        print(f"   Agents used: {', '.join(response.retrieval_stats.get('agents_used', []))}")
        print(f"   Complexity: {response.retrieval_stats.get('query_complexity', 'unknown')}")
        
        print(f"\n💬 LLM Stats:")
        print(f"   Provider: {response.llm_stats.get('provider', 'unknown')}")
        print(f"   Model: {response.llm_stats.get('model', 'unknown')}")
        print(f"   Tokens: {response.llm_stats.get('total_tokens', 0)}")
        
        print(f"\n📚 Citations:")
        print(f"   Total citations: {response.citations.get('total_citations', 0)}")
        print(f"   Unique sources: {response.citations.get('unique_sources_cited', 0)}")
        print(f"   All valid: {response.citations.get('all_citations_valid', False)}")
        
        print(f"\n📝 Answer:")
        print("-" * 80)
        print(response.answer)
        print("-" * 80)
        
        # Validation
        print(f"\n✓ Validation:")
        checks = [
            ("Response time reasonable", 1 < response.processing_time < 30),
            ("Chunks retrieved", response.retrieval_stats.get('chunks_retrieved', 0) > 0),
            ("Has answer", len(response.answer) > 50),
            ("Has citations", response.citations.get('total_citations', 0) > 0),
            ("Confidence > 30%", response.confidence_score > 0.3)
        ]
        
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
        
        all_passed = all(passed for _, passed in checks)
        
        if all_passed:
            print(f"\n🎉 ALL CHECKS PASSED! System is fully operational!")
        else:
            print(f"\n⚠️  Some checks failed. Review results above.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_full_pipeline()

