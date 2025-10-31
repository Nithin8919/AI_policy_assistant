#!/usr/bin/env python3
"""Test Gemini provider with fixed retrieval system"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ['QDRANT_URL'] = 'https://3bfa5117-dd8a-4048-abf9-5267856c164e.us-east4-0.gcp.cloud.qdrant.io:6333'
os.environ['QDRANT_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9Mk6YTL8BaQeHF3945J1_-MoWa4MWe-XvJxST5EeQ60'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDXkZEtW7gG8E_yuMGeM7SGAcQpWKVIGsc'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.agents.enhanced_router import EnhancedRouter
from src.query_processing.qa_pipeline_multi_llm import QAPipeline
import time

def test_gemini_pipeline():
    """Test the complete pipeline with Gemini"""
    print("=== TESTING COMPLETE QA PIPELINE WITH GEMINI ===")
    
    try:
        # Initialize router
        router = EnhancedRouter(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY')
        )
        print("✅ Router initialized")
        
        # Initialize QA Pipeline with Gemini
        pipeline = QAPipeline(
            router=router,
            llm_provider="gemini",
            api_key=os.getenv('GOOGLE_API_KEY')
        )
        print("✅ QA Pipeline with Gemini initialized")
        
        # Test queries
        test_queries = [
            "What is Section 12(1)(c) of RTE Act?",
            "Nadu-Nedu scheme implementation guidelines",
            "Teacher transfer rules in AP"
        ]
        
        print(f"\n=== TESTING {len(test_queries)} QUERIES WITH GEMINI ===")
        
        success_count = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            start_time = time.time()
            
            try:
                # Run complete pipeline
                response = pipeline.answer_query(query, mode="normal_qa", top_k=5)
                
                processing_time = time.time() - start_time
                
                print(f"✅ Pipeline completed in {processing_time:.3f}s")
                print(f"✅ Answer length: {len(response.answer)} characters")
                print(f"✅ Citations: {response.citations.get('total_citations', 0)}")
                print(f"✅ Chunks retrieved: {response.retrieval_stats.get('chunks_retrieved', 0)}")
                print(f"✅ Confidence: {response.confidence_score:.2f}")
                
                # Show first 150 chars of answer
                answer_preview = response.answer[:150] + "..." if len(response.answer) > 150 else response.answer
                print(f"📄 Answer preview: {answer_preview}")
                
                if response.answer and not response.answer.startswith("Error"):
                    success_count += 1
                    print("✅ SUCCESS")
                else:
                    print(f"❌ FAILED: {response.answer}")
                
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n=== GEMINI PIPELINE SUMMARY ===")
        print(f"Successful queries: {success_count}/{len(test_queries)}")
        print(f"Success rate: {(success_count/len(test_queries)*100):.1f}%")
        
        return success_count == len(test_queries)
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_pipeline()
    sys.exit(0 if success else 1)