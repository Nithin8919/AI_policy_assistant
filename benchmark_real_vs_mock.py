#!/usr/bin/env python3
"""
Benchmark Real vs Mock Retrieval Performance
"""

import time
import sys
import os
from dotenv import load_dotenv

sys.path.append('/Users/nitin/Documents/AI policy Assistant')
load_dotenv()

from src.query_processing.qa_pipeline_multi_llm import QAPipeline, MultiLLMAnswerGenerator
from src.agents.enhanced_router import EnhancedRouter
from unittest.mock import Mock

def create_mock_router():
    """Create mock router for comparison"""
    mock_router = Mock()
    
    # Mock response structure
    mock_response = Mock()
    mock_response.retrieval_results = []
    mock_response.selected_agents = []
    mock_response.complexity = Mock(value="medium")
    mock_response.total_processing_time = 0.001
    mock_router.route_query.return_value = mock_response
    
    return mock_router

def create_real_router():
    """Create real router with Qdrant"""
    try:
        return EnhancedRouter(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY')
        )
    except Exception as e:
        print(f"❌ Real router failed: {e}")
        return None

def benchmark_query(pipeline, query, label):
    """Benchmark a single query"""
    print(f"\n🔍 Testing: {label}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        response = pipeline.answer_query(
            query=query,
            mode="normal_qa",
            top_k=5
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Analysis
        chunks_retrieved = response.retrieval_stats.get('chunks_retrieved', 0)
        agents_used = response.retrieval_stats.get('agents_used', [])
        confidence = response.confidence_score
        answer_length = len(response.answer)
        citations = response.citations.get('unique_sources_cited', 0)
        
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📊 Chunks retrieved: {chunks_retrieved}")
        print(f"🤖 Agents used: {', '.join(agents_used) if agents_used else 'None'}")
        print(f"🎯 Confidence: {confidence:.2%}")
        print(f"📝 Answer length: {answer_length} chars")
        print(f"📚 Citations: {citations}")
        print(f"📄 Answer preview: {response.answer[:100]}...")
        
        # Determine if real or mock
        is_real = (
            processing_time > 1.0 or  # Real retrieval takes time
            chunks_retrieved > 0 or   # Real retrieval finds chunks
            len(agents_used) > 0      # Real retrieval uses agents
        )
        
        status = "🟢 REAL RETRIEVAL" if is_real else "🔴 MOCK RETRIEVAL"
        print(f"🔍 Assessment: {status}")
        
        return {
            'time': processing_time,
            'chunks': chunks_retrieved,
            'agents': len(agents_used),
            'confidence': confidence,
            'answer_length': answer_length,
            'citations': citations,
            'is_real': is_real,
            'answer': response.answer
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'time': 0,
            'chunks': 0,
            'agents': 0,
            'confidence': 0,
            'answer_length': 0,
            'citations': 0,
            'is_real': False,
            'error': str(e)
        }

def main():
    print("🚀 REAL vs MOCK RETRIEVAL BENCHMARK")
    print("=" * 80)
    
    test_query = "What are the responsibilities of teachers in government schools?"
    
    # Test 1: Mock Router (what we had before)
    print("\n📋 Test 1: MOCK ROUTER")
    print("=" * 40)
    
    mock_router = create_mock_router()
    mock_pipeline = QAPipeline(
        router=mock_router,
        llm_provider='groq'
    )
    
    mock_results = benchmark_query(mock_pipeline, test_query, "Mock Retrieval System")
    
    # Test 2: Real Router (current system)
    print("\n📋 Test 2: REAL ROUTER")  
    print("=" * 40)
    
    real_router = create_real_router()
    if real_router:
        real_pipeline = QAPipeline(
            router=real_router,
            llm_provider='groq'
        )
        
        real_results = benchmark_query(real_pipeline, test_query, "Real Retrieval System")
    else:
        print("❌ Could not create real router")
        real_results = {'error': 'Router creation failed'}
    
    # Comparison
    print("\n📊 COMPARISON SUMMARY")
    print("=" * 80)
    
    if 'error' not in mock_results and 'error' not in real_results:
        print(f"{'Metric':<20} {'Mock System':<15} {'Real System':<15} {'Difference'}")
        print("-" * 70)
        
        metrics = [
            ('Processing Time', 'time', 's'),
            ('Chunks Retrieved', 'chunks', ''),
            ('Agents Used', 'agents', ''),
            ('Confidence', 'confidence', '%'),
            ('Answer Length', 'answer_length', ' chars'),
            ('Citations', 'citations', '')
        ]
        
        for name, key, unit in metrics:
            mock_val = mock_results.get(key, 0)
            real_val = real_results.get(key, 0)
            
            if key == 'confidence':
                mock_str = f"{mock_val:.1%}"
                real_str = f"{real_val:.1%}"
                diff = f"{(real_val - mock_val):.1%}"
            elif key == 'time':
                mock_str = f"{mock_val:.2f}{unit}"
                real_str = f"{real_val:.2f}{unit}"
                diff = f"+{(real_val - mock_val):.2f}{unit}"
            else:
                mock_str = f"{mock_val}{unit}"
                real_str = f"{real_val}{unit}"
                diff = f"+{(real_val - mock_val)}{unit}"
            
            print(f"{name:<20} {mock_str:<15} {real_str:<15} {diff}")
    
    # Verdict
    print("\n🎯 VERDICT")
    print("=" * 20)
    
    if real_results.get('is_real', False):
        print("✅ SUCCESS: Real retrieval system is operational!")
        print("📈 Performance characteristics indicate real Qdrant integration")
    elif real_results.get('time', 0) > 1.0:
        print("⚠️  PARTIAL: System connecting to Qdrant but no data retrieved")
        print("💡 Suggestion: Check if vector database has documents")
    else:
        print("❌ ISSUE: System appears to be using mock retrieval")
        print("🔧 Action needed: Verify Qdrant connection and data")
    
    # Save results
    with open('benchmark_results.txt', 'w') as f:
        f.write("Real vs Mock Retrieval Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Query tested: {test_query}\n\n")
        f.write(f"Mock system results: {mock_results}\n\n")
        f.write(f"Real system results: {real_results}\n\n")
    
    print(f"\n💾 Results saved to benchmark_results.txt")

if __name__ == "__main__":
    main()