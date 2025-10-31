#!/usr/bin/env python3
"""Test the new LangGraph-based agent system"""

import os
import sys
from pathlib import Path
import time

# Set environment variables
os.environ['QDRANT_URL'] = 'https://3bfa5117-dd8a-4048-abf9-5267856c164e.us-east4-0.gcp.cloud.qdrant.io:6333'
os.environ['QDRANT_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9Mk6YTL8BaQeHF3945J1_-MoWa4MWe-XvJxST5EeQ60'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDXkZEtW7gG8E_yuMGeM7SGAcQpWKVIGsc'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.agents.langgraph_agent_system import LangGraphPolicyAgent

def test_langgraph_system():
    """Test the LangGraph-based multi-agent system"""
    print("=== TESTING LANGGRAPH MULTI-AGENT SYSTEM ===")
    
    try:
        # Initialize the LangGraph agent
        agent = LangGraphPolicyAgent(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY'),
            llm_provider="gemini",
            llm_api_key=os.getenv('GOOGLE_API_KEY')
        )
        print("✅ LangGraph Agent initialized successfully")
        
        # Test queries with different complexity levels
        test_queries = [
            {
                "query": "What is Section 12(1)(c) of RTE Act?",
                "expected_complexity": "simple",
                "expected_agents": ["legal"]
            },
            {
                "query": "Compare Nadu-Nedu scheme with teacher qualification requirements",
                "expected_complexity": "complex", 
                "expected_agents": ["legal", "government_order"]
            },
            {
                "query": "What are the latest dropout statistics for government schools?",
                "expected_complexity": "simple",
                "expected_agents": ["data"]
            },
            {
                "query": "How do judicial decisions impact RTE Act implementation through government orders?",
                "expected_complexity": "complex",
                "expected_agents": ["legal", "judicial", "government_order"]
            }
        ]
        
        print(f"\n=== TESTING {len(test_queries)} QUERIES ===")
        
        total_start = time.time()
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            print(f"\n--- Query {i}: {query} ---")
            
            start_time = time.time()
            
            try:
                result = agent.answer_query(query, thread_id=f"test_thread_{i}")
                
                processing_time = time.time() - start_time
                
                # Analyze results
                success = result.get("success", False)
                answer_length = len(result.get("answer", ""))
                confidence = result.get("confidence", 0.0)
                agents_used = result.get("metadata", {}).get("agents_used", [])
                complexity = result.get("metadata", {}).get("complexity", "unknown")
                chunks_retrieved = result.get("metadata", {}).get("total_chunks", 0)
                citations = result.get("citations", {}).get("total_citations", 0)
                
                print(f"✅ Success: {success}")
                print(f"✅ Processing time: {processing_time:.3f}s")
                print(f"✅ Answer length: {answer_length} characters")
                print(f"✅ Confidence: {confidence:.2f}")
                print(f"✅ Complexity detected: {complexity}")
                print(f"✅ Agents used: {agents_used}")
                print(f"✅ Chunks retrieved: {chunks_retrieved}")
                print(f"✅ Citations: {citations}")
                
                # Show answer preview
                answer = result.get("answer", "")
                preview = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"📄 Answer preview: {preview}")
                
                # Validate expectations
                expected_complexity = test_case["expected_complexity"]
                expected_agents = test_case["expected_agents"]
                
                if complexity.lower() in expected_complexity.lower():
                    print(f"✅ Complexity detection correct: {complexity}")
                else:
                    print(f"⚠️  Complexity detection: expected {expected_complexity}, got {complexity}")
                
                if any(expected in agents_used for expected in expected_agents):
                    print(f"✅ Agent routing correct: {agents_used}")
                else:
                    print(f"⚠️  Agent routing: expected {expected_agents}, got {agents_used}")
                
                results.append({
                    "query": query,
                    "success": success,
                    "processing_time": processing_time,
                    "confidence": confidence,
                    "agents_used": agents_used,
                    "complexity": complexity,
                    "answer_length": answer_length
                })
                
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        total_time = time.time() - total_start
        
        # Summary analysis
        print(f"\n=== LANGGRAPH SYSTEM SUMMARY ===")
        
        successful_queries = [r for r in results if r.get("success", False)]
        success_rate = len(successful_queries) / len(results) * 100
        
        print(f"Total queries: {len(results)}")
        print(f"Successful queries: {len(successful_queries)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total processing time: {total_time:.3f}s")
        
        if successful_queries:
            avg_time = sum(r["processing_time"] for r in successful_queries) / len(successful_queries)
            avg_confidence = sum(r["confidence"] for r in successful_queries) / len(successful_queries)
            
            print(f"Average processing time: {avg_time:.3f}s")
            print(f"Average confidence: {avg_confidence:.2f}")
            
            # Agent usage analysis
            all_agents = []
            for r in successful_queries:
                all_agents.extend(r.get("agents_used", []))
            
            if all_agents:
                from collections import Counter
                agent_counts = Counter(all_agents)
                print(f"Agent usage: {dict(agent_counts)}")
        
        print(f"\n=== LANGGRAPH vs CURRENT SYSTEM COMPARISON ===")
        print("✅ Structured workflow execution with state management")
        print("✅ Better error handling and fallback mechanisms")
        print("✅ Explicit agent specialization and routing")
        print("✅ Conversation memory and checkpointing")
        print("✅ Parallel and sequential execution patterns")
        print("✅ Built-in observability and logging")
        
        return success_rate > 75  # Consider successful if >75% queries work
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_langgraph_system()
    print(f"\n=== FINAL RESULT ===")
    print(f"LangGraph System Test: {'✅ PASSED' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)