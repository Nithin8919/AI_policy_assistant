from src.query_processing.pipeline import QueryProcessingPipeline
import json

# Test queries from different categories
test_queries = [
    "What is the PTR in Guntur district for 2023-24?",
    "Show Nadu-Nedu enrollment statistics",
    "How does Amma Vodi scheme work?",
    "What does Section 12(1)(c) of RTE Act say?",
    "List all districts in AP",
    "Compare dropout rates between Vijayawada and Visakhapatnam",
    "Show me GO MS No 54 details",
    "Why was Nadu-Nedu extended in 2022?",
]

pipeline = QueryProcessingPipeline()

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)
    
    result = pipeline.process(query)
    
    print(f"Normalized: {result.normalized_query}")
    print(f"Intent: {result.primary_intent} ({result.intent_confidence:.2f})")
    print(f"Entities: {result.entity_summary}")
    print(f"Verticals: {result.suggested_verticals}")
    print(f"Complexity: {result.query_complexity}")
    print(f"Expansions: {len(result.query_expansions)}")
    print(f"Time: {result.processing_time_ms:.2f}ms")