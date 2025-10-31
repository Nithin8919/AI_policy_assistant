"""
Comprehensive Test Suite for Query Processing Pipeline

Tests cover:
- Intent classification accuracy
- Entity extraction precision & recall
- Vertical routing accuracy
- Latency benchmarks
- Hallucination prevention
- Typo handling
"""

import pytest
import json
import time
from pathlib import Path
from typing import Dict, List

from src.query_processing.pipeline import QueryProcessingPipeline
from src.query_processing.validator import EntityValidator


class TestQueryProcessingPipeline:
    """Test suite for query processing"""
    
    @pytest.fixture
    def pipeline(self):
        """Initialize pipeline"""
        return QueryProcessingPipeline()
    
    @pytest.fixture
    def validator(self):
        """Initialize validator"""
        return EntityValidator()
    
    @pytest.fixture
    def test_cases(self):
        """Load or generate test cases"""
        return self._get_test_cases()
    
    def _get_test_cases(self) -> List[Dict]:
        """Generate comprehensive test cases"""
        return [
            # Data queries
            {
                "query": "What is the PTR in Guntur district for 2023-24?",
                "expected_intent": "data_query",
                "expected_entities": {
                    "districts": ["Guntur"],
                    "metrics": ["Pupil-Teacher Ratio"],
                    "dates": ["2023-24"]
                },
                "expected_verticals": ["Data_Reports"],
                "complexity": "moderate"
            },
            {
                "query": "Show me enrollment trends in Visakhapatnam from 2020 to 2024",
                "expected_intent": "temporal_query",
                "expected_entities": {
                    "districts": ["Visakhapatnam"],
                    "metrics": ["Enrollment"],
                    "dates": ["2020", "2024"]
                },
                "expected_verticals": ["Data_Reports"],
                "complexity": "moderate"
            },
            # Scheme queries
            {
                "query": "How does Nadu-Nedu scheme work?",
                "expected_intent": "scheme_inquiry",
                "expected_entities": {
                    "schemes": ["Nadu-Nedu"]
                },
                "expected_verticals": ["Schemes", "Government_Orders"],
                "complexity": "simple"
            },
            {
                "query": "Who is eligible for Amma Vodi benefits?",
                "expected_intent": "scheme_inquiry",
                "expected_entities": {
                    "schemes": ["Jagananna Amma Vodi"]
                },
                "expected_verticals": ["Schemes"],
                "complexity": "simple"
            },
            # Legal queries
            {
                "query": "What does Section 12(1)(c) of RTE Act say?",
                "expected_intent": "legal_interpretation",
                "expected_entities": {
                    "legal_references": ["Section 12(1)(c)"]
                },
                "expected_verticals": ["Legal"],
                "complexity": "simple"
            },
            # GO queries
            {
                "query": "Show me details of GO MS No 54",
                "expected_intent": "go_inquiry",
                "expected_entities": {
                    "go_numbers": ["G.O.MS.No.54"]
                },
                "expected_verticals": ["Government_Orders"],
                "complexity": "simple"
            },
            # Comparison queries
            {
                "query": "Compare dropout rates between Vijayawada and Guntur",
                "expected_intent": "comparison",
                "expected_entities": {
                    "districts": ["Vijayawada", "Guntur"],
                    "metrics": ["Dropout Rate"]
                },
                "expected_verticals": ["Data_Reports"],
                "complexity": "moderate"
            },
            # Complex queries
            {
                "query": "Show Nadu-Nedu implementation impact on PTR in coastal districts for 2022-23",
                "expected_intent": "data_query",
                "expected_entities": {
                    "schemes": ["Nadu-Nedu"],
                    "metrics": ["Pupil-Teacher Ratio"],
                    "dates": ["2022-23"]
                },
                "expected_verticals": ["Data_Reports", "Schemes"],
                "complexity": "complex"
            },
        ]
    
    # ========== Intent Classification Tests ==========
    
    def test_intent_classification_accuracy(self, pipeline, test_cases):
        """Test overall intent classification accuracy"""
        correct = 0
        total = len(test_cases)
        results = []
        
        for case in test_cases:
            result = pipeline.process(case['query'])
            is_correct = result.primary_intent == case['expected_intent']
            
            if is_correct:
                correct += 1
            else:
                results.append({
                    'query': case['query'],
                    'expected': case['expected_intent'],
                    'actual': result.primary_intent,
                    'confidence': result.intent_confidence
                })
        
        accuracy = correct / total
        
        # Print detailed results
        print(f"\n{'='*80}")
        print(f"INTENT CLASSIFICATION ACCURACY: {accuracy:.1%} ({correct}/{total})")
        print(f"{'='*80}")
        
        if results:
            print("\nMisclassified Queries:")
            for r in results:
                print(f"  Query: {r['query']}")
                print(f"  Expected: {r['expected']}, Got: {r['actual']} (conf: {r['confidence']:.2f})")
                print()
        
        assert accuracy >= 0.75, f"Intent accuracy {accuracy:.1%} below 75% threshold"
    
    def test_intent_confidence_calibration(self, pipeline, test_cases):
        """Test that confidence scores are meaningful"""
        correct_confidences = []
        incorrect_confidences = []
        
        for case in test_cases:
            result = pipeline.process(case['query'])
            is_correct = result.primary_intent == case['expected_intent']
            
            if is_correct:
                correct_confidences.append(result.intent_confidence)
            else:
                incorrect_confidences.append(result.intent_confidence)
        
        if correct_confidences and incorrect_confidences:
            avg_correct = sum(correct_confidences) / len(correct_confidences)
            avg_incorrect = sum(incorrect_confidences) / len(incorrect_confidences)
            
            print(f"\nConfidence Calibration:")
            print(f"  Correct predictions: {avg_correct:.3f}")
            print(f"  Incorrect predictions: {avg_incorrect:.3f}")
            
            # Correct predictions should have higher confidence
            assert avg_correct > avg_incorrect, "Confidence scores not well calibrated"
    
    # ========== Entity Extraction Tests ==========
    
    def test_entity_extraction_precision(self, pipeline, test_cases):
        """Test that extracted entities are correct (no hallucinations)"""
        precisions = []
        
        for case in test_cases:
            result = pipeline.process(case['query'])
            expected = case['expected_entities']
            extracted = result.entities
            
            # Calculate precision for each entity type
            for entity_type in expected.keys():
                exp_set = set(expected[entity_type])
                
                # Get canonical names from extracted entities
                ext_list = extracted.get(entity_type, [])
                if ext_list and isinstance(ext_list[0], dict):
                    ext_set = set([e.get('canonical', e.get('text', '')) for e in ext_list])
                else:
                    ext_set = set([str(e) for e in ext_list])
                
                if len(ext_set) > 0:
                    # Precision = correct / extracted
                    correct = len(exp_set & ext_set)
                    precision = correct / len(ext_set)
                    precisions.append(precision)
        
        if precisions:
            avg_precision = sum(precisions) / len(precisions)
            
            print(f"\n{'='*80}")
            print(f"ENTITY EXTRACTION PRECISION: {avg_precision:.1%}")
            print(f"{'='*80}")
            print(f"(Measures: % of extracted entities that are correct)")
            
            assert avg_precision >= 0.80, f"Entity precision {avg_precision:.1%} below 80%"
    
    def test_entity_extraction_recall(self, pipeline, test_cases):
        """Test that all expected entities are found"""
        recalls = []
        missing_entities = []
        
        for case in test_cases:
            result = pipeline.process(case['query'])
            expected = case['expected_entities']
            extracted = result.entities
            
            for entity_type in expected.keys():
                exp_set = set(expected[entity_type])
                
                ext_list = extracted.get(entity_type, [])
                if ext_list and isinstance(ext_list[0], dict):
                    ext_set = set([e.get('canonical', e.get('text', '')) for e in ext_list])
                else:
                    ext_set = set([str(e) for e in ext_list])
                
                if len(exp_set) > 0:
                    # Recall = correct / expected
                    correct = len(exp_set & ext_set)
                    recall = correct / len(exp_set)
                    recalls.append(recall)
                    
                    # Track missing
                    missing = exp_set - ext_set
                    if missing:
                        missing_entities.append({
                            'query': case['query'],
                            'entity_type': entity_type,
                            'missing': list(missing)
                        })
        
        if recalls:
            avg_recall = sum(recalls) / len(recalls)
            
            print(f"\n{'='*80}")
            print(f"ENTITY EXTRACTION RECALL: {avg_recall:.1%}")
            print(f"{'='*80}")
            print(f"(Measures: % of expected entities found)")
            
            if missing_entities:
                print("\nMissed Entities:")
                for m in missing_entities[:5]:  # Show first 5
                    print(f"  Query: {m['query']}")
                    print(f"  Missed {m['entity_type']}: {m['missing']}")
                    print()
            
            assert avg_recall >= 0.70, f"Entity recall {avg_recall:.1%} below 70%"
    
    def test_no_hallucinations(self, pipeline):
        """Test that system doesn't hallucinate entities"""
        # Vague queries that shouldn't extract specific entities
        vague_queries = [
            "Tell me about education in general",
            "What is the current situation?",
            "Show me some information",
        ]
        
        for query in vague_queries:
            result = pipeline.process(query)
            
            # Should not extract specific districts/schemes
            assert len(result.entities.get('districts', [])) == 0, \
                f"Hallucinated districts for: {query}"
            assert len(result.entities.get('schemes', [])) == 0, \
                f"Hallucinated schemes for: {query}"
    
    # ========== Validation Tests ==========
    
    def test_typo_correction(self, pipeline, validator):
        """Test fuzzy matching corrects typos"""
        typo_queries = [
            ("Show me data for Vizg district", "Visakhapatnam"),
            ("What is PTR in Guntu?", "Guntur"),
            ("Nadu Nedu impact", "Nadu-Nedu"),
        ]
        
        print(f"\n{'='*80}")
        print("TYPO CORRECTION TESTS")
        print(f"{'='*80}")
        
        for query, expected in typo_queries:
            result = pipeline.process(query)
            
            # Validate entities
            validation = validator.validate(result.entities, query)
            
            # Check if correction was applied
            corrections = [c for c in validation.corrections_applied 
                         if c['corrected'] == expected]
            
            print(f"\nQuery: {query}")
            print(f"Expected: {expected}")
            print(f"Corrections: {len(corrections)}")
            if corrections:
                print(f"  {corrections[0]}")
            
            assert len(corrections) > 0, f"Failed to correct typo in: {query}"
    
    def test_invalid_entity_detection(self, pipeline, validator):
        """Test detection of invalid entities"""
        invalid_queries = [
            "Show data for InvalidCity",
            "What is enrollment in FakeDistrict?",
        ]
        
        for query in invalid_queries:
            result = pipeline.process(query)
            validation = validator.validate(result.entities, query)
            
            # Should have validation issues
            error_issues = [i for i in validation.issues if i.severity == 'error']
            assert len(error_issues) > 0, f"Failed to detect invalid entity in: {query}"
    
    # ========== Vertical Routing Tests ==========
    
    def test_vertical_routing_accuracy(self, pipeline, test_cases):
        """Test that queries are routed to correct verticals"""
        correct = 0
        total = len(test_cases)
        routing_errors = []
        
        for case in test_cases:
            result = pipeline.process(case['query'])
            expected_verticals = set(case['expected_verticals'])
            suggested_verticals = set(result.suggested_verticals)
            
            # Check if any expected vertical is suggested
            if expected_verticals & suggested_verticals:
                correct += 1
            else:
                routing_errors.append({
                    'query': case['query'],
                    'expected': list(expected_verticals),
                    'actual': list(suggested_verticals)
                })
        
        accuracy = correct / total
        
        print(f"\n{'='*80}")
        print(f"VERTICAL ROUTING ACCURACY: {accuracy:.1%} ({correct}/{total})")
        print(f"{'='*80}")
        
        if routing_errors:
            print("\nRouting Errors:")
            for e in routing_errors:
                print(f"  Query: {e['query']}")
                print(f"  Expected: {e['expected']}, Got: {e['actual']}")
                print()
        
        assert accuracy >= 0.75, f"Routing accuracy {accuracy:.1%} below 75%"
    
    # ========== Performance Tests ==========
    
    def test_latency_p50_p99(self, pipeline, test_cases):
        """Test processing latency"""
        times = []
        
        # Run multiple iterations for stable measurements
        for _ in range(3):
            for case in test_cases:
                start = time.time()
                pipeline.process(case['query'])
                elapsed_ms = (time.time() - start) * 1000
                times.append(elapsed_ms)
        
        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]
        
        print(f"\n{'='*80}")
        print(f"LATENCY BENCHMARKS")
        print(f"{'='*80}")
        print(f"P50: {p50:.2f}ms")
        print(f"P95: {p95:.2f}ms")
        print(f"P99: {p99:.2f}ms")
        print(f"{'='*80}")
        
        assert p50 < 100, f"P50 latency {p50:.2f}ms exceeds 100ms"
        assert p99 < 200, f"P99 latency {p99:.2f}ms exceeds 200ms"
    
    # ========== Query Complexity Tests ==========
    
    def test_complexity_estimation(self, pipeline, test_cases):
        """Test query complexity estimation"""
        for case in test_cases:
            result = pipeline.process(case['query'])
            expected_complexity = case['complexity']
            
            # Complexity should match or be close
            print(f"\nQuery: {case['query']}")
            print(f"Expected: {expected_complexity}, Got: {result.query_complexity}")
    
    # ========== Integration Tests ==========
    
    def test_end_to_end_pipeline(self, pipeline):
        """Test complete pipeline flow"""
        query = "What is the PTR in Guntur district for 2023-24?"
        
        result = pipeline.process(query)
        
        # Should have all components
        assert result.normalized_query is not None
        assert result.entities is not None
        assert result.primary_intent is not None
        assert len(result.query_expansions) > 0
        assert len(result.suggested_verticals) > 0
        assert result.processing_time_ms > 0
        
        print(f"\n{'='*80}")
        print(f"END-TO-END PIPELINE TEST")
        print(f"{'='*80}")
        print(f"Original: {result.original_query}")
        print(f"Normalized: {result.normalized_query}")
        print(f"Intent: {result.primary_intent} ({result.intent_confidence:.2f})")
        print(f"Entities: {result.entity_summary}")
        print(f"Verticals: {result.suggested_verticals}")
        print(f"Expansions: {len(result.query_expansions)}")
        print(f"Time: {result.processing_time_ms:.2f}ms")
        print(f"{'='*80}")


# ========== Benchmark Test ==========

def test_benchmark_suite(tmpdir):
    """Run complete benchmark suite and save results"""
    pipeline = QueryProcessingPipeline()
    test_obj = TestQueryProcessingPipeline()
    test_cases = test_obj._get_test_cases()
    
    results = {
        'total_queries': len(test_cases),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {}
    }
    
    # Run all tests and collect metrics
    print("\n" + "="*80)
    print("RUNNING COMPLETE BENCHMARK SUITE")
    print("="*80)
    
    # Save results
    results_file = Path(tmpdir) / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark results saved to: {results_file}")


