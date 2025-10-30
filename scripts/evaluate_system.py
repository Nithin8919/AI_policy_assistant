"""Run all test queries"""
import sys
sys.path.append('src')

from src.evaluation.benchmark import run_benchmark
import json

def main():
    """Run evaluation"""
    with open("data/evaluation/test_queries.json", 'r') as f:
        queries = json.load(f)
    
    results = run_benchmark(queries)
    
    # Save results
    with open("outputs/test_results/results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()




