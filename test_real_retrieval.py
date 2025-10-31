"""
Test Script for Real Retrieval Integration
Verifies that the QA pipeline uses actual Qdrant retrieval, not mock data
"""

import os
import sys
import time
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from src.agents.enhanced_router import EnhancedRouter
from src.query_processing.qa_pipeline_multi_llm import QAPipeline

# Test queries that MUST retrieve real documents from knowledge base
TEST_QUERIES = [
    {
        "query": "What is Section 12(1)(c) of RTE Act?",
        "expected_keywords": ["25%", "reservation", "private", "schools", "economically weaker"],
        "expected_doc_type": "legal"
    },
    {
        "query": "What are the details of Nadu-Nedu scheme?",
        "expected_keywords": ["infrastructure", "schools", "renovation", "construction"],
        "expected_doc_type": "scheme"
    },
    {
        "query": "What are the responsibilities of School Management Committees under RTE?",
        "expected_keywords": ["SMC", "management", "committee", "monitoring", "school"],
        "expected_doc_type": "legal"
    },
    {
        "query": "What is the Amma Vodi scheme eligibility criteria?",
        "expected_keywords": ["mother", "children", "financial", "assistance", "eligibility"],
        "expected_doc_type": "scheme"
    },
    {
        "query": "What are teacher qualification requirements in AP?",
        "expected_keywords": ["B.Ed", "TET", "qualification", "teacher", "degree"],
        "expected_doc_type": "government_order"
    }
]

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text:^80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")

def print_warning(text):
    """Print warning message"""
    print(f"{Fore.YELLOW}⚠️  {text}{Style.RESET_ALL}")

def print_info(text):
    """Print info message"""
    print(f"{Fore.BLUE}ℹ️  {text}{Style.RESET_ALL}")

def verify_environment():
    """Verify all required environment variables are set"""
    print_header("ENVIRONMENT VERIFICATION")
    
    required_vars = {
        'QDRANT_URL': os.getenv('QDRANT_URL'),
        'QDRANT_API_KEY': os.getenv('QDRANT_API_KEY'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
    }
    
    all_set = True
    for var_name, var_value in required_vars.items():
        if var_value:
            print_success(f"{var_name}: {'*' * 10}...{var_value[-4:]}")
        else:
            if var_name in ['GROQ_API_KEY', 'GOOGLE_API_KEY']:
                print_warning(f"{var_name}: Not set (at least one LLM key required)")
            else:
                print_error(f"{var_name}: Not set (REQUIRED)")
                all_set = False
    
    if not (required_vars['GROQ_API_KEY'] or required_vars['GOOGLE_API_KEY']):
        print_error("At least one LLM API key (GROQ or GOOGLE) must be set!")
        all_set = False
    
    return all_set

def test_qdrant_connection():
    """Test connection to Qdrant"""
    print_header("QDRANT CONNECTION TEST")
    
    try:
        router = EnhancedRouter(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY')
        )
        
        # Try a simple query
        test_response = router.route_query("test query", top_k=1)
        
        print_success(f"Connected to Qdrant at {os.getenv('QDRANT_URL')}")
        print_success(f"Router initialized with {len(router.agents)} agents")
        print_info(f"Agents: {', '.join([a.agent_name for a in router.agents])}")
        
        return router
        
    except Exception as e:
        print_error(f"Failed to connect to Qdrant: {e}")
        return None

def test_query_with_provider(pipeline, test_case, provider):
    """Test a single query with specified provider"""
    query = test_case["query"]
    expected_keywords = test_case["expected_keywords"]
    expected_doc_type = test_case["expected_doc_type"]
    
    print(f"\n{Fore.MAGENTA}{'─'*80}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Query: {query}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Provider: {provider.upper()}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'─'*80}{Style.RESET_ALL}")
    
    start_time = time.time()
    
    try:
        response = pipeline.answer_query(
            query=query,
            mode="normal_qa",
            top_k=10
        )
        
        processing_time = time.time() - start_time
        
        # Verify it's real retrieval (should take 2-5 seconds)
        is_real_retrieval = processing_time >= 1.5
        
        if is_real_retrieval:
            print_success(f"Processing time: {processing_time:.2f}s (indicates REAL retrieval)")
        else:
            print_warning(f"Processing time: {processing_time:.2f}s (TOO FAST - might be mock data)")
        
        # Check if we got chunks
        chunks_retrieved = response.retrieval_stats.get('chunks_retrieved', 0)
        if chunks_retrieved > 0:
            print_success(f"Retrieved {chunks_retrieved} document chunks")
        else:
            print_error(f"No chunks retrieved! Using mock data?")
            return False
        
        # Check agents used
        agents_used = response.retrieval_stats.get('agents_used', [])
        if agents_used:
            print_success(f"Agents used: {', '.join(agents_used)}")
        else:
            print_warning("No agents listed (might indicate mock data)")
        
        # Verify citations reference real documents
        citations = response.citations.get('citation_details', {})
        if citations:
            print_success(f"Found {len(citations)} unique source citations")
            
            # Print first 3 sources
            for idx, (cite_num, source) in enumerate(list(citations.items())[:3]):
                doc_name = source.get('document', 'Unknown')
                doc_type = source.get('doc_type', 'unknown')
                score = source.get('score', 0.0)
                section = source.get('section', 'N/A')
                year = source.get('year', 'N/A')
                
                print(f"   [{cite_num}] {doc_name}")
                print(f"       Type: {doc_type} | Section: {section} | Year: {year} | Score: {score:.2f}")
                
                # Check if it's real (not "Mock" or "Unknown")
                if "Mock" in doc_name or doc_name == "Unknown Document":
                    print_error(f"       Source appears to be MOCK DATA!")
                    return False
        else:
            print_error("No citations found!")
            return False
        
        # Check for expected keywords in answer
        answer_lower = response.answer.lower()
        keywords_found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        
        if keywords_found:
            print_success(f"Found expected keywords: {', '.join(keywords_found[:3])}")
        else:
            print_warning(f"Expected keywords not found. Answer might be generic.")
        
        # Check confidence
        if response.confidence_score > 0.7:
            print_success(f"Confidence: {response.confidence_score:.2%} (HIGH)")
        elif response.confidence_score > 0.4:
            print_warning(f"Confidence: {response.confidence_score:.2%} (MEDIUM)")
        else:
            print_error(f"Confidence: {response.confidence_score:.2%} (LOW)")
        
        # Print answer snippet
        print(f"\n{Fore.CYAN}Answer (first 300 chars):{Style.RESET_ALL}")
        print(response.answer[:300] + "..." if len(response.answer) > 300 else response.answer)
        
        # VERIFICATION CRITERIA
        verification_passed = (
            is_real_retrieval and
            chunks_retrieved > 0 and
            len(citations) > 0 and
            response.confidence_score > 0.3 and
            all("Mock" not in source.get('document', '') for source in citations.values())
        )
        
        if verification_passed:
            print_success("\n✅ VERIFICATION PASSED: Using REAL retrieval!")
        else:
            print_error("\n❌ VERIFICATION FAILED: Might be using mock data!")
        
        return verification_passed
        
    except Exception as e:
        print_error(f"Query failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print_header("AI POLICY ASSISTANT - REAL RETRIEVAL VERIFICATION")
    
    # Step 1: Verify environment
    if not verify_environment():
        print_error("\n❌ Environment check failed. Please set required variables in .env")
        return
    
    # Step 2: Test Qdrant connection
    router = test_qdrant_connection()
    if not router:
        print_error("\n❌ Cannot proceed without Qdrant connection")
        return
    
    # Step 3: Initialize pipelines
    print_header("PIPELINE INITIALIZATION")
    
    pipelines = {}
    
    # Try Groq
    if os.getenv('GROQ_API_KEY'):
        try:
            pipelines['groq'] = QAPipeline(
                router=router,
                llm_provider='groq'
            )
            print_success("Groq pipeline initialized")
        except Exception as e:
            print_error(f"Groq initialization failed: {e}")
    
    # Try Gemini
    if os.getenv('GOOGLE_API_KEY'):
        try:
            pipelines['gemini'] = QAPipeline(
                router=router,
                llm_provider='gemini'
            )
            print_success("Gemini pipeline initialized")
        except Exception as e:
            print_error(f"Gemini initialization failed: {e}")
    
    if not pipelines:
        print_error("No LLM providers available!")
        return
    
    # Step 4: Run tests
    print_header("RUNNING RETRIEVAL TESTS")
    
    results = {provider: {'passed': 0, 'failed': 0} for provider in pipelines}
    
    for test_case in TEST_QUERIES:
        for provider, pipeline in pipelines.items():
            passed = test_query_with_provider(pipeline, test_case, provider)
            
            if passed:
                results[provider]['passed'] += 1
            else:
                results[provider]['failed'] += 1
            
            # Small delay between queries
            time.sleep(1)
    
    # Step 5: Summary
    print_header("TEST SUMMARY")
    
    for provider, result in results.items():
        total = result['passed'] + result['failed']
        success_rate = (result['passed'] / total * 100) if total > 0 else 0
        
        print(f"\n{provider.upper()}:")
        print(f"  Passed: {result['passed']}/{total}")
        print(f"  Failed: {result['failed']}/{total}")
        
        if success_rate >= 80:
            print_success(f"  Success Rate: {success_rate:.0f}% ✅")
        elif success_rate >= 50:
            print_warning(f"  Success Rate: {success_rate:.0f}% ⚠️")
        else:
            print_error(f"  Success Rate: {success_rate:.0f}% ❌")
    
    # Overall verdict
    print_header("VERDICT")
    
    all_providers_passed = all(
        (result['passed'] / (result['passed'] + result['failed'])) >= 0.8
        for result in results.values()
        if (result['passed'] + result['failed']) > 0
    )
    
    if all_providers_passed:
        print_success("🎉 ALL TESTS PASSED! System is using REAL retrieval from Qdrant!")
        print_info("You should now see:")
        print_info("  - Response times: 2-5 seconds (includes vector search)")
        print_info("  - Real document names in citations")
        print_info("  - Actual section numbers and years")
        print_info("  - Confidence scores > 70%")
    else:
        print_warning("⚠️  Some tests failed. Please review the results above.")
        print_info("Common issues:")
        print_info("  - Response time < 1.5s → Not calling Qdrant")
        print_info("  - 'Mock' in document names → Using fake data")
        print_info("  - No chunks retrieved → Router not connected")
        print_info("  - Low confidence < 30% → Poor retrieval quality")

if __name__ == "__main__":
    run_comprehensive_tests()
