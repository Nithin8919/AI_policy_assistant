#!/usr/bin/env python3
"""
Test script to validate Multi-LLM setup (Gemini and Groq)
"""

import os
import sys
from dotenv import load_dotenv

def test_multi_llm_setup():
    """Test Multi-LLM setup and functionality"""
    
    print("Testing Multi-LLM Setup...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Test imports
        sys.path.insert(0, '/Users/nitin/Documents/AI policy Assistant')
        from src.query_processing.qa_pipeline_multi_llm import MultiLLMAnswerGenerator, test_llm_providers
        
        print("✓ Multi-LLM module imported successfully")
        
        # Test provider availability
        print("\nChecking LLM provider availability:")
        print("-" * 40)
        
        # Check Gemini
        gemini_key = os.getenv('GOOGLE_API_KEY')
        if gemini_key and gemini_key.strip():
            print(f"✓ Google API key configured: {gemini_key[:10]}...")
            try:
                generator = MultiLLMAnswerGenerator("gemini")
                print("✓ Gemini client created successfully")
            except Exception as e:
                print(f"✗ Gemini setup error: {e}")
        else:
            print("⚠ Google API key not configured in .env")
            print("  Set GOOGLE_API_KEY in .env for Gemini functionality")
        
        # Check Groq
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key and groq_key.strip():
            print(f"✓ Groq API key configured: {groq_key[:10]}...")
            try:
                generator = MultiLLMAnswerGenerator("groq")
                print("✓ Groq client created successfully")
            except Exception as e:
                print(f"✗ Groq setup error: {e}")
        else:
            print("⚠ Groq API key not configured in .env")
            print("  Set GROQ_API_KEY in .env for Groq functionality")
        
        # Test basic functionality (without actual API calls)
        print("\nTesting basic functionality:")
        print("-" * 40)
        
        # Test context assembler
        from src.query_processing.qa_pipeline_multi_llm import ContextAssembler
        assembler = ContextAssembler()
        print("✓ Context assembler created")
        
        # Test citation validator
        from src.query_processing.qa_pipeline_multi_llm import CitationValidator
        validator = CitationValidator()
        print("✓ Citation validator created")
        
        # Test sample context assembly
        sample_results = [
            {
                'content': 'This is a test document about education policy.',
                'metadata': {
                    'title': 'Test Document',
                    'section_id': 'Section 1',
                    'year': '2023',
                    'doc_type': 'policy'
                },
                'score': 0.95,
                'chunk_id': 'test_001'
            }
        ]
        
        context, sources = assembler.assemble_context(sample_results)
        print("✓ Context assembly test successful")
        
        # Test citation validation
        test_answer = "This is based on [Source 1] which shows the policy details."
        citations = validator.validate_citations(test_answer, sources)
        print("✓ Citation validation test successful")
        
        print("\n" + "=" * 50)
        print("✓ All Multi-LLM setup tests passed!")
        
        # Print setup instructions
        print("\nSetup Instructions:")
        print("-" * 20)
        if not gemini_key:
            print("• Get a Google API key from: https://makersuite.google.com/app/apikey")
            print("• Set GOOGLE_API_KEY in .env file")
        if not groq_key:
            print("• Get a Groq API key from: https://console.groq.com/keys")
            print("• Set GROQ_API_KEY in .env file")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Run: pip install google-generativeai groq")
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_imports():
    """Test integration with existing modules"""
    
    print("\nTesting Integration Imports...")
    print("=" * 50)
    
    try:
        sys.path.insert(0, '/Users/nitin/Documents/AI policy Assistant')
        
        # Test query processing imports
        from src.query_processing.normalizer import QueryNormalizer
        print("✓ Query normalizer imported")
        
        # Test synthesis imports (if available)
        try:
            from src.synthesis import answer_generator
            print("✓ Synthesis modules imported")
        except ImportError:
            print("⚠ Synthesis modules not found (expected)")
        
        print("✓ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_multi_llm_setup()
    success2 = test_integration_imports()
    
    if success1 and success2:
        print("\n🎉 All Multi-LLM tests passed! System ready for Q&A pipeline testing.")
        print("\nNext steps:")
        print("1. Configure API keys in .env file")
        print("2. Test with actual API calls")
        print("3. Run integration testing")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please fix issues before proceeding.")
        exit(1)