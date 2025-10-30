#!/usr/bin/env python3
"""Test script to check basic imports"""

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd()))
    
    print("Testing imports...")
    
    # Test query processing
    from src.query_processing.pipeline import QueryProcessingPipeline
    print("✅ QueryProcessingPipeline imported successfully")
    
    # Test embeddings
    from src.embeddings.embedder import Embedder
    from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType
    print("✅ Embedding components imported successfully")
    
    # Test agents
    from src.agents.legal_agent import LegalAgent
    from src.agents.go_agent import GOAgent
    print("✅ Agent components imported successfully")
    
    print("\n🎉 All imports successful!")
    
except Exception as e:
    print(f"❌ Import error: {str(e)}")
    import traceback
    traceback.print_exc()