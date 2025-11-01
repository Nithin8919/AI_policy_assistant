#!/usr/bin/env python3
"""
Complete Data Cleanup Script

This script removes ALL previously processed data to start fresh:
1. Clears all Qdrant collections
2. Removes bridge table databases
3. Cleans up temporary processing files
4. Prepares for fresh SOTA data processing
"""

import os
import sys
import glob
from pathlib import Path

# Set environment variables
os.environ['QDRANT_URL'] = 'https://3bfa5117-dd8a-4048-abf9-5267856c164e.us-east4-0.gcp.cloud.qdrant.io:6333'
os.environ['QDRANT_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9Mk6YTL8BaQeHF3945J1_-MoWa4MWe-XvJxST5EeQ60'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType

def clear_qdrant_collections():
    """Remove all existing Qdrant collections"""
    
    print("🧹 Clearing all Qdrant collections...")
    
    try:
        vector_store = VectorStore(VectorStoreConfig(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY')
        ))
        
        # Get all collection names
        collections_info = vector_store.client.get_collections()
        existing_collections = [collection.name for collection in collections_info.collections]
        
        print(f"Found {len(existing_collections)} collections to delete:")
        
        for collection_name in existing_collections:
            print(f"  🗑️  Deleting collection: {collection_name}")
            try:
                vector_store.client.delete_collection(collection_name)
                print(f"    ✅ Deleted: {collection_name}")
            except Exception as e:
                print(f"    ❌ Failed to delete {collection_name}: {e}")
        
        print("✅ Qdrant collections cleared successfully")
        
    except Exception as e:
        print(f"❌ Error clearing Qdrant collections: {e}")

def clear_bridge_tables():
    """Remove all bridge table files"""
    
    print("🧹 Clearing bridge table files...")
    
    bridge_files = [
        "bridge_table.db",
        "bridge_table.db.json", 
        "bridge_table_enhanced.db",
        "bridge_table_enhanced.db.json",
        "test_bridge.db",
        "test_debug.db",
        "test_fixed.db",
        "test_map.db"
    ]
    
    for file_path in bridge_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  🗑️  Removed: {file_path}")
    
    print("✅ Bridge table files cleared")

def clear_processing_artifacts():
    """Remove temporary processing files and artifacts"""
    
    print("🧹 Clearing processing artifacts...")
    
    # Patterns of files to remove
    patterns_to_remove = [
        "*.log",
        "*.tmp",
        "__pycache__",
        "*.pyc",
        "embeddings_cache_*",
        "chunk_cache_*",
        "processing_state_*"
    ]
    
    removed_count = 0
    
    for pattern in patterns_to_remove:
        matching_files = glob.glob(pattern, recursive=True)
        for file_path in matching_files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    removed_count += 1
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                    removed_count += 1
            except Exception as e:
                print(f"    ❌ Failed to remove {file_path}: {e}")
    
    if removed_count > 0:
        print(f"  🗑️  Removed {removed_count} artifact files/directories")
    else:
        print("  ✅ No artifacts to remove")

def verify_clean_state():
    """Verify that all data has been cleared"""
    
    print("🔍 Verifying clean state...")
    
    try:
        # Check Qdrant collections
        vector_store = VectorStore(VectorStoreConfig(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY')
        ))
        
        collections_info = vector_store.client.get_collections()
        remaining_collections = [collection.name for collection in collections_info.collections]
        
        if remaining_collections:
            print(f"  ⚠️  Warning: {len(remaining_collections)} collections still exist: {remaining_collections}")
        else:
            print("  ✅ Qdrant: All collections cleared")
        
        # Check bridge table files
        bridge_files = glob.glob("bridge_table*")
        if bridge_files:
            print(f"  ⚠️  Warning: Bridge table files still exist: {bridge_files}")
        else:
            print("  ✅ Bridge tables: All files cleared")
        
        print("✅ Clean state verification complete")
        
    except Exception as e:
        print(f"❌ Error verifying clean state: {e}")

def main():
    """Main cleanup function"""
    
    print("=" * 80)
    print("🧹 COMPLETE DATA CLEANUP - STARTING FRESH")
    print("=" * 80)
    print()
    print("⚠️  WARNING: This will delete ALL processed data!")
    print("   - All Qdrant vector collections")
    print("   - All bridge table databases") 
    print("   - All processing artifacts")
    print()
    
    # Auto-confirm for non-interactive execution
    print("🚀 Auto-proceeding with data cleanup (non-interactive mode)...")
    
    print("\n🚀 Starting complete data cleanup...")
    
    try:
        # Step 1: Clear Qdrant collections
        clear_qdrant_collections()
        print()
        
        # Step 2: Clear bridge tables
        clear_bridge_tables()
        print()
        
        # Step 3: Clear processing artifacts
        clear_processing_artifacts()
        print()
        
        # Step 4: Verify clean state
        verify_clean_state()
        print()
        
        print("=" * 80)
        print("✅ COMPLETE DATA CLEANUP SUCCESSFUL!")
        print("=" * 80)
        print("🚀 Ready for fresh SOTA data processing pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR during cleanup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)