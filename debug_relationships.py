#!/usr/bin/env python3
"""
Debug Relationship Extraction

This script helps debug why relationships aren't being extracted.
"""

import os
import sys
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.knowledge_graph.bridge_table_builder import BridgeTableBuilder, RelationshipType, EntityType

def test_relationship_extraction_directly():
    """Test relationship extraction on sample content"""
    
    print("🔍 Testing relationship extraction directly...")
    
    # Sample content with known relationships
    test_chunks = [
        {
            'chunk_id': 'test1',
            'doc_id': 'GO_54_2023',
            'content': 'This order G.O.Ms.No.54 supersedes G.O.Ms.No.45 dated 2023. Implementation of RTE Act Section 12(1)(c) is hereby ordered.',
            'metadata': {'doc_type': 'government_order'}
        },
        {
            'chunk_id': 'test2', 
            'doc_id': 'GO_123_2024',
            'content': 'In supersession of GO Ms No 123, this order is issued. As per Section 15 of the Act, Nadu-Nedu scheme implementation is required.',
            'metadata': {'doc_type': 'government_order'}
        },
        {
            'chunk_id': 'test3',
            'doc_id': 'legal_ref',
            'content': 'Section 12(1)(c) read with Section 15 provides implementation guidelines. Reference GO Ms No 789 for details.',
            'metadata': {'doc_type': 'legal_documents'}
        }
    ]
    
    # Initialize bridge builder
    bridge_builder = BridgeTableBuilder("./test_debug.db")
    
    # Manually extract entities and relationships from each chunk
    for chunk in test_chunks:
        print(f"\n📄 Testing chunk: {chunk['chunk_id']}")
        print(f"Content: {chunk['content']}")
        
        # Extract entities
        entities = bridge_builder._extract_entities_from_chunk(chunk)
        print(f"Entities found: {len(entities)}")
        for entity in entities:
            print(f"  - {entity.entity_type.value}: {entity.entity_value}")
        
        # Extract relationships
        relationships = bridge_builder._extract_relationships_from_chunk(chunk)
        print(f"Relationships found: {len(relationships)}")
        for rel in relationships:
            print(f"  - {rel.relationship_type.value}: {rel.source_entity_id} -> {rel.target_entity_id}")
    
    # Cleanup
    import os
    if os.path.exists("./test_debug.db"):
        os.remove("./test_debug.db")

def test_pattern_matching_step_by_step():
    """Test each step of pattern matching"""
    
    print("\n🧪 Testing pattern matching step by step...")
    
    test_content = "This order G.O.Ms.No.54 supersedes G.O.Ms.No.45 dated 2023."
    
    # Test entity patterns
    print(f"Testing content: {test_content}")
    
    print("\n1. Entity Extraction:")
    for entity_type, patterns in BridgeTableBuilder.ENTITY_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, test_content, re.IGNORECASE)
            for match in matches:
                print(f"  {entity_type.value}: {match.group(1)} (pattern: {pattern[:50]}...)")
    
    print("\n2. Relationship Extraction:")
    for rel_type, patterns in BridgeTableBuilder.RELATIONSHIP_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, test_content, re.IGNORECASE)
            for match in matches:
                print(f"  {rel_type.value}: {match.groups()} (pattern: {pattern[:50]}...)")

def test_entity_map_creation():
    """Test entity map creation for relationship determination"""
    
    print("\n🗺️  Testing entity map creation...")
    
    # Simulate entity extraction
    chunk = {
        'chunk_id': 'test_map',
        'doc_id': 'GO_test',
        'content': 'G.O.Ms.No.54 supersedes G.O.Ms.No.45 and implements Section 12(1)(c)',
        'metadata': {}
    }
    
    # Initialize bridge builder with temp DB
    bridge_builder = BridgeTableBuilder("./test_map.db")
    
    # Extract entities
    entities = bridge_builder._extract_entities_from_chunk(chunk)
    print(f"Entities extracted: {len(entities)}")
    
    # Create entity map like in relationship extraction
    entity_map = {e.entity_value: e.entity_id for e in entities}
    print(f"Entity map: {entity_map}")
    
    # Test relationship pattern matching
    content = chunk['content']
    for rel_type, patterns in BridgeTableBuilder.RELATIONSHIP_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                print(f"\nFound {rel_type.value} pattern: {match.groups()}")
                
                # Test entity determination
                source_entity, target_entity = bridge_builder._determine_relationship_entities(
                    match, content, rel_type, entity_map
                )
                print(f"  Source: {source_entity}")
                print(f"  Target: {target_entity}")
    
    # Cleanup
    import os
    if os.path.exists("./test_map.db"):
        os.remove("./test_map.db")

def main():
    """Main debug function"""
    
    print("=" * 80)
    print("🐛 DEBUGGING RELATIONSHIP EXTRACTION")
    print("=" * 80)
    
    # Test direct extraction
    test_relationship_extraction_directly()
    
    # Test pattern matching
    test_pattern_matching_step_by_step()
    
    # Test entity mapping  
    test_entity_map_creation()
    
    print("\n✅ Debug tests complete!")

if __name__ == "__main__":
    main()