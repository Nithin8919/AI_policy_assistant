#!/usr/bin/env python3
"""
Fix Relationship Extraction Logic

The issue is in the entity map key matching in _determine_relationship_entities.
This script creates a patched version that fixes the key matching.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.knowledge_graph.bridge_table_builder import BridgeTableBuilder, RelationshipType, EntityType
import re
from typing import Dict, Optional, Tuple

def create_fixed_entity_map(entities):
    """Create entity map with multiple key formats for better matching"""
    entity_map = {}
    
    for entity in entities:
        value = entity.entity_value
        entity_id = entity.entity_id
        entity_type = entity.entity_type
        
        # Add raw value
        entity_map[value] = entity_id
        
        # Add formatted keys for different entity types
        if entity_type == EntityType.GOVERNMENT_ORDER:
            entity_map[f"G.O.No.{value}"] = entity_id
            entity_map[f"GO.Ms.No.{value}"] = entity_id
            entity_map[f"Ms.No.{value}"] = entity_id
            entity_map[f"Rt.No.{value}"] = entity_id
            entity_map[f"Order No.{value}"] = entity_id
            
        elif entity_type == EntityType.LEGAL_SECTION:
            entity_map[f"Section {value}"] = entity_id
            entity_map[f"Sec.{value}"] = entity_id
            entity_map[f"Rule {value}"] = entity_id
            entity_map[f"Article {value}"] = entity_id
    
    return entity_map

def fixed_determine_relationship_entities(
    self, 
    match: re.Match, 
    content: str, 
    rel_type: RelationshipType,
    entity_map: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    """Fixed version of _determine_relationship_entities with better key matching"""
    
    if rel_type == RelationshipType.SUPERSEDES:
        # For supersession: current document supersedes mentioned GO
        target_go = match.group(1)
        
        # Try multiple key formats
        target_entity = None
        for key_format in [target_go, f"G.O.No.{target_go}", f"GO.Ms.No.{target_go}", f"Ms.No.{target_go}", f"Order No.{target_go}"]:
            if key_format in entity_map:
                target_entity = entity_map[key_format]
                break
        
        # Source is the current document (infer from context)
        source_entity = self._infer_current_document_entity(content, entity_map)
        
        return source_entity, target_entity
    
    elif rel_type in [RelationshipType.REFERENCES, RelationshipType.IMPLEMENTS]:
        # For references: current section references target section  
        target_section = match.group(1)
        
        # Try multiple key formats
        target_entity = None
        for key_format in [target_section, f"Section {target_section}", f"Sec.{target_section}", f"Rule {target_section}"]:
            if key_format in entity_map:
                target_entity = entity_map[key_format]
                break
        
        source_entity = self._infer_current_section_entity(match, content, entity_map)
        
        return source_entity, target_entity
    
    elif rel_type == RelationshipType.CROSS_REFERENCES:
        # For cross-references: bidirectional relationship
        if len(match.groups()) >= 2:
            section1 = match.group(1)
            section2 = match.group(2)
            
            # Try multiple key formats for both sections
            entity1 = None
            for key_format in [section1, f"Section {section1}", f"Sec.{section1}"]:
                if key_format in entity_map:
                    entity1 = entity_map[key_format]
                    break
            
            entity2 = None
            for key_format in [section2, f"Section {section2}", f"Sec.{section2}"]:
                if key_format in entity_map:
                    entity2 = entity_map[key_format]
                    break
            
            return entity1, entity2
    
    return None, None

def fixed_infer_current_document_entity(self, content: str, entity_map: Dict[str, str]) -> Optional[str]:
    """Fixed version that uses the enhanced entity map"""
    
    # Look for GO number in the beginning of content
    for pattern in BridgeTableBuilder.ENTITY_PATTERNS[EntityType.GOVERNMENT_ORDER]:
        match = re.search(pattern, content[:500], re.IGNORECASE)
        if match:
            go_num = match.group(1)
            
            # Try multiple key formats
            for key_format in [go_num, f"G.O.No.{go_num}", f"GO.Ms.No.{go_num}", f"Ms.No.{go_num}"]:
                if key_format in entity_map:
                    return entity_map[key_format]
    
    return None

def patch_bridge_table_builder():
    """Apply the fixes to BridgeTableBuilder"""
    
    print("🔧 Applying relationship extraction fixes...")
    
    # Patch the methods
    BridgeTableBuilder._determine_relationship_entities = fixed_determine_relationship_entities
    BridgeTableBuilder._infer_current_document_entity = fixed_infer_current_document_entity
    
    # Patch the entity extraction to use fixed entity map
    original_extract_relationships = BridgeTableBuilder._extract_relationships_from_chunk
    
    def fixed_extract_relationships_from_chunk(self, chunk):
        """Fixed version that uses enhanced entity map"""
        relationships = []
        content = chunk.get('content', '')
        doc_id = chunk.get('doc_id', '')
        chunk_id = chunk.get('chunk_id', '')
        
        # Extract entities first to establish source/target
        chunk_entities = self._extract_entities_from_chunk(chunk)
        
        # Create enhanced entity map
        entity_map = create_fixed_entity_map(chunk_entities)
        
        for rel_type, patterns in self.RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    # Determine source and target entities
                    source_entity, target_entity = self._determine_relationship_entities(
                        match, content, rel_type, entity_map
                    )
                    
                    if source_entity and target_entity:
                        relationship_id = f"{rel_type.value}_{source_entity}_{target_entity}_{chunk_id}"
                        
                        confidence = self._calculate_relationship_confidence(match, content, rel_type)
                        
                        from src.knowledge_graph.bridge_table_builder import Relationship
                        
                        relationship = Relationship(
                            relationship_id=relationship_id,
                            source_entity_id=source_entity,
                            target_entity_id=target_entity,
                            relationship_type=rel_type,
                            confidence=confidence,
                            evidence_chunk_id=chunk_id,
                            metadata={
                                'evidence_text': match.group(0),
                                'context': content[max(0, match.start()-100):match.end()+100],
                                'doc_id': doc_id
                            }
                        )
                        
                        relationships.append(relationship)
        
        return relationships
    
    BridgeTableBuilder._extract_relationships_from_chunk = fixed_extract_relationships_from_chunk
    
    print("✅ Relationship extraction fixes applied")

def test_fixed_extraction():
    """Test the fixed extraction"""
    
    print("🧪 Testing fixed relationship extraction...")
    
    test_chunk = {
        'chunk_id': 'test_fixed',
        'doc_id': 'GO_54_2023',
        'content': 'G.O.Ms.No.54 supersedes G.O.Ms.No.45 dated 2023. Section 12(1)(c) read with Section 15.',
        'metadata': {'doc_type': 'government_order'}
    }
    
    # Initialize bridge builder
    bridge_builder = BridgeTableBuilder("./test_fixed.db")
    
    # Extract entities
    entities = bridge_builder._extract_entities_from_chunk(test_chunk)
    print(f"Entities: {[(e.entity_type.value, e.entity_value) for e in entities]}")
    
    # Extract relationships
    relationships = bridge_builder._extract_relationships_from_chunk(test_chunk)
    print(f"Relationships: {len(relationships)}")
    
    for rel in relationships:
        print(f"  - {rel.relationship_type.value}: {rel.source_entity_id} -> {rel.target_entity_id}")
        print(f"    Evidence: {rel.metadata['evidence_text']}")
    
    # Cleanup
    import os
    if os.path.exists("./test_fixed.db"):
        os.remove("./test_fixed.db")

def main():
    """Main function"""
    
    print("=" * 80)
    print("🔧 FIXING RELATIONSHIP EXTRACTION")
    print("=" * 80)
    
    # Apply fixes
    patch_bridge_table_builder()
    
    # Test fixes
    test_fixed_extraction()
    
    print("\n✅ Relationship extraction fixes complete!")
    print("Now run rebuild_bridge_table_enhanced.py again to rebuild with fixes.")

if __name__ == "__main__":
    main()