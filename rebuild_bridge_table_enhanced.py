#!/usr/bin/env python3
"""
Rebuild Bridge Table with Enhanced Patterns

This script rebuilds the bridge table using enhanced relationship
extraction patterns for better supersession and cross-reference detection.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import re

# Set environment variables
os.environ['QDRANT_URL'] = 'https://3bfa5117-dd8a-4048-abf9-5267856c164e.us-east4-0.gcp.cloud.qdrant.io:6333'
os.environ['QDRANT_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9Mk6YTL8BaQeHF3945J1_-MoWa4MWe-XvJxST5EeQ60'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType
from src.knowledge_graph.bridge_table_builder import BridgeTableBuilder, RelationshipType, EntityType, Relationship
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
    match, 
    content: str, 
    rel_type: RelationshipType,
    entity_map
):
    """Fixed version of _determine_relationship_entities"""
    
    if rel_type == RelationshipType.SUPERSEDES:
        target_go = match.group(1)
        
        # Try multiple key formats
        target_entity = None
        for key_format in [target_go, f"G.O.No.{target_go}", f"GO.Ms.No.{target_go}", f"Ms.No.{target_go}", f"Order No.{target_go}"]:
            if key_format in entity_map:
                target_entity = entity_map[key_format]
                break
        
        source_entity = self._infer_current_document_entity(content, entity_map)
        return source_entity, target_entity
    
    elif rel_type in [RelationshipType.REFERENCES, RelationshipType.IMPLEMENTS]:
        target_section = match.group(1)
        
        target_entity = None
        for key_format in [target_section, f"Section {target_section}", f"Sec.{target_section}", f"Rule {target_section}"]:
            if key_format in entity_map:
                target_entity = entity_map[key_format]
                break
        
        source_entity = self._infer_current_section_entity(match, content, entity_map)
        return source_entity, target_entity
    
    return None, None

def fixed_infer_current_document_entity(self, content: str, entity_map):
    """Fixed version that uses the enhanced entity map"""
    
    for pattern in BridgeTableBuilder.ENTITY_PATTERNS[EntityType.GOVERNMENT_ORDER]:
        match = re.search(pattern, content[:500], re.IGNORECASE)
        if match:
            go_num = match.group(1)
            
            for key_format in [go_num, f"G.O.No.{go_num}", f"GO.Ms.No.{go_num}", f"Ms.No.{go_num}"]:
                if key_format in entity_map:
                    return entity_map[key_format]
    
    return None

def fixed_extract_relationships_from_chunk(self, chunk):
    """Fixed version that uses enhanced entity map"""
    relationships = []
    content = chunk.get('content', '')
    doc_id = chunk.get('doc_id', '')
    chunk_id = chunk.get('chunk_id', '')
    
    chunk_entities = self._extract_entities_from_chunk(chunk)
    entity_map = create_fixed_entity_map(chunk_entities)
    
    for rel_type, patterns in self.RELATIONSHIP_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                source_entity, target_entity = self._determine_relationship_entities(
                    match, content, rel_type, entity_map
                )
                
                if source_entity and target_entity:
                    relationship_id = f"{rel_type.value}_{source_entity}_{target_entity}_{chunk_id}"
                    confidence = self._calculate_relationship_confidence(match, content, rel_type)
                    
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

def apply_enhanced_patterns():
    """Apply enhanced patterns and fixes to BridgeTableBuilder"""
    
    print("🔧 Applying enhanced relationship extraction patterns and fixes...")
    
    # Enhanced GO patterns
    enhanced_go_patterns = [
        r'G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'Government\s+Order.*?No\.?\s*(\d+)',
        r'GO\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'Order\s+No\.?\s*(\d+)',
        r'(?:Ms\.?|Rt\.?)\s*No\.?\s*(\d+)'
    ]
    
    # Enhanced supersession patterns
    enhanced_supersession_patterns = [
        r'supersedes?\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'in\s+supersession\s+of\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'replaces?\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'cancels?\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'(?:this\s+order\s+)?supersedes?\s+(?:the\s+)?(?:order\s+)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'(?:hereby\s+)?supersedes?\s+(?:all\s+)?(?:previous\s+)?(?:orders?\s+)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'(?:vide|ref|reference)\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+).*?(?:is|stands?)\s+(?:superseded|cancelled|replaced)',
        r'(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+).*?(?:is|stands?)\s+(?:hereby\s+)?(?:superseded|cancelled|replaced)'
    ]
    
    # Enhanced reference patterns
    enhanced_reference_patterns = [
        r'(?:refer|see|vide|ref)\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'as\s+per\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'in\s+accordance\s+with\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'pursuant\s+to\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'under\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
        r'subject\s+to\s+(?:G\.?O\.?\s*)?(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)'
    ]
    
    # Enhanced section patterns
    enhanced_section_patterns = [
        r'Section\s+(\d+(?:\([a-z0-9]\))*(?:\([ivx]+\))*)',
        r'Sec\.\s*(\d+(?:\([a-z0-9]\))*)',
        r'Article\s+(\d+[A-Z]*)',
        r'Rule\s+(\d+(?:\([a-z0-9]\))*)',
        r'Chapter\s+(\d+)',
        r'Para\s+(\d+(?:\.\d+)*)',
        r'Clause\s+(\d+(?:\([a-z]\))*)'
    ]
    
    # Enhanced scheme patterns for AP
    enhanced_scheme_patterns = [
        r'(Nadu[\\-\\s]?Nedu)',
        r'(Amma\\s+Vodi)',
        r'(Jagananna\\s+[A-Za-z\\s]+)',
        r'(Mid[\\-\\s]?Day[\\-\\s]?Meal)',
        r'(Sarva\\s+Shiksha\\s+Abhiyan)',
        r'(SSA)',
        r'(RMSA)',
        r'(Samagra\\s+Shiksha)',
        r'(PM\\s+POSHAN)',
        r'(Gorumudda)',
        r'(Vidya\\s+Volunteers?)',
        r'(SMC)'
    ]
    
    # Apply to BridgeTableBuilder
    BridgeTableBuilder.ENTITY_PATTERNS[EntityType.GOVERNMENT_ORDER] = enhanced_go_patterns
    BridgeTableBuilder.ENTITY_PATTERNS[EntityType.LEGAL_SECTION] = enhanced_section_patterns  
    BridgeTableBuilder.ENTITY_PATTERNS[EntityType.SCHEME] = enhanced_scheme_patterns
    
    BridgeTableBuilder.RELATIONSHIP_PATTERNS[RelationshipType.SUPERSEDES] = enhanced_supersession_patterns
    BridgeTableBuilder.RELATIONSHIP_PATTERNS[RelationshipType.REFERENCES] = enhanced_reference_patterns
    
    # Apply relationship extraction fixes
    BridgeTableBuilder._determine_relationship_entities = fixed_determine_relationship_entities
    BridgeTableBuilder._infer_current_document_entity = fixed_infer_current_document_entity
    BridgeTableBuilder._extract_relationships_from_chunk = fixed_extract_relationships_from_chunk
    
    print("✅ Enhanced patterns and fixes applied successfully")

def fetch_all_chunks_from_qdrant() -> List[Dict[str, Any]]:
    """Fetch all document chunks from Qdrant collections"""
    
    print("🔍 Connecting to Qdrant and fetching all document chunks...")
    
    vector_store = VectorStore(VectorStoreConfig(
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY')
    ))
    
    all_chunks = []
    
    # Get all document types
    doc_types = [
        DocumentType.LEGAL_DOCUMENTS,
        DocumentType.GOVERNMENT_ORDERS, 
        DocumentType.JUDICIAL_DOCUMENTS,
        DocumentType.DATA_REPORTS,
        DocumentType.EXTERNAL_SOURCES
    ]
    
    for doc_type in doc_types:
        try:
            collection_name = vector_store.get_collection_name(doc_type)
            print(f"  📚 Fetching from {collection_name}...")
            
            scroll_result = vector_store.client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=10000,
                with_payload=True
            )
            
            points = scroll_result[0]
            print(f"    Found {len(points)} chunks")
            
            for point in points:
                chunk = {
                    'chunk_id': str(point.id),
                    'doc_id': point.payload.get('doc_id', ''),
                    'content': point.payload.get('content', ''),
                    'metadata': {
                        'doc_type': doc_type.value,
                        'year': point.payload.get('year'),
                        'priority': point.payload.get('priority'),
                        'department': point.payload.get('department'),
                        'collection': collection_name
                    }
                }
                all_chunks.append(chunk)
                
        except Exception as e:
            print(f"    ❌ Error fetching from {collection_name}: {e}")
            continue
    
    print(f"✅ Total chunks fetched: {len(all_chunks)}")
    return all_chunks

def build_enhanced_bridge_table(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Build bridge table with enhanced patterns"""
    
    print("🔗 Building enhanced bridge table from chunks...")
    
    # Remove existing bridge table
    bridge_db_path = "./bridge_table_enhanced.db"
    if os.path.exists(bridge_db_path):
        os.remove(bridge_db_path)
        print(f"  🗑️  Removed existing {bridge_db_path}")
    
    # Initialize bridge table builder
    bridge_builder = BridgeTableBuilder(bridge_db_path)
    
    # Build from chunks
    stats = bridge_builder.build_from_chunks(chunks)
    
    print(f"✅ Enhanced bridge table built:")
    for key, value in stats.items():
        print(f"  📊 {key}: {value}")
    
    return stats

def export_enhanced_bridge_table_to_json(db_path: str) -> str:
    """Export enhanced bridge table to JSON format"""
    
    print("📤 Exporting enhanced bridge table to JSON...")
    
    import sqlite3
    
    json_path = f"{db_path}.json"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Export entities
        cursor.execute("SELECT * FROM entities")
        entities_rows = cursor.fetchall()
        
        cursor.execute("PRAGMA table_info(entities)")
        entities_columns = [column[1] for column in cursor.fetchall()]
        
        entities = {}
        for row in entities_rows:
            entity_data = dict(zip(entities_columns, row))
            entity_id = entity_data['entity_id']
            entities[entity_id] = {
                'id': entity_id,
                'type': entity_data['entity_type'],
                'value': entity_data['entity_value'],
                'source_doc_id': entity_data['source_doc_id'],
                'confidence': entity_data['confidence'],
                'metadata': json.loads(entity_data['metadata']) if entity_data['metadata'] else {}
            }
        
        # Export relationships  
        cursor.execute("SELECT * FROM relationships")
        relationships_rows = cursor.fetchall()
        
        cursor.execute("PRAGMA table_info(relationships)")
        relationships_columns = [column[1] for column in cursor.fetchall()]
        
        relationships = {}
        for row in relationships_rows:
            rel_data = dict(zip(relationships_columns, row))
            rel_id = rel_data['relationship_id']
            relationships[rel_id] = {
                'id': rel_id,
                'source': rel_data['source_entity_id'],
                'target': rel_data['target_entity_id'],
                'type': rel_data['relationship_type'],
                'confidence': rel_data['confidence'],
                'evidence': rel_data['evidence_chunk_id'],
                'metadata': json.loads(rel_data['metadata']) if rel_data['metadata'] else {}
            }
        
        conn.close()
        
        # Create JSON structure
        bridge_json = {
            'entities': entities,
            'relationships': relationships,
            'stats': {
                'total_entities': len(entities),
                'total_relationships': len(relationships),
                'entity_types': {},
                'relationship_types': {}
            }
        }
        
        # Calculate stats
        for entity in entities.values():
            entity_type = entity['type']
            bridge_json['stats']['entity_types'][entity_type] = bridge_json['stats']['entity_types'].get(entity_type, 0) + 1
        
        for rel in relationships.values():
            rel_type = rel['type']
            bridge_json['stats']['relationship_types'][rel_type] = bridge_json['stats']['relationship_types'].get(rel_type, 0) + 1
        
        # Write JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(bridge_json, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Enhanced bridge table exported to {json_path}")
        print(f"  📊 Entities: {len(entities)}")
        print(f"  📊 Relationships: {len(relationships)}")
        print(f"  📊 Entity types: {bridge_json['stats']['entity_types']}")
        print(f"  📊 Relationship types: {bridge_json['stats']['relationship_types']}")
        
        return json_path
        
    except Exception as e:
        print(f"❌ Error exporting enhanced bridge table: {e}")
        return ""

def test_enhanced_relationships(json_path: str):
    """Test the enhanced bridge table"""
    
    print("🧪 Testing enhanced relationship extraction...")
    
    try:
        from src.retrieval.bridge_lookup import BridgeTableLookup
        
        bridge_lookup = BridgeTableLookup(json_path)
        
        # Test specific entities
        test_cases = [
            ("go", "54"),
            ("go", "123"), 
            ("go", "456"),
            ("scheme", "Nadu-Nedu"),
            ("scheme", "Amma Vodi"),
            ("section", "12")
        ]
        
        for entity_type, entity_value in test_cases:
            print(f"  🔍 Testing: {entity_type} = {entity_value}")
            
            entity = bridge_lookup.lookup_entity(entity_type, entity_value)
            if entity:
                print(f"    ✅ Found: {entity.get('value')}")
                
                relationships = bridge_lookup.get_related_entities(entity.get('id'))
                print(f"    🔗 Relationships: {len(relationships)}")
                
                for rel in relationships[:3]:
                    print(f"      - {rel['relationship']}: {rel['entity']['value']} (conf: {rel['confidence']:.2f})")
            else:
                print(f"    ❌ Not found")
        
        print("✅ Enhanced relationship test complete")
        
    except Exception as e:
        print(f"❌ Error testing enhanced relationships: {e}")

def main():
    """Main enhanced rebuild workflow"""
    
    print("=" * 80)
    print("🚀 REBUILDING BRIDGE TABLE WITH ENHANCED PATTERNS")
    print("=" * 80)
    
    try:
        # Step 1: Apply enhanced patterns
        apply_enhanced_patterns()
        
        # Step 2: Fetch chunks
        chunks = fetch_all_chunks_from_qdrant()
        
        if not chunks:
            print("❌ No chunks found in Qdrant")
            return False
        
        # Step 3: Build enhanced bridge table
        stats = build_enhanced_bridge_table(chunks)
        
        # Step 4: Export to JSON
        json_path = export_enhanced_bridge_table_to_json("./bridge_table_enhanced.db")
        
        if not json_path:
            return False
        
        # Step 5: Test enhanced relationships
        test_enhanced_relationships(json_path)
        
        # Step 6: Copy to main bridge table location if successful
        if stats.get('relationships_created', 0) > 0:
            import shutil
            shutil.copy("./bridge_table_enhanced.db", "./bridge_table.db")
            shutil.copy("./bridge_table_enhanced.db.json", "./bridge_table.db.json")
            print("✅ Enhanced bridge table copied to main location")
        
        print("\n" + "=" * 80)
        print("✅ ENHANCED BRIDGE TABLE REBUILD COMPLETE!")
        print("=" * 80)
        print(f"📁 Enhanced Database: ./bridge_table_enhanced.db")
        print(f"📁 Enhanced JSON: ./bridge_table_enhanced.db.json")
        print(f"📊 Relationships extracted: {stats.get('relationships_created', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)