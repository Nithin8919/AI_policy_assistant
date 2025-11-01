"""
SOTA Bridge Table Implementation for Policy Document Relationships

This module builds and manages a comprehensive knowledge graph of relationships
between policy documents, enabling advanced query capabilities like:
- Supersession tracking (GO 123 supersedes GO 45)
- Legal cross-references (Section 12 read with Section 15)
- Temporal relationships (effective from date X)
- Implementation chains (Act → Rules → GOs)

Key Features:
1. Automated relationship extraction from document content
2. Graph-based relationship storage and traversal
3. Temporal validity tracking
4. Confidence scoring for relationships
5. Query expansion using relationship context
"""

import os
import sys
import re
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, date
import logging
from collections import defaultdict, deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RelationshipType(Enum):
    """Types of relationships between documents/entities"""
    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"
    AMENDS = "amends"
    AMENDED_BY = "amended_by"
    IMPLEMENTS = "implements"
    IMPLEMENTED_BY = "implemented_by"
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    CROSS_REFERENCES = "cross_references"
    DEFINES = "defines"
    DEFINED_BY = "defined_by"
    EXTENDS = "extends"
    EXTENDED_BY = "extended_by"
    PART_OF = "part_of"
    CONTAINS = "contains"
    TEMPORAL_SUCCESSOR = "temporal_successor"
    TEMPORAL_PREDECESSOR = "temporal_predecessor"


class EntityType(Enum):
    """Types of entities in the knowledge graph"""
    GOVERNMENT_ORDER = "government_order"
    LEGAL_SECTION = "legal_section"
    ACT = "act"
    RULE = "rule"
    SCHEME = "scheme"
    DEPARTMENT = "department"
    DISTRICT = "district"
    SCHOOL_TYPE = "school_type"
    QUALIFICATION = "qualification"
    DOCUMENT = "document"
    DATE = "date"
    AMOUNT = "amount"


@dataclass
class Entity:
    """Knowledge graph entity"""
    entity_id: str
    entity_type: EntityType
    entity_value: str
    source_doc_id: str
    first_seen_date: date
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Relationship:
    """Knowledge graph relationship"""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    confidence: float
    evidence_chunk_id: str
    valid_from: Optional[date] = None
    valid_until: Optional[date] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BridgeTableBuilder:
    """
    Builds and manages the bridge table (knowledge graph) for policy documents.
    
    The bridge table captures semantic relationships between documents,
    enabling advanced query capabilities and context-aware retrieval.
    """
    
    # Relationship extraction patterns
    RELATIONSHIP_PATTERNS = {
        RelationshipType.SUPERSEDES: [
            r'supersedes?\s+G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
            r'in\s+supersession\s+of\s+.*?G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
            r'replaces?\s+.*?G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
            r'cancels?\s+.*?G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)'
        ],
        
        RelationshipType.AMENDS: [
            r'amends?\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'amendment\s+to\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'modified?\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'substitut(?:es?|ed)\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)'
        ],
        
        RelationshipType.IMPLEMENTS: [
            r'(?:in\s+)?(?:pursuant|pursuance)\s+(?:to|of)\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'under\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'as\s+per\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'implementation\s+of\s+([A-Z][a-zA-Z\s]+(?:Act|Rule))',
            r'under\s+the\s+([A-Z][a-zA-Z\s]+(?:Act|Rule))'
        ],
        
        RelationshipType.REFERENCES: [
            r'(?:refer|see|vide)\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'as\s+defined\s+in\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'subject\s+to\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'read\s+with\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'conjunction\s+with\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)'
        ],
        
        RelationshipType.CROSS_REFERENCES: [
            r'(?:Section\s+)?(\d+(?:\([a-z]\))?)\s+read\s+with\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'(?:Section\s+)?(\d+(?:\([a-z]\))?)\s+and\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)',
            r'(?:Section\s+)?(\d+(?:\([a-z]\))?)\s+(?:along\s+with|together\s+with)\s+(?:Section\s+)?(\d+(?:\([a-z]\))?)'
        ]
    }
    
    # Entity extraction patterns
    ENTITY_PATTERNS = {
        EntityType.GOVERNMENT_ORDER: [
            r'G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
            r'Government\s+Order.*?No\.?\s*(\d+)'
        ],
        
        EntityType.LEGAL_SECTION: [
            r'Section\s+(\d+(?:\([a-z]\))?)',
            r'Article\s+(\d+)',
            r'Rule\s+(\d+)',
            r'Chapter\s+(\d+)'
        ],
        
        EntityType.ACT: [
            r'([A-Z][a-zA-Z\s]+Act\s+(?:of\s+)?\d{4})',
            r'(Right\s+to\s+Education\s+Act)',
            r'(RTE\s+Act)'
        ],
        
        EntityType.SCHEME: [
            r'(Nadu[\-\s]?Nedu)',
            r'(Amma\s+Vodi)',
            r'(Jagananna\s+[A-Za-z\s]+)',
            r'(Mid[\-\s]?Day[\-\s]?Meal)',
            r'(Sarva\s+Shiksha\s+Abhiyan)'
        ],
        
        EntityType.DISTRICT: [
            r'(?:in|of|for)\s+([A-Z][a-z]+)\s+district',
            r'([A-Z][a-z]+)\s+district'
        ],
        
        EntityType.DATE: [
            r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
            r'(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4})',
            r'(w\.?e\.?f\.?\s+\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})'
        ]
    }
    
    def __init__(self, db_path: str = "./bridge_table.db"):
        """
        Initialize bridge table builder.
        
        Args:
            db_path: Path to SQLite database for storing relationships
        """
        self.db_path = db_path
        self._init_database()
        
        logger.info(f"Bridge table builder initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with bridge table schema."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_value TEXT NOT NULL,
                    source_doc_id TEXT NOT NULL,
                    first_seen_date DATE NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Relationships table  
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    relationship_id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_chunk_id TEXT NOT NULL,
                    valid_from DATE,
                    valid_until DATE,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_entity_id) REFERENCES entities (entity_id),
                    FOREIGN KEY (target_entity_id) REFERENCES entities (entity_id)
                )
            """)
            
            # Temporal validity table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_temporal (
                    entity_id TEXT NOT NULL,
                    valid_from DATE,
                    valid_until DATE,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (entity_id) REFERENCES entities (entity_id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type_value ON entities(entity_type, entity_value)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relationship_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_validity ON entity_temporal(valid_from, valid_until)")
            
            conn.commit()
        
        logger.info("Bridge table database schema initialized")
    
    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Build bridge table from processed document chunks.
        
        Args:
            chunks: List of document chunks with metadata
            
        Returns:
            Statistics about entities and relationships created
        """
        logger.info(f"Building bridge table from {len(chunks)} chunks")
        
        entities_created = 0
        relationships_created = 0
        
        # Step 1: Extract entities from all chunks
        for chunk in chunks:
            chunk_entities = self._extract_entities_from_chunk(chunk)
            
            for entity in chunk_entities:
                if self._add_entity(entity):
                    entities_created += 1
        
        # Step 2: Extract relationships from all chunks
        for chunk in chunks:
            chunk_relationships = self._extract_relationships_from_chunk(chunk)
            
            for relationship in chunk_relationships:
                if self._add_relationship(relationship):
                    relationships_created += 1
        
        # Step 3: Build derived relationships
        derived_relationships = self._build_derived_relationships()
        relationships_created += derived_relationships
        
        # Step 4: Build supersession chains
        supersession_chains = self._build_supersession_chains()
        
        stats = {
            'entities_created': entities_created,
            'relationships_created': relationships_created,
            'derived_relationships': derived_relationships,
            'supersession_chains': supersession_chains
        }
        
        logger.info(f"Bridge table built: {stats}")
        
        return stats
    
    def _extract_entities_from_chunk(self, chunk: Dict[str, Any]) -> List[Entity]:
        """Extract entities from a single chunk."""
        entities = []
        content = chunk.get('content', '')
        doc_id = chunk.get('doc_id', '')
        chunk_id = chunk.get('chunk_id', '')
        
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    entity_value = match.group(1).strip()
                    
                    # Generate entity ID
                    entity_id = f"{entity_type.value}_{entity_value.replace(' ', '_').replace('.', '')}"
                    
                    # Calculate confidence based on context
                    confidence = self._calculate_entity_confidence(match, content, entity_type)
                    
                    entity = Entity(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        entity_value=entity_value,
                        source_doc_id=doc_id,
                        first_seen_date=date.today(),
                        confidence=confidence,
                        metadata={
                            'chunk_id': chunk_id,
                            'match_context': content[max(0, match.start()-50):match.end()+50],
                            'position': match.start()
                        }
                    )
                    
                    entities.append(entity)
        
        return entities
    
    def _extract_relationships_from_chunk(self, chunk: Dict[str, Any]) -> List[Relationship]:
        """Extract relationships from a single chunk."""
        relationships = []
        content = chunk.get('content', '')
        doc_id = chunk.get('doc_id', '')
        chunk_id = chunk.get('chunk_id', '')
        
        # Extract entities first to establish source/target
        chunk_entities = self._extract_entities_from_chunk(chunk)
        entity_map = {e.entity_value: e.entity_id for e in chunk_entities}
        
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
    
    def _determine_relationship_entities(
        self, 
        match: re.Match, 
        content: str, 
        rel_type: RelationshipType,
        entity_map: Dict[str, str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Determine source and target entities for a relationship."""
        
        if rel_type == RelationshipType.SUPERSEDES:
            # For supersession: current document supersedes mentioned GO
            target_go = match.group(1)
            target_entity = entity_map.get(f"G.O.No.{target_go}")
            
            # Source is the current document (infer from context)
            source_entity = self._infer_current_document_entity(content, entity_map)
            
            return source_entity, target_entity
        
        elif rel_type in [RelationshipType.REFERENCES, RelationshipType.IMPLEMENTS]:
            # For references: current section references target section
            target_section = match.group(1)
            target_entity = entity_map.get(f"Section {target_section}")
            
            source_entity = self._infer_current_section_entity(match, content, entity_map)
            
            return source_entity, target_entity
        
        elif rel_type == RelationshipType.CROSS_REFERENCES:
            # For cross-references: bidirectional relationship
            if len(match.groups()) >= 2:
                section1 = match.group(1)
                section2 = match.group(2)
                
                entity1 = entity_map.get(f"Section {section1}")
                entity2 = entity_map.get(f"Section {section2}")
                
                return entity1, entity2
        
        return None, None
    
    def _calculate_entity_confidence(
        self, 
        match: re.Match, 
        content: str, 
        entity_type: EntityType
    ) -> float:
        """Calculate confidence score for entity extraction."""
        confidence = 0.8  # Base confidence
        
        # Context-based confidence adjustments
        context = content[max(0, match.start()-50):match.end()+50].lower()
        
        # Higher confidence for formal document language
        if any(word in context for word in ['hereby', 'whereas', 'provided', 'shall']):
            confidence += 0.1
        
        # Higher confidence for structured references
        if entity_type == EntityType.LEGAL_SECTION and 'section' in context:
            confidence += 0.1
        
        # Lower confidence for indirect mentions
        if any(word in context for word in ['may', 'might', 'could', 'similar']):
            confidence -= 0.1
        
        return min(max(confidence, 0.1), 1.0)
    
    def _calculate_relationship_confidence(
        self, 
        match: re.Match, 
        content: str, 
        rel_type: RelationshipType
    ) -> float:
        """Calculate confidence score for relationship extraction."""
        confidence = 0.7  # Base confidence
        
        # Pattern specificity
        pattern_length = len(match.group(0))
        if pattern_length > 10:
            confidence += 0.1
        
        # Context strength
        context = content[max(0, match.start()-100):match.end()+100].lower()
        
        # Strong legal language
        if any(phrase in context for phrase in ['hereby', 'it is ordered', 'government orders']):
            confidence += 0.2
        
        # Formal structure
        if rel_type == RelationshipType.SUPERSEDES and 'supersession' in context:
            confidence += 0.2
        
        return min(max(confidence, 0.2), 1.0)
    
    def _infer_current_document_entity(self, content: str, entity_map: Dict[str, str]) -> Optional[str]:
        """Infer the entity representing the current document."""
        # Look for GO number in the beginning of content
        for pattern in self.ENTITY_PATTERNS[EntityType.GOVERNMENT_ORDER]:
            match = re.search(pattern, content[:500], re.IGNORECASE)
            if match:
                go_num = match.group(1)
                return entity_map.get(f"G.O.No.{go_num}")
        
        return None
    
    def _infer_current_section_entity(
        self, 
        match: re.Match, 
        content: str, 
        entity_map: Dict[str, str]
    ) -> Optional[str]:
        """Infer the current section making the reference."""
        # Look backwards for section header
        preceding_text = content[:match.start()]
        
        for pattern in self.ENTITY_PATTERNS[EntityType.LEGAL_SECTION]:
            matches = list(re.finditer(pattern, preceding_text, re.IGNORECASE))
            if matches:
                # Take the closest preceding section
                last_match = matches[-1]
                section_num = last_match.group(1)
                return entity_map.get(f"Section {section_num}")
        
        return None
    
    def _add_entity(self, entity: Entity) -> bool:
        """Add entity to database if it doesn't exist."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if entity exists
            cursor.execute(
                "SELECT entity_id FROM entities WHERE entity_id = ?",
                (entity.entity_id,)
            )
            
            if cursor.fetchone():
                return False  # Entity already exists
            
            # Insert new entity
            cursor.execute("""
                INSERT INTO entities 
                (entity_id, entity_type, entity_value, source_doc_id, first_seen_date, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.entity_id,
                entity.entity_type.value,
                entity.entity_value,
                entity.source_doc_id,
                entity.first_seen_date.isoformat(),
                entity.confidence,
                json.dumps(entity.metadata)
            ))
            
            conn.commit()
            return True
    
    def _add_relationship(self, relationship: Relationship) -> bool:
        """Add relationship to database if it doesn't exist."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if relationship exists
            cursor.execute(
                "SELECT relationship_id FROM relationships WHERE relationship_id = ?",
                (relationship.relationship_id,)
            )
            
            if cursor.fetchone():
                return False  # Relationship already exists
            
            # Verify source and target entities exist
            cursor.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_id IN (?, ?)",
                (relationship.source_entity_id, relationship.target_entity_id)
            )
            
            if cursor.fetchone()[0] != 2:
                logger.warning(f"Missing entities for relationship {relationship.relationship_id}")
                return False
            
            # Insert new relationship
            cursor.execute("""
                INSERT INTO relationships 
                (relationship_id, source_entity_id, target_entity_id, relationship_type, 
                 confidence, evidence_chunk_id, valid_from, valid_until, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship.relationship_id,
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.relationship_type.value,
                relationship.confidence,
                relationship.evidence_chunk_id,
                relationship.valid_from.isoformat() if relationship.valid_from else None,
                relationship.valid_until.isoformat() if relationship.valid_until else None,
                json.dumps(relationship.metadata)
            ))
            
            conn.commit()
            return True
    
    def _build_derived_relationships(self) -> int:
        """Build derived relationships (inverse, transitive)."""
        derived_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build inverse relationships
            inverse_mapping = {
                RelationshipType.SUPERSEDES.value: RelationshipType.SUPERSEDED_BY.value,
                RelationshipType.AMENDS.value: RelationshipType.AMENDED_BY.value,
                RelationshipType.IMPLEMENTS.value: RelationshipType.IMPLEMENTED_BY.value,
                RelationshipType.REFERENCES.value: RelationshipType.REFERENCED_BY.value,
            }
            
            for forward_type, inverse_type in inverse_mapping.items():
                cursor.execute("""
                    INSERT OR IGNORE INTO relationships 
                    (relationship_id, source_entity_id, target_entity_id, relationship_type, 
                     confidence, evidence_chunk_id, metadata)
                    SELECT 
                        ? || '_inv_' || relationship_id,
                        target_entity_id,
                        source_entity_id,
                        ?,
                        confidence * 0.9,
                        evidence_chunk_id,
                        metadata
                    FROM relationships 
                    WHERE relationship_type = ?
                """, (inverse_type, inverse_type, forward_type))
                
                derived_count += cursor.rowcount
            
            conn.commit()
        
        logger.info(f"Created {derived_count} derived relationships")
        
        return derived_count
    
    def _build_supersession_chains(self) -> int:
        """Build supersession chains (transitive closure)."""
        chains_built = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all supersession relationships
            cursor.execute("""
                SELECT source_entity_id, target_entity_id, confidence
                FROM relationships 
                WHERE relationship_type = ?
            """, (RelationshipType.SUPERSEDES.value,))
            
            supersessions = cursor.fetchall()
            
            # Build transitive closure using graph traversal
            supersession_graph = defaultdict(list)
            for source, target, confidence in supersessions:
                supersession_graph[source].append((target, confidence))
            
            # For each GO, find all GOs it transitively supersedes
            for start_go in supersession_graph:
                visited = set()
                queue = deque([(start_go, 1.0)])
                
                while queue:
                    current_go, path_confidence = queue.popleft()
                    
                    if current_go in visited:
                        continue
                    
                    visited.add(current_go)
                    
                    for next_go, edge_confidence in supersession_graph[current_go]:
                        if next_go not in visited:
                            # Create transitive supersession relationship
                            new_confidence = path_confidence * edge_confidence * 0.8  # Decay for transitivity
                            
                            if new_confidence > 0.3:  # Minimum confidence threshold
                                chain_id = f"chain_{RelationshipType.SUPERSEDES.value}_{start_go}_{next_go}"
                                
                                cursor.execute("""
                                    INSERT OR IGNORE INTO relationships 
                                    (relationship_id, source_entity_id, target_entity_id, relationship_type, 
                                     confidence, evidence_chunk_id, metadata)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    chain_id,
                                    start_go,
                                    next_go,
                                    RelationshipType.SUPERSEDES.value,
                                    new_confidence,
                                    "transitive_chain",
                                    json.dumps({"transitive": True, "chain_length": len(visited)})
                                ))
                                
                                if cursor.rowcount > 0:
                                    chains_built += 1
                            
                            queue.append((next_go, new_confidence))
            
            conn.commit()
        
        logger.info(f"Built {chains_built} supersession chains")
        
        return chains_built
    
    # ========== QUERY METHODS ==========
    
    def get_entity_relationships(
        self, 
        entity_id: str, 
        relationship_types: Optional[List[RelationshipType]] = None,
        include_confidence: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            where_clause = "WHERE (source_entity_id = ? OR target_entity_id = ?)"
            params = [entity_id, entity_id]
            
            if relationship_types:
                type_placeholders = ",".join("?" * len(relationship_types))
                where_clause += f" AND relationship_type IN ({type_placeholders})"
                params.extend([rt.value for rt in relationship_types])
            
            query = f"""
                SELECT r.*, 
                       se.entity_value as source_value,
                       te.entity_value as target_value
                FROM relationships r
                JOIN entities se ON r.source_entity_id = se.entity_id
                JOIN entities te ON r.target_entity_id = te.entity_id
                {where_clause}
                ORDER BY r.confidence DESC
            """
            
            cursor.execute(query, params)
            
            relationships = []
            for row in cursor.fetchall():
                rel_data = {
                    'relationship_id': row[0],
                    'source_entity_id': row[1],
                    'target_entity_id': row[2],
                    'relationship_type': row[3],
                    'confidence': row[4] if include_confidence else None,
                    'evidence_chunk_id': row[5],
                    'source_value': row[10],
                    'target_value': row[11],
                    'metadata': json.loads(row[9]) if row[9] else {}
                }
                relationships.append(rel_data)
            
            return relationships
    
    def get_supersession_chain(self, go_number: str) -> Dict[str, List[str]]:
        """Get complete supersession chain for a GO."""
        
        entity_id = f"government_order_GO_No_{go_number}"
        
        # Get what this GO supersedes
        supersedes = self.get_entity_relationships(
            entity_id, 
            [RelationshipType.SUPERSEDES]
        )
        
        # Get what supersedes this GO
        superseded_by = self.get_entity_relationships(
            entity_id,
            [RelationshipType.SUPERSEDED_BY]
        )
        
        return {
            'supersedes': [rel['target_value'] for rel in supersedes],
            'superseded_by': [rel['source_value'] for rel in superseded_by]
        }
    
    def expand_query_with_relationships(
        self, 
        query_entities: List[str],
        relationship_types: Optional[List[RelationshipType]] = None,
        max_expansion: int = 5
    ) -> Dict[str, Any]:
        """Expand query entities using relationship context."""
        
        expanded_entities = set(query_entities)
        relationship_context = []
        
        for entity in query_entities:
            relationships = self.get_entity_relationships(entity, relationship_types)
            
            # Add high-confidence related entities
            for rel in relationships[:max_expansion]:
                if rel['confidence'] > 0.7:
                    if rel['source_entity_id'] == entity:
                        expanded_entities.add(rel['target_entity_id'])
                    else:
                        expanded_entities.add(rel['source_entity_id'])
                    
                    relationship_context.append({
                        'type': rel['relationship_type'],
                        'description': f"{rel['source_value']} {rel['relationship_type']} {rel['target_value']}",
                        'confidence': rel['confidence']
                    })
        
        return {
            'original_entities': query_entities,
            'expanded_entities': list(expanded_entities),
            'relationship_context': relationship_context,
            'expansion_count': len(expanded_entities) - len(query_entities)
        }
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get statistics about the bridge table."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Entity counts by type
            cursor.execute("""
                SELECT entity_type, COUNT(*) 
                FROM entities 
                GROUP BY entity_type
            """)
            entity_stats = dict(cursor.fetchall())
            
            # Relationship counts by type
            cursor.execute("""
                SELECT relationship_type, COUNT(*) 
                FROM relationships 
                GROUP BY relationship_type
            """)
            relationship_stats = dict(cursor.fetchall())
            
            # Total counts
            cursor.execute("SELECT COUNT(*) FROM entities")
            total_entities = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM relationships")
            total_relationships = cursor.fetchone()[0]
            
            return {
                'total_entities': total_entities,
                'total_relationships': total_relationships,
                'entities_by_type': entity_stats,
                'relationships_by_type': relationship_stats
            }


# Convenience functions

def build_bridge_table_from_chunks(
    chunks: List[Dict[str, Any]],
    db_path: str = "./bridge_table.db"
) -> Dict[str, int]:
    """Convenience function to build bridge table from chunks."""
    
    builder = BridgeTableBuilder(db_path)
    return builder.build_from_chunks(chunks)


if __name__ == "__main__":
    print("Bridge Table Builder module loaded successfully")
    
    # Test with sample data
    sample_chunks = [
        {
            'chunk_id': 'test_go_123',
            'doc_id': 'GO_123_2024',
            'content': 'G.O.Ms.No.123 supersedes G.O.Ms.No.45 dated 2023. This order implements Section 12(1)(c) of the RTE Act.',
            'metadata': {'doc_type': 'government_order'}
        },
        {
            'chunk_id': 'test_rte_sec12',
            'doc_id': 'RTE_Act_2009',
            'content': 'Section 12(1)(c) shall be read with Section 15 for implementation purposes.',
            'metadata': {'doc_type': 'legal_documents'}
        }
    ]
    
    builder = BridgeTableBuilder("./test_bridge.db")
    stats = builder.build_from_chunks(sample_chunks)
    
    print(f"✅ Test bridge table built: {stats}")
    
    # Test query expansion
    expansion = builder.expand_query_with_relationships(['government_order_GO_No_123'])
    print(f"✅ Query expansion test: {expansion}")
    
    # Cleanup test database
    import os
    if os.path.exists("./test_bridge.db"):
        os.remove("./test_bridge.db")