"""
Bridge Table Lookup for Relationship-Aware Retrieval
Enables queries like "What superseded GO 42?" to find relationships
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BridgeTableLookup:
    """Fast relationship lookup via bridge table"""
    
    def __init__(self, bridge_file_path: Optional[str] = None):
        """
        Initialize bridge table lookup
        
        Args:
            bridge_file_path: Path to bridge table JSON file
        """
        self.bridge_file_path = bridge_file_path or "data/knowledge_graph/bridge_table.json"
        self.bridge_data = {}
        self.entity_index = {}
        self.relationship_index = {}
        
        self._load_bridge_table()
    
    def _load_bridge_table(self):
        """Load bridge table from file"""
        bridge_path = Path(self.bridge_file_path)
        
        if not bridge_path.exists():
            logger.warning(f"Bridge table not found at {self.bridge_file_path}")
            return
        
        try:
            with open(bridge_path, 'r', encoding='utf-8') as f:
                self.bridge_data = json.load(f)
            
            # Build indices for fast lookup
            self._build_indices()
            
            logger.info(f"Loaded bridge table with {len(self.entity_index)} entities")
        except Exception as e:
            logger.error(f"Failed to load bridge table: {e}")
    
    def _build_indices(self):
        """Build indices for fast entity and relationship lookup"""
        # Index entities by type and value
        for entity_id, entity_data in self.bridge_data.get('entities', {}).items():
            entity_type = entity_data.get('type')
            entity_value = entity_data.get('value', '').lower()
            
            # Build entity index: {type: {value: entity_id}}
            if entity_type not in self.entity_index:
                self.entity_index[entity_type] = {}
            
            self.entity_index[entity_type][entity_value] = entity_id
        
        # Index relationships by type
        for rel_id, rel_data in self.bridge_data.get('relationships', {}).items():
            rel_type = rel_data.get('type')
            source_id = rel_data.get('source')
            target_id = rel_data.get('target')
            
            # Build relationship index: {rel_type: [(source, target, data)]}
            if rel_type not in self.relationship_index:
                self.relationship_index[rel_type] = []
            
            self.relationship_index[rel_type].append({
                'source': source_id,
                'target': target_id,
                'data': rel_data
            })
    
    def lookup_entity(self, entity_type: str, entity_value: str) -> Optional[Dict]:
        """
        Lookup an entity by type and value
        
        Args:
            entity_type: Type of entity (e.g., 'go', 'section', 'scheme')
            entity_value: Value of entity (e.g., 'GO.Ms.No.54', 'Section 12(1)(c)')
        
        Returns:
            Entity data or None
        """
        entity_value_lower = entity_value.lower()
        
        if entity_type in self.entity_index:
            entity_id = self.entity_index[entity_type].get(entity_value_lower)
            if entity_id:
                return self.bridge_data.get('entities', {}).get(entity_id)
        
        return None
    
    def get_related_entities(
        self, 
        entity_id: str, 
        relationship_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all entities related to a given entity
        
        Args:
            entity_id: ID of the source entity
            relationship_type: Optional filter by relationship type
        
        Returns:
            List of related entities with relationship info
        """
        related = []
        
        # Search all relationships if type not specified
        rel_types = [relationship_type] if relationship_type else self.relationship_index.keys()
        
        for rel_type in rel_types:
            if rel_type not in self.relationship_index:
                continue
            
            for rel in self.relationship_index[rel_type]:
                if rel['source'] == entity_id:
                    target_entity = self.bridge_data.get('entities', {}).get(rel['target'])
                    if target_entity:
                        related.append({
                            'entity': target_entity,
                            'relationship': rel_type,
                            'confidence': rel['data'].get('confidence', 1.0),
                            'evidence': rel['data'].get('evidence')
                        })
        
        return related
    
    def get_supersession_chain(self, go_number: str) -> List[Dict]:
        """
        Get complete supersession chain for a GO
        
        Args:
            go_number: GO number (e.g., 'GO.Ms.No.54')
        
        Returns:
            List of GOs in supersession chain, ordered from oldest to newest
        """
        # Find the GO entity
        go_entity = self.lookup_entity('go', go_number)
        if not go_entity:
            return []
        
        entity_id = go_entity.get('id')
        chain = [go_entity]
        
        # Follow "supersedes" relationships backwards
        visited = {entity_id}
        current_id = entity_id
        
        # Find what this GO supersedes (older GOs)
        while True:
            superseded = self.get_related_entities(current_id, 'supersedes')
            if not superseded:
                break
            
            superseded_entity = superseded[0]['entity']
            superseded_id = superseded_entity.get('id')
            
            if superseded_id in visited:
                break  # Circular reference
            
            chain.insert(0, superseded_entity)  # Add to beginning
            visited.add(superseded_id)
            current_id = superseded_id
        
        # Find what supersedes this GO (newer GOs)
        current_id = entity_id
        while True:
            supersedes_this = []
            for rel_type in ['supersedes']:
                if rel_type not in self.relationship_index:
                    continue
                
                for rel in self.relationship_index[rel_type]:
                    if rel['target'] == current_id:
                        superseding_entity = self.bridge_data.get('entities', {}).get(rel['source'])
                        if superseding_entity:
                            supersedes_this.append(superseding_entity)
            
            if not supersedes_this:
                break
            
            superseding_entity = supersedes_this[0]
            superseding_id = superseding_entity.get('id')
            
            if superseding_id in visited:
                break
            
            chain.append(superseding_entity)
            visited.add(superseding_id)
            current_id = superseding_id
        
        return chain
    
    def get_current_go(self, go_number: str) -> Optional[Dict]:
        """
        Get the current (most recent) GO in a supersession chain
        
        Args:
            go_number: Any GO in the chain
        
        Returns:
            Current GO entity or None
        """
        chain = self.get_supersession_chain(go_number)
        if not chain:
            return None
        
        # Last in chain is the current one
        return chain[-1]
    
    def get_implementing_documents(self, scheme_name: str) -> List[Dict]:
        """
        Get all GOs that implement a scheme
        
        Args:
            scheme_name: Name of the scheme
        
        Returns:
            List of implementing documents
        """
        scheme_entity = self.lookup_entity('scheme', scheme_name)
        if not scheme_entity:
            return []
        
        return self.get_related_entities(scheme_entity.get('id'), 'implements')
    
    def enhance_query_with_context(self, query: str, entities: Dict[str, List]) -> Dict[str, Any]:
        """
        Enhance query with contextual information from bridge table
        
        Args:
            query: Original query
            entities: Extracted entities from query
        
        Returns:
            Enhancement context with related entities and relationships
        """
        context = {
            'related_entities': [],
            'warnings': [],
            'expansions': []
        }
        
        # Check for GO references
        if 'go_refs' in entities:
            for go_ref in entities['go_refs']:
                # Check if superseded
                current_go = self.get_current_go(go_ref)
                if current_go and current_go.get('value', '').lower() != go_ref.lower():
                    context['warnings'].append(
                        f"{go_ref} was superseded by {current_go['value']}"
                    )
                    context['expansions'].append(current_go['value'])
                
                # Get supersession chain
                chain = self.get_supersession_chain(go_ref)
                if len(chain) > 1:
                    context['related_entities'].extend([
                        {'type': 'go', 'value': go['value'], 'relation': 'supersession_chain'}
                        for go in chain
                    ])
        
        # Check for scheme references
        if 'schemes' in entities:
            for scheme in entities['schemes']:
                implementing_docs = self.get_implementing_documents(scheme)
                for doc in implementing_docs:
                    context['expansions'].append(doc['entity']['value'])
                    context['related_entities'].append({
                        'type': 'go',
                        'value': doc['entity']['value'],
                        'relation': 'implements',
                        'scheme': scheme
                    })
        
        return context


# Convenience function
def create_bridge_lookup(bridge_file_path: Optional[str] = None) -> BridgeTableLookup:
    """Create a BridgeTableLookup instance"""
    return BridgeTableLookup(bridge_file_path)
