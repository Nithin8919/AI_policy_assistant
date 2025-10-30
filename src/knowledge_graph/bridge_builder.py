"""Bridge table builder for creating topic-centered knowledge graph connections"""
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from src.knowledge_graph.relation_extractor import ChunkAnalysis, ExtractedEntity, ExtractedRelation
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class BridgeTopicEntry:
    """Represents a bridge topic entry"""
    topic_id: str
    topic_name: str
    keywords: List[str]
    legal_doc_ids: List[str]  # Acts/Rules that define it
    go_doc_ids: List[str]     # GOs that implement it
    judicial_doc_ids: List[str]  # Cases that interpret it
    data_doc_ids: List[str]   # Data reports with metrics
    external_doc_ids: List[str]  # External sources
    metric_codes: List[str]
    scheme_names: List[str]
    districts: List[str]  # Empty = applies to all
    related_topics: List[str]  # Related bridge topic IDs
    supersession_chain: List[Dict[str, Any]]  # For GOs
    confidence_score: float
    last_updated: str
    curator: str  # "system" or "manual"

class BridgeTableBuilder:
    """Build and maintain the bridge table for intelligent cross-referencing"""
    
    def __init__(self):
        """Initialize the bridge table builder"""
        self.bridge_table = {
            "topics": {},
            "relations": [],
            "metadata": {
                "total_topics": 0,
                "total_relations": 0,
                "last_updated": datetime.now().isoformat()
            }
        }
        
        # Document type mapping for routing
        self.doc_type_mapping = {
            "acts": "legal_doc_ids",
            "rules": "legal_doc_ids", 
            "government_orders": "go_doc_ids",
            "go": "go_doc_ids",
            "judicial_documents": "judicial_doc_ids",
            "judicial": "judicial_doc_ids",
            "data_reports": "data_doc_ids",
            "budget_finance": "data_doc_ids",
            "external_sources": "external_doc_ids",
            "frameworks": "external_doc_ids"
        }
    
    def load_seed_topics(self, seed_file: str) -> bool:
        """Load seed bridge topics from JSON file"""
        try:
            with open(seed_file, 'r', encoding='utf-8') as f:
                seed_data = json.load(f)
            
            for topic_id, topic_data in seed_data.get("topics", {}).items():
                # Create BridgeTopicEntry from seed data
                bridge_entry = BridgeTopicEntry(
                    topic_id=topic_id,
                    topic_name=topic_data.get("topic_name", ""),
                    keywords=topic_data.get("keywords", []),
                    legal_doc_ids=[],
                    go_doc_ids=[],
                    judicial_doc_ids=[],
                    data_doc_ids=[],
                    external_doc_ids=[],
                    metric_codes=topic_data.get("metric_codes", []),
                    scheme_names=topic_data.get("scheme_names", []),
                    districts=topic_data.get("districts", []),
                    related_topics=topic_data.get("related_topics", []),
                    supersession_chain=[],
                    confidence_score=1.0,  # Manual seeds have high confidence
                    last_updated=datetime.now().isoformat(),
                    curator="manual"
                )
                
                self.bridge_table["topics"][topic_id] = asdict(bridge_entry)
            
            logger.info(f"Loaded {len(self.bridge_table['topics'])} seed topics from {seed_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load seed topics: {e}")
            return False
    
    def load_existing_bridge_table(self, bridge_file: str) -> bool:
        """Load existing bridge table"""
        try:
            with open(bridge_file, 'r', encoding='utf-8') as f:
                self.bridge_table = json.load(f)
            
            logger.info(f"Loaded existing bridge table with {len(self.bridge_table['topics'])} topics")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load existing bridge table: {e}")
            return False
    
    def update_bridge_from_chunk_analysis(self, analysis: ChunkAnalysis) -> List[str]:
        """
        Update bridge table based on chunk analysis results
        
        Args:
            analysis: ChunkAnalysis results from relation extractor
            
        Returns:
            List of bridge topic IDs that were updated
        """
        updated_topics = []
        
        # Get document metadata
        doc_id = analysis.doc_id
        doc_type = self._infer_doc_type_from_doc_id(doc_id)
        
        # Process bridge topic matches
        for topic_id in analysis.bridge_topic_matches:
            if topic_id in self.bridge_table["topics"]:
                topic_data = self.bridge_table["topics"][topic_id]
                
                # Route document to appropriate category
                doc_category = self.doc_type_mapping.get(doc_type, "external_doc_ids")
                
                # Add document ID if not already present
                if doc_id not in topic_data[doc_category]:
                    topic_data[doc_category].append(doc_id)
                    topic_data["last_updated"] = datetime.now().isoformat()
                    updated_topics.append(topic_id)
                    
                    logger.debug(f"Added {doc_id} to bridge topic {topic_id} ({doc_category})")
        
        # Extract new entities for potential topic expansion
        self._extract_entities_for_topics(analysis)
        
        # Extract relations
        self._extract_relations_for_bridge(analysis)
        
        return updated_topics
    
    def suggest_new_bridge_topics(self, chunk_analyses: List[ChunkAnalysis], min_frequency: int = 3) -> List[Dict[str, Any]]:
        """
        Suggest new bridge topics based on frequent entity co-occurrences
        
        Args:
            chunk_analyses: List of chunk analyses
            min_frequency: Minimum frequency for suggesting new topics
            
        Returns:
            List of suggested topic dictionaries
        """
        suggestions = []
        
        # Track entity co-occurrences
        entity_cooccurrence = {}
        legal_entity_groups = []
        scheme_entity_groups = []
        
        for analysis in chunk_analyses:
            # Group entities by type for this chunk
            legal_refs = [e.entity_value for e in analysis.entities if e.entity_type == "legal_ref"]
            schemes = [e.entity_value for e in analysis.entities if e.entity_type == "scheme"]
            metrics = [e.entity_value for e in analysis.entities if e.entity_type == "metric"]
            go_refs = [e.entity_value for e in analysis.entities if e.entity_type == "go_ref"]
            
            # Track co-occurrences of legal refs + schemes (potential new topics)
            if legal_refs and schemes:
                for legal_ref in legal_refs:
                    for scheme in schemes:
                        key = f"{legal_ref}_{scheme}"
                        entity_cooccurrence[key] = entity_cooccurrence.get(key, 0) + 1
            
            # Track co-occurrences of legal refs + metrics
            if legal_refs and metrics:
                for legal_ref in legal_refs:
                    for metric in metrics:
                        key = f"{legal_ref}_{metric}"
                        entity_cooccurrence[key] = entity_cooccurrence.get(key, 0) + 1
        
        # Generate suggestions based on frequent co-occurrences
        for cooccurrence, frequency in entity_cooccurrence.items():
            if frequency >= min_frequency:
                parts = cooccurrence.split("_", 1)
                if len(parts) == 2:
                    entity1, entity2 = parts
                    
                    # Generate topic suggestion
                    topic_id = f"suggested_{entity1.lower().replace(' ', '_')}_{entity2.lower().replace(' ', '_')}"
                    topic_name = f"{entity1} - {entity2}"
                    
                    suggestion = {
                        "topic_id": topic_id,
                        "topic_name": topic_name,
                        "keywords": [entity1.lower(), entity2.lower()],
                        "frequency": frequency,
                        "suggestion_reason": f"Frequent co-occurrence ({frequency} times)",
                        "confidence": min(frequency / 10.0, 0.9)  # Scale confidence
                    }
                    
                    suggestions.append(suggestion)
        
        # Sort by frequency
        suggestions.sort(key=lambda x: x["frequency"], reverse=True)
        
        logger.info(f"Generated {len(suggestions)} bridge topic suggestions")
        return suggestions[:20]  # Return top 20 suggestions
    
    def add_manual_bridge_topic(self, topic_data: Dict[str, Any]) -> bool:
        """
        Add a manually curated bridge topic
        
        Args:
            topic_data: Dictionary with topic information
            
        Returns:
            True if successful
        """
        try:
            topic_id = topic_data["topic_id"]
            
            bridge_entry = BridgeTopicEntry(
                topic_id=topic_id,
                topic_name=topic_data.get("topic_name", ""),
                keywords=topic_data.get("keywords", []),
                legal_doc_ids=topic_data.get("legal_doc_ids", []),
                go_doc_ids=topic_data.get("go_doc_ids", []),
                judicial_doc_ids=topic_data.get("judicial_doc_ids", []),
                data_doc_ids=topic_data.get("data_doc_ids", []),
                external_doc_ids=topic_data.get("external_doc_ids", []),
                metric_codes=topic_data.get("metric_codes", []),
                scheme_names=topic_data.get("scheme_names", []),
                districts=topic_data.get("districts", []),
                related_topics=topic_data.get("related_topics", []),
                supersession_chain=topic_data.get("supersession_chain", []),
                confidence_score=topic_data.get("confidence_score", 0.9),
                last_updated=datetime.now().isoformat(),
                curator="manual"
            )
            
            self.bridge_table["topics"][topic_id] = asdict(bridge_entry)
            self._update_metadata()
            
            logger.info(f"Added manual bridge topic: {topic_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add manual bridge topic: {e}")
            return False
    
    def update_supersession_chain(self, go_doc_id: str, supersession_info: List[Dict[str, Any]]):
        """Update supersession chain for a GO document"""
        # Find bridge topics that contain this GO
        for topic_id, topic_data in self.bridge_table["topics"].items():
            if go_doc_id in topic_data["go_doc_ids"]:
                # Update supersession chain
                topic_data["supersession_chain"] = supersession_info
                topic_data["last_updated"] = datetime.now().isoformat()
                
                logger.debug(f"Updated supersession chain for {go_doc_id} in topic {topic_id}")
    
    def get_bridge_topics_for_matching(self) -> Dict[str, Any]:
        """Get bridge topics in format suitable for matching"""
        return self.bridge_table["topics"]
    
    def get_related_documents(self, topic_id: str) -> Dict[str, List[str]]:
        """Get all documents related to a bridge topic"""
        if topic_id not in self.bridge_table["topics"]:
            return {}
        
        topic_data = self.bridge_table["topics"][topic_id]
        
        return {
            "legal_documents": topic_data["legal_doc_ids"],
            "government_orders": topic_data["go_doc_ids"],
            "judicial_documents": topic_data["judicial_doc_ids"],
            "data_reports": topic_data["data_doc_ids"],
            "external_sources": topic_data["external_doc_ids"]
        }
    
    def find_topics_by_keyword(self, keyword: str) -> List[str]:
        """Find bridge topics that contain a specific keyword"""
        matching_topics = []
        keyword_lower = keyword.lower()
        
        for topic_id, topic_data in self.bridge_table["topics"].items():
            # Check keywords
            keywords = [k.lower() for k in topic_data["keywords"]]
            if keyword_lower in keywords or any(keyword_lower in k for k in keywords):
                matching_topics.append(topic_id)
                continue
            
            # Check topic name
            if keyword_lower in topic_data["topic_name"].lower():
                matching_topics.append(topic_id)
        
        return matching_topics
    
    def find_topics_by_entity(self, entity_value: str, entity_type: str) -> List[str]:
        """Find bridge topics related to a specific entity"""
        matching_topics = []
        entity_lower = entity_value.lower()
        
        for topic_id, topic_data in self.bridge_table["topics"].items():
            found = False
            
            # Check schemes
            if entity_type == "scheme":
                if entity_lower in [s.lower() for s in topic_data["scheme_names"]]:
                    found = True
            
            # Check metrics
            elif entity_type == "metric":
                if entity_lower in [m.lower() for m in topic_data["metric_codes"]]:
                    found = True
            
            # Check districts
            elif entity_type == "district":
                if entity_lower in [d.lower() for d in topic_data["districts"]]:
                    found = True
            
            # Check keywords as fallback
            if not found:
                if entity_lower in [k.lower() for k in topic_data["keywords"]]:
                    found = True
            
            if found:
                matching_topics.append(topic_id)
        
        return matching_topics
    
    def get_bridge_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the bridge table"""
        total_topics = len(self.bridge_table["topics"])
        total_relations = len(self.bridge_table["relations"])
        
        # Document distribution
        doc_distribution = {
            "legal_documents": 0,
            "government_orders": 0,
            "judicial_documents": 0,
            "data_reports": 0,
            "external_sources": 0
        }
        
        # Topic coverage stats
        topics_with_legal = 0
        topics_with_go = 0
        topics_with_judicial = 0
        topics_with_data = 0
        topics_with_external = 0
        
        for topic_data in self.bridge_table["topics"].values():
            doc_distribution["legal_documents"] += len(topic_data["legal_doc_ids"])
            doc_distribution["government_orders"] += len(topic_data["go_doc_ids"])
            doc_distribution["judicial_documents"] += len(topic_data["judicial_doc_ids"])
            doc_distribution["data_reports"] += len(topic_data["data_doc_ids"])
            doc_distribution["external_sources"] += len(topic_data["external_doc_ids"])
            
            if topic_data["legal_doc_ids"]: topics_with_legal += 1
            if topic_data["go_doc_ids"]: topics_with_go += 1
            if topic_data["judicial_doc_ids"]: topics_with_judicial += 1
            if topic_data["data_doc_ids"]: topics_with_data += 1
            if topic_data["external_doc_ids"]: topics_with_external += 1
        
        return {
            "total_topics": total_topics,
            "total_relations": total_relations,
            "document_distribution": doc_distribution,
            "topic_coverage": {
                "topics_with_legal": topics_with_legal,
                "topics_with_go": topics_with_go,
                "topics_with_judicial": topics_with_judicial,
                "topics_with_data": topics_with_data,
                "topics_with_external": topics_with_external
            },
            "coverage_percentages": {
                "legal_coverage": topics_with_legal / total_topics * 100 if total_topics > 0 else 0,
                "go_coverage": topics_with_go / total_topics * 100 if total_topics > 0 else 0,
                "judicial_coverage": topics_with_judicial / total_topics * 100 if total_topics > 0 else 0,
                "data_coverage": topics_with_data / total_topics * 100 if total_topics > 0 else 0,
                "external_coverage": topics_with_external / total_topics * 100 if total_topics > 0 else 0
            }
        }
    
    def save_bridge_table(self, output_file: str) -> bool:
        """Save bridge table to JSON file"""
        try:
            self._update_metadata()
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.bridge_table, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved bridge table to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save bridge table: {e}")
            return False
    
    def _infer_doc_type_from_doc_id(self, doc_id: str) -> str:
        """Infer document type from document ID"""
        doc_id_lower = doc_id.lower()
        
        if "act" in doc_id_lower or "rule" in doc_id_lower:
            return "acts"
        elif "go" in doc_id_lower or "order" in doc_id_lower:
            return "government_orders"
        elif "case" in doc_id_lower or "court" in doc_id_lower or "judgment" in doc_id_lower:
            return "judicial_documents"
        elif "budget" in doc_id_lower or "data" in doc_id_lower or "report" in doc_id_lower:
            return "data_reports"
        else:
            return "external_sources"
    
    def _extract_entities_for_topics(self, analysis: ChunkAnalysis):
        """Extract entities that could be used for topic expansion"""
        # This could be enhanced to suggest new topics based on entity patterns
        pass
    
    def _extract_relations_for_bridge(self, analysis: ChunkAnalysis):
        """Extract relations and add to bridge table"""
        for relation in analysis.relations:
            # Convert to bridge table relation format
            bridge_relation = {
                "source_doc_id": relation.source_doc_id,
                "target_entity": relation.target_entity,
                "relation_type": relation.relation_type,
                "confidence": relation.confidence,
                "evidence": relation.evidence,
                "extraction_method": relation.extraction_method,
                "extracted_at": datetime.now().isoformat()
            }
            
            # Avoid duplicates
            existing_relations = self.bridge_table["relations"]
            if not any(
                r["source_doc_id"] == bridge_relation["source_doc_id"] and
                r["target_entity"] == bridge_relation["target_entity"] and
                r["relation_type"] == bridge_relation["relation_type"]
                for r in existing_relations
            ):
                self.bridge_table["relations"].append(bridge_relation)
    
    def _update_metadata(self):
        """Update bridge table metadata"""
        self.bridge_table["metadata"] = {
            "total_topics": len(self.bridge_table["topics"]),
            "total_relations": len(self.bridge_table["relations"]),
            "last_updated": datetime.now().isoformat()
        }


