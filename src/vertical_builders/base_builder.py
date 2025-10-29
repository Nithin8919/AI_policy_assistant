"""
Base Vertical Database Builder.

Provides common functionality for all vertical database builders including:
- Chunk loading and filtering
- Entity aggregation
- Relation processing
- Output structure management
- Quality validation
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from collections import defaultdict, Counter

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseVerticalBuilder(ABC):
    """
    Abstract base class for vertical database builders.
    
    Provides common functionality for loading processed data and
    building specialized vertical databases.
    """
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize base vertical builder.
        
        Args:
            data_dir: Path to processed data directory
            output_dir: Path to output directory for vertical databases
        """
        if data_dir is None:
            data_dir = project_root / "data" / "processed"
        if output_dir is None:
            output_dir = project_root / "data" / "verticals"
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Statistics tracking
        self.stats = {
            "chunks_processed": 0,
            "documents_processed": 0,
            "entities_aggregated": 0,
            "relations_processed": 0,
            "database_entries_created": 0,
            "processing_errors": []
        }
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def load_processed_chunks(self, chunk_file: str = "all_chunks.jsonl") -> List[Dict]:
        """
        Load processed chunks from ingestion pipeline.
        
        Args:
            chunk_file: Name of chunks file
            
        Returns:
            List of chunk dictionaries
        """
        chunks_file = self.data_dir / "chunks" / chunk_file
        
        if not chunks_file.exists():
            logger.error(f"Chunks file not found: {chunks_file}")
            return []
        
        chunks = []
        try:
            with open(chunks_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            chunk = json.loads(line)
                            chunks.append(chunk)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num}: {e}")
            
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
            self.stats["chunks_processed"] = len(chunks)
            
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return []
        
        return chunks
    
    def load_relations(self, relations_dir: str = "relations") -> List[Dict]:
        """
        Load extracted relations from all documents.
        
        Args:
            relations_dir: Directory containing relation files
            
        Returns:
            List of all relations
        """
        relations_path = self.data_dir / relations_dir
        
        if not relations_path.exists():
            logger.warning(f"Relations directory not found: {relations_path}")
            return []
        
        all_relations = []
        relation_files = list(relations_path.glob("*_relations.json"))
        
        for rel_file in relation_files:
            try:
                with open(rel_file, 'r') as f:
                    relations = json.load(f)
                    if isinstance(relations, list):
                        all_relations.extend(relations)
                    else:
                        logger.warning(f"Unexpected relations format in {rel_file}")
            except Exception as e:
                logger.error(f"Error loading relations from {rel_file}: {e}")
        
        logger.info(f"Loaded {len(all_relations)} relations from {len(relation_files)} files")
        self.stats["relations_processed"] = len(all_relations)
        
        return all_relations
    
    def filter_chunks_by_doc_type(self, chunks: List[Dict], doc_types: List[str]) -> List[Dict]:
        """
        Filter chunks by document type.
        
        Args:
            chunks: List of chunks
            doc_types: List of document types to include
            
        Returns:
            Filtered chunks
        """
        filtered = []
        for chunk in chunks:
            doc_type = chunk.get("metadata", {}).get("doc_type")
            if doc_type in doc_types:
                filtered.append(chunk)
        
        logger.debug(f"Filtered {len(filtered)} chunks from {len(chunks)} for types: {doc_types}")
        return filtered
    
    def aggregate_entities_by_doc(self, chunks: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate entities by document ID.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary mapping doc_id to aggregated entities
        """
        doc_entities = defaultdict(lambda: defaultdict(list))
        
        for chunk in chunks:
            doc_id = chunk.get("doc_id")
            entities = chunk.get("entities", {})
            
            if doc_id and entities:
                for entity_type, entity_list in entities.items():
                    if isinstance(entity_list, list):
                        doc_entities[doc_id][entity_type].extend(entity_list)
        
        # Deduplicate entities within each document
        for doc_id in doc_entities:
            for entity_type in doc_entities[doc_id]:
                doc_entities[doc_id][entity_type] = list(dict.fromkeys(doc_entities[doc_id][entity_type]))
        
        logger.debug(f"Aggregated entities for {len(doc_entities)} documents")
        self.stats["entities_aggregated"] = sum(
            sum(len(entities) for entities in doc_entities[doc_id].values())
            for doc_id in doc_entities
        )
        
        return dict(doc_entities)
    
    def group_chunks_by_document(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group chunks by document ID.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary mapping doc_id to list of chunks
        """
        doc_chunks = defaultdict(list)
        
        for chunk in chunks:
            doc_id = chunk.get("doc_id")
            if doc_id:
                doc_chunks[doc_id].append(chunk)
        
        logger.debug(f"Grouped chunks into {len(doc_chunks)} documents")
        self.stats["documents_processed"] = len(doc_chunks)
        
        return dict(doc_chunks)
    
    def extract_metadata_from_chunks(self, chunks: List[Dict]) -> Dict:
        """
        Extract common metadata from document chunks.
        
        Args:
            chunks: List of chunks for a document
            
        Returns:
            Consolidated metadata
        """
        if not chunks:
            return {}
        
        # Use first chunk's metadata as base
        metadata = chunks[0].get("metadata", {}).copy()
        
        # Extract document-level information
        doc_id = chunks[0].get("doc_id")
        
        # Aggregate temporal information
        all_temporal = []
        for chunk in chunks:
            temporal = chunk.get("temporal", {})
            if temporal:
                all_temporal.append(temporal)
        
        if all_temporal:
            metadata["temporal_info"] = all_temporal[0]  # Use first chunk's temporal info
        
        # Calculate document statistics
        metadata["total_chunks"] = len(chunks)
        metadata["total_characters"] = sum(chunk.get("char_count", 0) for chunk in chunks)
        metadata["total_words"] = sum(chunk.get("word_count", 0) for chunk in chunks)
        
        return metadata
    
    def find_relations_for_document(self, relations: List[Dict], doc_id: str) -> List[Dict]:
        """
        Find all relations involving a specific document.
        
        Args:
            relations: List of all relations
            doc_id: Document ID to search for
            
        Returns:
            List of relations involving the document
        """
        doc_relations = []
        
        for relation in relations:
            source_chunk_id = relation.get("source_chunk_id", "")
            if source_chunk_id.startswith(doc_id):
                doc_relations.append(relation)
        
        return doc_relations
    
    def create_output_directory(self, vertical_name: str) -> Path:
        """
        Create output directory for vertical database.
        
        Args:
            vertical_name: Name of the vertical (e.g., 'legal', 'government_orders')
            
        Returns:
            Path to created directory
        """
        output_path = self.output_dir / vertical_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory: {output_path}")
        return output_path
    
    def save_json(self, data: Any, file_path: Path, description: str = "data"):
        """
        Save data to JSON file with error handling.
        
        Args:
            data: Data to save
            file_path: Path to save file
            description: Description for logging
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {description} to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving {description} to {file_path}: {e}")
            self.stats["processing_errors"].append({
                "type": "save_error",
                "file": str(file_path),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    def validate_database_structure(self, database: Dict, required_keys: List[str]) -> bool:
        """
        Validate that database has required structure.
        
        Args:
            database: Database dictionary to validate
            required_keys: List of required top-level keys
            
        Returns:
            True if valid, False otherwise
        """
        missing_keys = []
        for key in required_keys:
            if key not in database:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Database missing required keys: {missing_keys}")
            return False
        
        return True
    
    def generate_database_stats(self, database: Dict) -> Dict:
        """
        Generate statistics for a vertical database.
        
        Args:
            database: Vertical database dictionary
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_entries": 0,
            "creation_timestamp": datetime.now().isoformat(),
            "builder_version": "1.0.0"
        }
        
        # Count entries in main data structures
        for key, value in database.items():
            if isinstance(value, list):
                stats[f"{key}_count"] = len(value)
                stats["total_entries"] += len(value)
            elif isinstance(value, dict):
                stats[f"{key}_count"] = len(value)
                stats["total_entries"] += len(value)
        
        return stats
    
    @abstractmethod
    def build_database(self) -> Dict:
        """
        Build the vertical database.
        
        Must be implemented by concrete vertical builders.
        
        Returns:
            Complete vertical database dictionary
        """
        pass
    
    @abstractmethod
    def get_vertical_name(self) -> str:
        """
        Get the name of this vertical for directory creation.
        
        Returns:
            Vertical name (e.g., 'legal', 'government_orders')
        """
        pass
    
    def process_and_save(self) -> Dict:
        """
        Complete workflow: build database and save to files.
        
        Returns:
            Processing results including stats and file paths
        """
        logger.info(f"Starting {self.__class__.__name__} processing...")
        
        try:
            # Build the database
            database = self.build_database()
            
            if not database:
                logger.error("Database building failed - no data returned")
                return {"status": "failed", "error": "No database created"}
            
            # Create output directory
            vertical_name = self.get_vertical_name()
            output_path = self.create_output_directory(vertical_name)
            
            # Save main database file
            main_file = output_path / f"{vertical_name}_database.json"
            self.save_json(database, main_file, f"{vertical_name} database")
            
            # Generate and save statistics
            db_stats = self.generate_database_stats(database)
            stats_file = output_path / f"{vertical_name}_statistics.json"
            self.save_json(db_stats, stats_file, f"{vertical_name} statistics")
            
            # Save processing stats
            processing_stats = {
                "processing_stats": self.stats,
                "database_stats": db_stats,
                "output_files": {
                    "main_database": str(main_file),
                    "statistics": str(stats_file)
                }
            }
            
            processing_file = output_path / f"{vertical_name}_processing_log.json"
            self.save_json(processing_stats, processing_file, f"{vertical_name} processing log")
            
            logger.info(f"Completed {self.__class__.__name__} processing")
            
            return {
                "status": "success",
                "vertical_name": vertical_name,
                "output_directory": str(output_path),
                "database_entries": db_stats.get("total_entries", 0),
                "processing_stats": self.stats
            }
            
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__} processing: {e}")
            return {
                "status": "failed", 
                "error": str(e),
                "processing_stats": self.stats
            }


# Utility functions for common operations
def normalize_text_for_matching(text: str) -> str:
    """Normalize text for consistent matching across documents."""
    if not text:
        return ""
    
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    text = " ".join(text.split())
    
    return text


def extract_number_from_text(text: str, pattern: str) -> Optional[int]:
    """Extract number from text using regex pattern."""
    import re
    
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, IndexError):
            pass
    
    return None


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity between two strings."""
    if not text1 or not text2:
        return 0.0
    
    # Simple Jaccard similarity on words
    words1 = set(normalize_text_for_matching(text1).split())
    words2 = set(normalize_text_for_matching(text2).split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


# Example usage and testing
if __name__ == "__main__":
    # This is an abstract class, so we can't instantiate it directly
    # But we can test the utility functions
    
    print("Testing utility functions:")
    
    text1 = "Section 12(1)(c) of the RTE Act"
    text2 = "Section 12 of RTE Act 2009"
    
    normalized1 = normalize_text_for_matching(text1)
    normalized2 = normalize_text_for_matching(text2)
    
    similarity = calculate_text_similarity(text1, text2)
    
    print(f"Text 1: {text1}")
    print(f"Normalized 1: {normalized1}")
    print(f"Text 2: {text2}")
    print(f"Normalized 2: {normalized2}")
    print(f"Similarity: {similarity:.2f}")
    
    # Test number extraction
    go_text = "G.O.MS.No. 67 dated 15.04.2023"
    go_number = extract_number_from_text(go_text, r'(?:G\.?O\.?\s*(?:Ms\.?|MS\.?)?\s*(?:No\.?)?\s*)(\d+)')
    print(f"GO Text: {go_text}")
    print(f"Extracted Number: {go_number}")