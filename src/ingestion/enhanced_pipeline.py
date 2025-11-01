"""
Enhanced Ingestion Pipeline for education policy documents.

Comprehensive 7-stage pipeline that extracts everything needed for the 
knowledge graph in a single pass:
- Text extraction with quality checks
- Entity and relation extraction  
- Bridge topic matching
- Quality control and validation
- Complete output generation
"""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import traceback

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

# Import all pipeline components
from .pdf_extractor import PDFExtractor
from .text_cleaner import TextCleaner
from .section_parser import SectionParser
from .semantic_chunker import SemanticChunker
from .enhanced_metadata_builder import MetadataBuilder
from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor
from .topic_matcher import TopicMatcher
from .temporal_extractor import TemporalExtractor
from .document_classifier import DocumentClassifier
from .quality_checker import QualityChecker
from .deduplicator import DocumentDeduplicator

logger = get_logger(__name__)


class EnhancedIngestionPipeline:
    """
    Complete 7-stage ingestion pipeline with entity awareness.
    
    Stages:
    0. Document Discovery & Indexing
    1. Text Extraction & Quality Check
    2. Text Cleaning & Classification
    3. Section Parsing & Temporal Extraction
    4. Entity & Relation Extraction
    5. Chunking with Entity Propagation
    6. Topic Matching & Bridge Population
    7. Quality Control & Output Generation
    """
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the enhanced pipeline.
        
        Args:
            data_dir: Path to data directory
            output_dir: Path to output directory
        """
        if data_dir is None:
            data_dir = project_root / "data"
        if output_dir is None:
            output_dir = project_root / "data" / "processed"
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize all pipeline components
        self._init_components()
        
        # Pipeline statistics
        self.stats = {
            "documents_processed": 0,
            "documents_successful": 0,
            "documents_failed": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relations": 0,
            "total_topics_matched": 0,
            "processing_errors": [],
            "quality_distribution": {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0, "critical": 0}
        }
        
        logger.info("EnhancedIngestionPipeline initialized")
    
    def _init_components(self):
        """Initialize all pipeline components."""
        try:
            self.pdf_extractor = PDFExtractor()
            self.text_cleaner = TextCleaner()
            self.section_parser = SectionParser()
            self.chunker = SemanticChunker()
            self.metadata_builder = MetadataBuilder()
            self.entity_extractor = EntityExtractor(str(self.data_dir))
            self.relation_extractor = RelationExtractor()
            self.topic_matcher = TopicMatcher(str(self.data_dir))
            self.temporal_extractor = TemporalExtractor()
            self.document_classifier = DocumentClassifier()
            self.quality_checker = QualityChecker()
            self.deduplicator = DocumentDeduplicator()
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline components: {e}")
            raise
    
    def discover_documents(self, documents_dir: str) -> List[Dict]:
        """
        Stage 0: Document Discovery & Indexing
        
        Args:
            documents_dir: Directory containing documents to process
            
        Returns:
            List of document metadata for processing
        """
        logger.info(f"Stage 0: Discovering documents in {documents_dir}")
        
        documents_path = Path(documents_dir)
        if not documents_path.exists():
            logger.error(f"Documents directory not found: {documents_dir}")
            return []
        
        # Find all PDF files
        pdf_files = list(documents_path.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Create document index
        document_index = []
        for pdf_file in pdf_files:
            doc_info = {
                "file_path": str(pdf_file),
                "file_name": pdf_file.name,
                "doc_id": self._generate_doc_id(pdf_file),
                "file_size": pdf_file.stat().st_size,
                "parent_folders": self._get_parent_folders(pdf_file, documents_path),
                "discovery_timestamp": datetime.now().isoformat()
            }
            document_index.append(doc_info)
        
        # Save document index
        index_file = self.output_dir / "document_index.json"
        with open(index_file, 'w') as f:
            json.dump(document_index, f, indent=2)
        
        logger.info(f"Document index saved to {index_file}")
        return document_index
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID from file path."""
        # Use file name without extension + parent folder
        name_part = file_path.stem.replace(' ', '_').replace('-', '_')
        parent_part = file_path.parent.name.replace(' ', '_').replace('-', '_')
        return f"{parent_part}_{name_part}".lower()
    
    def _get_parent_folders(self, file_path: Path, base_path: Path) -> List[str]:
        """Extract parent folder hierarchy."""
        relative_path = file_path.relative_to(base_path)
        return list(relative_path.parent.parts)
    
    def extract_and_validate_text(self, file_path: str) -> Tuple[str, Dict]:
        """
        Stage 1: Text Extraction & Quality Check
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, extraction_quality_report)
        """
        logger.debug(f"Stage 1: Extracting text from {file_path}")
        
        try:
            # Extract text using multiple strategies
            extraction_result = self.pdf_extractor.extract(Path(file_path))
            
            if not extraction_result or not extraction_result.get("text") or not extraction_result.get("success"):
                return "", {"score": 0, "issues": ["extraction_failed"]}
            
            text = extraction_result["text"]
            
            # Quality check on extraction
            quality_report = self.quality_checker.check_text_extraction_quality(text)
            
            logger.debug(f"Text extraction quality score: {quality_report['score']}")
            
            return text, quality_report
            
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return "", {"score": 0, "issues": ["extraction_error"], "error": str(e)}
    
    def clean_and_classify_text(self, text: str, title: str = "", file_path: str = "") -> Tuple[str, Dict, Dict]:
        """
        Stage 2: Text Cleaning & Classification
        
        Args:
            text: Raw extracted text
            title: Document title
            file_path: Original file path
            
        Returns:
            Tuple of (cleaned_text, classification_result, temporal_info)
        """
        logger.debug("Stage 2: Cleaning and classifying text")
        
        try:
            # Clean text
            cleaned_text = self.text_cleaner.clean(text)
            
            # Extract basic metadata for classification
            metadata = {"parent_folders": self._get_parent_folders(Path(file_path), Path(file_path).parent.parent)}
            
            # Classify document type
            classification_result = self.document_classifier.classify_document(
                cleaned_text, title, metadata
            )
            
            # Extract temporal information
            temporal_info = self.temporal_extractor.extract_all_temporal(cleaned_text)
            
            logger.debug(f"Document classified as: {classification_result.get('predicted_type')}")
            
            return cleaned_text, classification_result, temporal_info
            
        except Exception as e:
            logger.error(f"Error in text cleaning/classification: {e}")
            return text, {"predicted_type": "unknown", "confidence": 0}, {"dates": []}
    
    def parse_sections_and_structure(self, text: str, doc_type: str) -> List[Dict]:
        """
        Stage 3: Section Parsing & Structure Analysis
        
        Args:
            text: Cleaned text
            doc_type: Document type from classification
            
        Returns:
            List of section dictionaries
        """
        logger.debug("Stage 3: Parsing sections and structure")
        
        try:
            # Parse into sections based on document type
            sections = self.section_parser.parse(text, doc_type)
            
            logger.debug(f"Parsed {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"Error in section parsing: {e}")
            # Return single section with full text as fallback
            return [{
                "section_id": "full_text",
                "title": "Full Document",
                "content": text,
                "level": 0,
                "position": (0, len(text))
            }]
    
    def extract_entities_and_relations(self, sections: List[Dict], doc_id: str) -> Tuple[Dict, List[Dict]]:
        """
        Stage 4: Entity & Relation Extraction
        
        Args:
            sections: Parsed sections
            doc_id: Document ID
            
        Returns:
            Tuple of (combined_entities, all_relations)
        """
        logger.debug("Stage 4: Extracting entities and relations")
        
        try:
            combined_entities = {
                "legal_refs": [],
                "go_refs": [],
                "schemes": [],
                "districts": [],
                "metrics": [],
                "social_categories": [],
                "school_types": [],
                "educational_levels": [],
                "spacy_entities": [],
                "keywords": []
            }
            all_relations = []
            
            # Extract from each section
            for section in sections:
                section_text = section.get("content", "")
                section_id = section.get("section_id", "unknown")
                
                # Extract entities
                section_entities = self.entity_extractor.extract_all_entities(section_text)
                
                # Extract relations
                chunk_id = f"{doc_id}_{section_id}"
                section_relations = self.relation_extractor.extract_all_relations(section_text, chunk_id)
                
                # Store entities in section
                section["entities"] = section_entities
                
                # Combine entities (deduplicate)
                for entity_type, entity_list in section_entities.items():
                    if isinstance(entity_list, list):
                        combined_entities[entity_type].extend(entity_list)
                
                # Collect relations
                all_relations.extend(section_relations)
            
            # Deduplicate combined entities (handle both strings and dicts)
            for entity_type in combined_entities:
                if isinstance(combined_entities[entity_type], list):
                    combined_entities[entity_type] = self._deduplicate_entities(combined_entities[entity_type])
            
            # Convert relations to dicts
            relations_dicts = self.relation_extractor.relations_to_dict(all_relations)
            
            logger.debug(f"Extracted {sum(len(v) if isinstance(v, list) else 0 for v in combined_entities.values())} entities")
            logger.debug(f"Extracted {len(relations_dicts)} relations")
            
            return combined_entities, relations_dicts
            
        except Exception as e:
            logger.error(f"Error in entity/relation extraction: {e}")
            return {}, []
    
    def create_chunks_with_entities(self, sections: List[Dict], doc_id: str, metadata: Dict) -> List[Dict]:
        """
        Stage 5: Chunking with Entity Propagation
        
        Args:
            sections: Sections with entities
            doc_id: Document ID
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries with entities
        """
        logger.debug("Stage 5: Creating chunks with entity propagation")
        
        try:
            all_chunks = []
            
            for section in sections:
                section_text = section.get("text", section.get("content", ""))
                section_id = section.get("section_id")
                section_entities = section.get("entities", {})
                
                # Create chunks for this section
                chunks = self.chunker.chunk_section(
                    section_text=section_text,
                    doc_id=doc_id,
                    section_id=section_id,
                    metadata=metadata
                )
                
                # Propagate section entities to each chunk
                for chunk in chunks:
                    chunk["entities"] = section_entities.copy()
                    
                    # Extract chunk-specific entities (more granular)
                    chunk_text = chunk.get("text", "")
                    chunk_entities = self.entity_extractor.extract_all_entities(chunk_text)
                    
                    # Merge chunk entities with section entities
                    for entity_type, entity_list in chunk_entities.items():
                        if isinstance(entity_list, list):
                            existing = chunk["entities"].get(entity_type, [])
                            combined = existing + entity_list
                            chunk["entities"][entity_type] = self._deduplicate_entities(combined)
                
                all_chunks.extend(chunks)
            
            logger.debug(f"Created {len(all_chunks)} chunks with entities")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error in chunking: {e}")
            return []
    
    def match_topics_and_bridge(self, chunks: List[Dict]) -> Dict:
        """
        Stage 6: Topic Matching & Bridge Population
        
        Args:
            chunks: Chunks with entities
            
        Returns:
            Topic matching results
        """
        logger.debug("Stage 6: Matching topics and building bridge connections")
        
        try:
            # Match each chunk to bridge topics
            for chunk in chunks:
                chunk_text = chunk.get("text", "")
                chunk_entities = chunk.get("entities", {})
                chunk_id = chunk.get("chunk_id")
                
                topic_matches = self.topic_matcher.match_chunk_to_topics(
                    chunk_text, chunk_entities, chunk_id
                )
                
                chunk["bridge_topics"] = topic_matches
            
            # Aggregate document-level topic matches
            topic_to_chunks = self.topic_matcher.match_document_to_topics(chunks)
            
            # Generate statistics
            topic_stats = self.topic_matcher.get_topic_statistics(topic_to_chunks)
            
            logger.debug(f"Matched chunks to {len(topic_to_chunks)} bridge topics")
            
            return {
                "topic_to_chunks": topic_to_chunks,
                "statistics": topic_stats
            }
            
        except Exception as e:
            logger.error(f"Error in topic matching: {e}")
            return {"topic_to_chunks": {}, "statistics": {}}
    
    def validate_quality_and_generate_outputs(
        self, 
        text: str, 
        metadata: Dict, 
        entities: Dict, 
        relations: List[Dict], 
        classification_result: Dict,
        chunks: List[Dict],
        topic_results: Dict
    ) -> Dict:
        """
        Stage 7: Quality Control & Output Generation
        
        Args:
            text: Processed text
            metadata: Document metadata
            entities: Extracted entities
            relations: Extracted relations
            classification_result: Classification result
            chunks: Final chunks
            topic_results: Topic matching results
            
        Returns:
            Complete processing result
        """
        logger.debug("Stage 7: Quality validation and output generation")
        
        try:
            # Comprehensive quality check
            quality_report = self.quality_checker.check_document_quality(
                text, metadata, entities, relations, classification_result
            )
            
            # Generate processing summary
            processing_summary = {
                "document_info": {
                    "doc_id": metadata.get("doc_id"),
                    "title": metadata.get("title"),
                    "doc_type": classification_result.get("predicted_type"),
                    "classification_confidence": classification_result.get("confidence"),
                    "file_path": metadata.get("file_path"),
                    "processing_timestamp": datetime.now().isoformat()
                },
                "processing_results": {
                    "text_length": len(text),
                    "chunk_count": len(chunks),
                    "entity_count": sum(len(v) if isinstance(v, list) else 0 for v in entities.values()),
                    "relation_count": len(relations),
                    "topics_matched": len(topic_results.get("topic_to_chunks", {})),
                    "quality_score": quality_report.get("overall", {}).get("overall_score", 0),
                    "quality_level": quality_report.get("overall", {}).get("quality_level", "unknown")
                },
                "quality_report": quality_report,
                "topic_results": topic_results
            }
            
            return processing_summary
            
        except Exception as e:
            logger.error(f"Error in quality validation: {e}")
            return {"error": str(e)}
    
    def process_single_document(self, file_path: str, doc_info: Dict) -> Dict:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to document file
            doc_info: Document information from discovery
            
        Returns:
            Complete processing result
        """
        doc_id = doc_info.get("doc_id")
        logger.info(f"Processing document: {doc_id}")
        
        try:
            # Stage 1: Text Extraction & Quality Check
            text, extraction_quality = self.extract_and_validate_text(file_path)
            
            if not text or extraction_quality.get("score", 0) < 20:
                logger.warning(f"Poor text extraction quality for {doc_id}")
                return {
                    "doc_id": doc_id,
                    "status": "failed",
                    "stage": "text_extraction",
                    "error": "Poor text extraction quality",
                    "quality_score": extraction_quality.get("score", 0)
                }
            
            # Stage 2: Text Cleaning & Classification
            cleaned_text, classification_result, temporal_info = self.clean_and_classify_text(
                text, doc_info.get("file_name", ""), file_path
            )
            
            # Stage 3: Section Parsing
            sections = self.parse_sections_and_structure(
                cleaned_text, classification_result.get("predicted_type", "unknown")
            )
            
            # Stage 4: Entity & Relation Extraction
            entities, relations = self.extract_entities_and_relations(sections, doc_id)
            
            # Build metadata
            metadata = self.metadata_builder.build_metadata(
                file_path=file_path,
                doc_id=doc_id,
                classification_result=classification_result,
                temporal_info=temporal_info,
                entities=entities
            )
            
            # Stage 5: Chunking with Entity Propagation
            chunks = self.create_chunks_with_entities(sections, doc_id, metadata)
            
            # Stage 6: Topic Matching & Bridge Population
            topic_results = self.match_topics_and_bridge(chunks)
            
            # Stage 7: Quality Control & Output Generation
            processing_summary = self.validate_quality_and_generate_outputs(
                cleaned_text, metadata, entities, relations, 
                classification_result, chunks, topic_results
            )
            
            # Update statistics
            self._update_stats(processing_summary, entities, relations, topic_results)
            
            # Save outputs
            self._save_document_outputs(doc_id, {
                "metadata": metadata,
                "sections": sections,
                "chunks": chunks,
                "entities": entities,
                "relations": relations,
                "topic_results": topic_results,
                "processing_summary": processing_summary
            })
            
            logger.info(f"Successfully processed document: {doc_id}")
            return processing_summary
            
        except Exception as e:
            error_msg = f"Error processing document {doc_id}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.stats["documents_failed"] += 1
            self.stats["processing_errors"].append({
                "doc_id": doc_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "doc_id": doc_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _update_stats(self, processing_summary: Dict, entities: Dict, relations: List, topic_results: Dict):
        """Update pipeline statistics."""
        self.stats["documents_processed"] += 1
        
        if processing_summary.get("error"):
            self.stats["documents_failed"] += 1
        else:
            self.stats["documents_successful"] += 1
            
            # Update entity/relation/topic counts
            entity_count = sum(len(v) if isinstance(v, list) else 0 for v in entities.values())
            self.stats["total_entities"] += entity_count
            self.stats["total_relations"] += len(relations)
            
            chunks = processing_summary.get("processing_results", {}).get("chunk_count", 0)
            self.stats["total_chunks"] += chunks
            
            topics = len(topic_results.get("topic_to_chunks", {}))
            self.stats["total_topics_matched"] += topics
            
            # Update quality distribution
            quality_level = processing_summary.get("processing_results", {}).get("quality_level", "unknown")
            if quality_level in self.stats["quality_distribution"]:
                self.stats["quality_distribution"][quality_level] += 1
    
    def _save_document_outputs(self, doc_id: str, outputs: Dict):
        """Save document processing outputs."""
        try:
            # Create output directories
            chunks_dir = self.output_dir / "chunks"
            entities_dir = self.output_dir / "entities"
            relations_dir = self.output_dir / "relations"
            metadata_dir = self.output_dir / "metadata"
            
            for dir_path in [chunks_dir, entities_dir, relations_dir, metadata_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Save chunks (append to master file)
            chunks_file = chunks_dir / "all_chunks.jsonl"
            with open(chunks_file, 'a') as f:
                for chunk in outputs["chunks"]:
                    f.write(json.dumps(chunk) + "\n")
            
            # Save entities
            entities_file = entities_dir / f"{doc_id}_entities.json"
            with open(entities_file, 'w') as f:
                json.dump(outputs["entities"], f, indent=2)
            
            # Save relations
            if outputs["relations"]:
                relations_file = relations_dir / f"{doc_id}_relations.json"
                with open(relations_file, 'w') as f:
                    json.dump(outputs["relations"], f, indent=2)
            
            # Save metadata
            metadata_file = metadata_dir / f"{doc_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(outputs["metadata"], f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving outputs for {doc_id}: {e}")
    
    def process_corpus(self, documents_dir: str, max_documents: Optional[int] = None) -> Dict:
        """
        Process entire document corpus through the pipeline.
        
        Args:
            documents_dir: Directory containing documents
            max_documents: Maximum number of documents to process (for testing)
            
        Returns:
            Complete processing results and statistics
        """
        logger.info(f"Starting corpus processing from {documents_dir}")
        
        # Stage 0: Document Discovery
        document_index = self.discover_documents(documents_dir)
        
        if not document_index:
            logger.error("No documents found for processing")
            return {"error": "No documents found"}
        
        # Limit documents if specified
        if max_documents:
            document_index = document_index[:max_documents]
            logger.info(f"Limited to {len(document_index)} documents for processing")
        
        # Process each document
        processing_results = []
        
        for doc_info in document_index:
            file_path = doc_info["file_path"]
            result = self.process_single_document(file_path, doc_info)
            processing_results.append(result)
            
            # Log progress
            if self.stats["documents_processed"] % 10 == 0:
                logger.info(f"Processed {self.stats['documents_processed']} documents")
        
        # Generate final corpus statistics
        corpus_stats = self._generate_corpus_statistics(processing_results)
        
        # Save final outputs
        self._save_corpus_outputs(corpus_stats, processing_results)
        
        logger.info(f"Corpus processing complete: {self.stats['documents_successful']}/{self.stats['documents_processed']} documents successful")
        
        return {
            "statistics": corpus_stats,
            "processing_results": processing_results,
            "pipeline_stats": self.stats
        }
    
    def _generate_corpus_statistics(self, processing_results: List[Dict]) -> Dict:
        """Generate comprehensive corpus statistics."""
        return {
            "corpus_overview": {
                "total_documents": len(processing_results),
                "successful_documents": self.stats["documents_successful"],
                "failed_documents": self.stats["documents_failed"],
                "success_rate": self.stats["documents_successful"] / len(processing_results) if processing_results else 0
            },
            "content_statistics": {
                "total_chunks": self.stats["total_chunks"],
                "total_entities": self.stats["total_entities"],
                "total_relations": self.stats["total_relations"],
                "total_topics_matched": self.stats["total_topics_matched"],
                "avg_chunks_per_doc": self.stats["total_chunks"] / self.stats["documents_successful"] if self.stats["documents_successful"] > 0 else 0,
                "avg_entities_per_doc": self.stats["total_entities"] / self.stats["documents_successful"] if self.stats["documents_successful"] > 0 else 0
            },
            "quality_distribution": self.stats["quality_distribution"],
            "processing_timestamp": datetime.now().isoformat()
        }
    
    def _save_corpus_outputs(self, corpus_stats: Dict, processing_results: List[Dict]):
        """Save final corpus-level outputs."""
        try:
            # Save corpus statistics
            stats_file = self.output_dir / "corpus_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(corpus_stats, f, indent=2)
            
            # Save processing log
            log_file = self.output_dir / "processing_log.json"
            with open(log_file, 'w') as f:
                json.dump({
                    "pipeline_stats": self.stats,
                    "processing_results": processing_results
                }, f, indent=2)
            
            # Save ingestion quality report
            quality_file = self.output_dir / "ingestion_quality_report.json"
            quality_report = {
                "summary": corpus_stats,
                "errors": self.stats["processing_errors"],
                "recommendations": self._generate_corpus_recommendations()
            }
            with open(quality_file, 'w') as f:
                json.dump(quality_report, f, indent=2)
            
            logger.info(f"Corpus outputs saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving corpus outputs: {e}")
    
    def _deduplicate_entities(self, entity_list: List) -> List:
        """
        Deduplicate entities that can be strings or dictionaries.
        
        Args:
            entity_list: List of entities (strings or dicts)
            
        Returns:
            Deduplicated list of entities
        """
        if not entity_list:
            return []
        
        # Handle dictionaries (like spacy_entities)
        if isinstance(entity_list[0], dict):
            seen = set()
            unique_entities = []
            for entity in entity_list:
                # Create a hashable key from the entity
                key = json.dumps(entity, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            return unique_entities
        else:
            # Handle strings - use simple deduplication
            return list(dict.fromkeys(entity_list))
    
    def _generate_corpus_recommendations(self) -> List[str]:
        """Generate recommendations for corpus improvement."""
        recommendations = []
        
        success_rate = self.stats["documents_successful"] / self.stats["documents_processed"] if self.stats["documents_processed"] > 0 else 0
        
        if success_rate < 0.8:
            recommendations.append("Low success rate - review text extraction settings")
        
        if self.stats["quality_distribution"]["critical"] > 0:
            recommendations.append("Some documents have critical quality issues - manual review needed")
        
        if self.stats["total_entities"] / self.stats["documents_successful"] < 10 if self.stats["documents_successful"] > 0 else True:
            recommendations.append("Low entity density - review entity extraction patterns")
        
        if len(self.stats["processing_errors"]) > 0:
            recommendations.append("Processing errors detected - check error log for details")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced pipeline
    pipeline = EnhancedIngestionPipeline()
    
    # Test with a small subset
    test_docs_dir = "/Users/nitin/Documents/AI policy Assistant/data/raw/Documents/Critical Priority"
    
    print("Testing Enhanced Ingestion Pipeline...")
    results = pipeline.process_corpus(test_docs_dir, max_documents=2)
    
    print("\nPipeline Results:")
    print(f"Documents processed: {results['pipeline_stats']['documents_processed']}")
    print(f"Success rate: {results['statistics']['corpus_overview']['success_rate']:.1%}")
    print(f"Total chunks: {results['statistics']['content_statistics']['total_chunks']}")
    print(f"Total entities: {results['statistics']['content_statistics']['total_entities']}")
    print(f"Total relations: {results['statistics']['content_statistics']['total_relations']}")