"""
Enhanced Document ingestion pipeline for AP Policy Co-Pilot.

Enhanced 7-Stage Pipeline:
1. Document Discovery & Indexing
2. Text Extraction & Quality Check  
3. Text Cleaning & Classification
4. Section Parsing & Temporal Extraction
5. Entity & Relation Extraction
6. Chunking with Entity Propagation
7. Topic Matching & Quality Control

Provides comprehensive document processing capabilities including:
- Multi-strategy PDF text extraction
- Entity and relation extraction for knowledge graph
- Topic matching and bridge table population
- Quality control and validation
- Complete pipeline orchestration
"""

# Core extraction modules
from .pdf_extractor import PDFExtractor, extract_pdf_batch
from .text_cleaner import TextCleaner
from .section_parser import SectionParser
from .chunker import SmartChunker

# Enhanced analysis modules  
from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor, Relation
from .topic_matcher import TopicMatcher
from .temporal_extractor import TemporalExtractor

# Classification and quality modules
from .document_classifier import DocumentClassifier
from .quality_checker import QualityChecker
from .deduplicator import DocumentDeduplicator

# Metadata and pipeline modules
from .enhanced_metadata_builder import MetadataBuilder
from .enhanced_pipeline import EnhancedIngestionPipeline

# Legacy imports for compatibility - using enhanced metadata builder
try:
    from .metadata_builder import build_metadata, save_metadata
except ImportError:
    # Fallback to enhanced metadata builder
    from .enhanced_metadata_builder import MetadataBuilder
    def build_metadata(*args, **kwargs):
        builder = MetadataBuilder()
        return builder.build_metadata(*args, **kwargs)
    def save_metadata(*args, **kwargs):
        builder = MetadataBuilder()
        return builder.save_metadata(*args, **kwargs)

__version__ = "2.0.0"

__all__ = [
    # Core extraction
    "PDFExtractor",
    "extract_pdf_batch",
    "TextCleaner", 
    "SectionParser",
    "SmartChunker",
    
    # Entity and relation extraction
    "EntityExtractor",
    "RelationExtractor",
    "Relation",
    "TopicMatcher",
    "TemporalExtractor",
    
    # Classification and quality
    "DocumentClassifier",
    "QualityChecker", 
    "DocumentDeduplicator",
    
    # Metadata and pipeline
    "MetadataBuilder",
    "EnhancedIngestionPipeline",
    
    # Legacy functions
    "build_metadata",
    "save_metadata"
]