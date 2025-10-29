"""
Enhanced Metadata Builder for education policy documents.

Builds comprehensive metadata including entity-level information, temporal data,
classification results, and quality metrics.
"""

import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataBuilder:
    """
    Enhanced metadata builder that creates comprehensive document metadata.
    
    Combines traditional metadata (title, date, type) with modern entity-aware
    metadata (extracted entities, temporal info, classification confidence).
    """
    
    def __init__(self):
        """Initialize metadata builder."""
        self._init_extraction_patterns()
        logger.info("Enhanced MetadataBuilder initialized")
    
    def _init_extraction_patterns(self):
        """Initialize patterns for extracting metadata from file paths and names."""
        # Year extraction patterns
        self.year_patterns = [
            re.compile(r'\b(19|20)\d{2}\b'),  # 4-digit years
            re.compile(r'\b(\d{2})-(\d{2})\b')  # 2-digit year ranges like 22-23
        ]
        
        # GO number patterns in filenames
        self.go_patterns = [
            re.compile(r'G\.?O\.?\s*(?:Ms\.?|MS\.?)?\s*(?:No\.?)?\s*(\d+)', re.IGNORECASE),
            re.compile(r'MS\s*(\d+)', re.IGNORECASE)
        ]
        
        # Document type indicators in paths/names
        self.type_indicators = {
            'act': ['act', 'legislation', 'statute'],
            'rule': ['rule', 'regulation'],
            'government_order': ['go', 'order', 'circular'],
            'judicial': ['judgment', 'court', 'vs', 'versus'],
            'data_report': ['report', 'data', 'statistics', 'udise'],
            'budget_finance': ['budget', 'finance', 'expenditure']
        }
    
    def extract_title_from_filename(self, file_path: str) -> str:
        """
        Extract a clean title from filename.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Cleaned title string
        """
        file_path = Path(file_path)
        
        # Start with filename without extension
        title = file_path.stem
        
        # Clean up common patterns
        title = title.replace('_', ' ').replace('-', ' ')
        title = re.sub(r'\s+', ' ', title)  # Multiple spaces to single
        title = title.strip()
        
        # Capitalize properly
        title = title.title()
        
        # Fix common abbreviations
        replacements = {
            'Go ': 'GO ',
            'Ms ': 'MS ',
            'Rte ': 'RTE ',
            'Ap ': 'AP ',
            'Dt ': 'Dt.',
            'No ': 'No.'
        }
        
        for old, new in replacements.items():
            title = title.replace(old, new)
        
        return title
    
    def extract_year_from_path(self, file_path: str, text: str = "") -> Optional[int]:
        """
        Extract year from file path or text content.
        
        Args:
            file_path: Path to the file
            text: Document text content (optional)
            
        Returns:
            Extracted year or None
        """
        # First try file path
        path_text = str(file_path)
        
        # Look for 4-digit years
        for pattern in self.year_patterns:
            matches = pattern.findall(path_text)
            if matches:
                if isinstance(matches[0], tuple):
                    # Handle year ranges like 22-23
                    start_year = int(matches[0][0])
                    if start_year < 50:  # Assume 2000s
                        return 2000 + start_year
                    else:  # Assume 1900s
                        return 1900 + start_year
                else:
                    year = int(matches[0])
                    if 1900 <= year <= 2030:
                        return year
        
        # If not found in path, try text content
        if text:
            text_sample = text[:1000]  # First 1000 chars
            for pattern in self.year_patterns:
                matches = pattern.findall(text_sample)
                if matches:
                    if isinstance(matches[0], tuple):
                        start_year = int(matches[0][0])
                        if start_year < 50:
                            return 2000 + start_year
                        else:
                            return 1900 + start_year
                    else:
                        year = int(matches[0])
                        if 1900 <= year <= 2030:
                            return year
        
        return None
    
    def extract_go_number_from_path(self, file_path: str) -> Optional[Dict]:
        """
        Extract GO number from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            GO number information or None
        """
        path_text = str(file_path)
        
        for pattern in self.go_patterns:
            match = pattern.search(path_text)
            if match:
                go_number = int(match.group(1))
                
                # Try to extract year as well
                year_match = re.search(r'(\d{4})', path_text)
                year = int(year_match.group(1)) if year_match else None
                
                return {
                    "number": go_number,
                    "year": year,
                    "department": "MS"  # Default
                }
        
        return None
    
    def infer_document_type_from_path(self, file_path: str) -> str:
        """
        Infer document type from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Inferred document type
        """
        path_lower = str(file_path).lower()
        
        # Check folder names and filename
        for doc_type, indicators in self.type_indicators.items():
            for indicator in indicators:
                if indicator in path_lower:
                    return doc_type
        
        return "unknown"
    
    def extract_parent_folders(self, file_path: str, base_path: str = None) -> List[str]:
        """
        Extract parent folder hierarchy.
        
        Args:
            file_path: Path to the file
            base_path: Base path to make relative from
            
        Returns:
            List of parent folder names
        """
        file_path = Path(file_path)
        
        if base_path:
            try:
                relative_path = file_path.relative_to(Path(base_path))
                return list(relative_path.parent.parts)
            except ValueError:
                # file_path is not relative to base_path
                pass
        
        # Return last few parts of the path
        return list(file_path.parent.parts[-3:])  # Last 3 levels
    
    def calculate_priority_from_path(self, file_path: str) -> str:
        """
        Calculate document priority from path structure.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Priority level (critical, high, medium, low)
        """
        path_lower = str(file_path).lower()
        
        if 'critical' in path_lower:
            return 'critical'
        elif 'high' in path_lower:
            return 'high'
        elif 'medium' in path_lower:
            return 'medium'
        else:
            return 'low'
    
    def build_basic_metadata(
        self, 
        file_path: str, 
        doc_id: str,
        text: str = "",
        base_path: str = None
    ) -> Dict:
        """
        Build basic metadata from file information.
        
        Args:
            file_path: Path to the document file
            doc_id: Unique document identifier
            text: Document text content (optional)
            base_path: Base path for relative folder extraction
            
        Returns:
            Basic metadata dictionary
        """
        file_path_obj = Path(file_path)
        
        # Extract basic information
        title = self.extract_title_from_filename(file_path)
        year = self.extract_year_from_path(file_path, text)
        doc_type = self.infer_document_type_from_path(file_path)
        parent_folders = self.extract_parent_folders(file_path, base_path)
        priority = self.calculate_priority_from_path(file_path)
        go_info = self.extract_go_number_from_path(file_path)
        
        # File information
        file_stats = file_path_obj.stat()
        
        metadata = {
            # Basic identifiers
            "doc_id": doc_id,
            "title": title,
            "file_path": str(file_path),
            "file_name": file_path_obj.name,
            "file_format": file_path_obj.suffix.lower(),
            
            # Document classification
            "doc_type": doc_type,
            "priority": priority,
            "parent_folders": parent_folders,
            
            # Temporal information
            "year": year,
            "creation_timestamp": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modification_timestamp": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "processing_timestamp": datetime.now().isoformat(),
            
            # File properties
            "file_size": file_stats.st_size,
            
            # Government Order specific
            "go_info": go_info,
            
            # Status
            "status": "active",
            "version": "1.0"
        }
        
        return metadata
    
    def enhance_metadata_with_classification(
        self, 
        metadata: Dict, 
        classification_result: Dict
    ) -> Dict:
        """
        Enhance metadata with document classification results.
        
        Args:
            metadata: Basic metadata
            classification_result: Document classification result
            
        Returns:
            Enhanced metadata
        """
        # Update document type if classification is more confident
        predicted_type = classification_result.get("predicted_type")
        confidence = classification_result.get("confidence", 0)
        
        if predicted_type and confidence > 0.6:
            metadata["doc_type"] = predicted_type
        
        # Add classification details
        metadata["classification"] = {
            "predicted_type": predicted_type,
            "confidence": confidence,
            "confidence_level": classification_result.get("confidence_level"),
            "method": classification_result.get("method"),
            "scores": classification_result.get("scores", {}),
            "structural_features": classification_result.get("structural_features", {})
        }
        
        return metadata
    
    def enhance_metadata_with_temporal(
        self, 
        metadata: Dict, 
        temporal_info: Dict
    ) -> Dict:
        """
        Enhance metadata with temporal information.
        
        Args:
            metadata: Basic metadata
            temporal_info: Temporal extraction results
            
        Returns:
            Enhanced metadata
        """
        # Add temporal extraction results
        metadata["temporal"] = temporal_info
        
        # Extract key dates for top-level metadata
        dates = temporal_info.get("dates", [])
        if dates:
            # Find the most relevant date (document/issue date)
            issue_dates = [d for d in dates if d.get("type") in ["document_date", "issue_date"]]
            effective_dates = [d for d in dates if d.get("type") == "effective_date"]
            
            if issue_dates:
                metadata["issue_date"] = issue_dates[0]["normalized_date"]
            if effective_dates:
                metadata["effective_date"] = effective_dates[0]["normalized_date"]
        
        # Extract academic years
        academic_years = temporal_info.get("academic_years", [])
        if academic_years:
            latest_ay = max(academic_years, key=lambda x: x.get("start_year", 0))
            metadata["academic_year"] = latest_ay["year_range"]
        
        # Update year if we found a better one
        summary = temporal_info.get("summary", {})
        primary_years = summary.get("primary_years", [])
        if primary_years and not metadata.get("year"):
            metadata["year"] = primary_years[0]
        
        return metadata
    
    def enhance_metadata_with_entities(
        self, 
        metadata: Dict, 
        entities: Dict
    ) -> Dict:
        """
        Enhance metadata with entity extraction results.
        
        Args:
            metadata: Basic metadata
            entities: Entity extraction results
            
        Returns:
            Enhanced metadata
        """
        # Add entity summary
        entity_counts = {
            entity_type: len(entity_list) if isinstance(entity_list, list) else 0
            for entity_type, entity_list in entities.items()
        }
        
        metadata["entities"] = {
            "counts": entity_counts,
            "total_entities": sum(entity_counts.values()),
            "entity_density": sum(entity_counts.values()) / len(metadata.get("text", "")) * 1000 if metadata.get("text") else 0
        }
        
        # Extract key entities for top-level metadata
        if entities.get("schemes"):
            metadata["schemes"] = entities["schemes"][:5]  # Top 5 schemes
        
        if entities.get("districts"):
            metadata["districts"] = entities["districts"][:5]  # Top 5 districts
        
        if entities.get("metrics"):
            metadata["metrics"] = entities["metrics"][:5]  # Top 5 metrics
        
        # Extract GO references for government orders
        go_refs = entities.get("go_refs", [])
        if go_refs and metadata.get("doc_type") == "government_order":
            # Update GO info with extracted data
            go_ref = go_refs[0]  # Use first GO reference
            if not metadata.get("go_info"):
                metadata["go_info"] = {}
            
            metadata["go_info"].update({
                "number": go_ref.get("number"),
                "year": go_ref.get("year"),
                "department": go_ref.get("department"),
                "date": go_ref.get("date")
            })
        
        # Extract legal references for acts/rules
        legal_refs = entities.get("legal_refs", [])
        if legal_refs and metadata.get("doc_type") in ["act", "rule"]:
            sections = [ref for ref in legal_refs if ref.get("type") == "section"]
            if sections:
                metadata["sections_count"] = len(sections)
                metadata["key_sections"] = [ref.get("number") for ref in sections[:10]]
        
        return metadata
    
    def build_metadata(
        self,
        file_path: str,
        doc_id: str,
        text: str = "",
        classification_result: Optional[Dict] = None,
        temporal_info: Optional[Dict] = None,
        entities: Optional[Dict] = None,
        base_path: str = None
    ) -> Dict:
        """
        Build comprehensive metadata for a document.
        
        Args:
            file_path: Path to document file
            doc_id: Unique document identifier
            text: Document text content
            classification_result: Document classification result
            temporal_info: Temporal extraction results
            entities: Entity extraction results
            base_path: Base path for relative folder extraction
            
        Returns:
            Complete metadata dictionary
        """
        logger.debug(f"Building metadata for document: {doc_id}")
        
        try:
            # Start with basic metadata
            metadata = self.build_basic_metadata(file_path, doc_id, text, base_path)
            
            # Enhance with classification results
            if classification_result:
                metadata = self.enhance_metadata_with_classification(metadata, classification_result)
            
            # Enhance with temporal information
            if temporal_info:
                metadata = self.enhance_metadata_with_temporal(metadata, temporal_info)
            
            # Enhance with entity information
            if entities:
                metadata = self.enhance_metadata_with_entities(metadata, entities)
            
            logger.debug(f"Built metadata for {doc_id}: type={metadata.get('doc_type')}, year={metadata.get('year')}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error building metadata for {doc_id}: {e}")
            # Return minimal metadata
            return {
                "doc_id": doc_id,
                "title": Path(file_path).stem,
                "file_path": file_path,
                "doc_type": "unknown",
                "status": "error",
                "error": str(e),
                "processing_timestamp": datetime.now().isoformat()
            }
    
    def save_metadata(self, metadata: Dict, output_path: str):
        """
        Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to save metadata
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Metadata saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving metadata to {output_path}: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test the metadata builder
    builder = MetadataBuilder()
    
    # Test file path
    test_file = "/data/raw/Documents/GO/G.O.MS.No.67-Dt.15.04.2023.pdf"
    
    # Test classification result
    test_classification = {
        "predicted_type": "government_order",
        "confidence": 0.95,
        "confidence_level": "high",
        "method": "content_based"
    }
    
    # Test temporal info
    test_temporal = {
        "dates": [
            {"type": "issue_date", "normalized_date": "2023-04-15"}
        ],
        "academic_years": [
            {"year_range": "2023-24", "start_year": 2023, "end_year": 2024}
        ]
    }
    
    # Test entities
    test_entities = {
        "go_refs": [
            {"number": 67, "year": 2023, "department": "MS", "date": "2023-04-15"}
        ],
        "schemes": ["Nadu-Nedu"],
        "districts": ["Visakhapatnam"]
    }
    
    # Build metadata
    metadata = builder.build_metadata(
        file_path=test_file,
        doc_id="test_go_67_2023",
        text="Sample GO text content...",
        classification_result=test_classification,
        temporal_info=test_temporal,
        entities=test_entities
    )
    
    print("Generated Metadata:")
    print(json.dumps(metadata, indent=2))