#!/usr/bin/env python3
"""
Vertical-Specific Document Processor

This script processes documents from a specific vertical using specialized
extraction and enrichment strategies.

Usage:
    python process_vertical.py --vertical Legal
    python process_vertical.py --vertical Data_Reports --output-dir data/processed
    python process_vertical.py --all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.text_cleaner import TextCleaner
from src.ingestion.enhanced_metadata_builder import EnhancedMetadataBuilder
from src.ingestion.entity_extractor import EntityExtractor
from src.ingestion.chunker import SemanticChunker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

VERTICALS = {
    "Legal": {
        "priority": "CRITICAL",
        "processing_config": {
            "extract_sections": True,
            "track_amendments": True,
            "build_cross_references": True,
            "extract_definitions": True,
        },
        "entity_types": ["ACT", "SECTION", "RULE", "AMENDMENT", "AUTHORITY"]
    },
    "Government_Orders": {
        "priority": "HIGH",
        "processing_config": {
            "extract_go_number": True,
            "track_supersession": True,
            "extract_dates": True,
            "identify_authority": True,
        },
        "entity_types": ["GO_NUMBER", "DATE", "DEPARTMENT", "AUTHORITY", "BUDGET"]
    },
    "Judicial": {
        "priority": "HIGH",
        "processing_config": {
            "extract_case_citation": True,
            "extract_parties": True,
            "identify_precedents": True,
            "extract_ratio": True,
        },
        "entity_types": ["CASE_NUMBER", "COURT", "JUDGE", "PARTY", "PRECEDENT"]
    },
    "Data_Reports": {
        "priority": "HIGH",
        "processing_config": {
            "extract_tables": True,
            "parse_metrics": True,
            "extract_time_series": True,
            "extract_comparisons": True,
        },
        "entity_types": ["METRIC", "DATE", "VALUE", "DISTRICT", "SCHOOL_TYPE"]
    },
    "Schemes": {
        "priority": "HIGH",
        "processing_config": {
            "extract_scheme_details": True,
            "track_budget": True,
            "extract_beneficiaries": True,
            "parse_timeline": True,
        },
        "entity_types": ["SCHEME", "BUDGET", "BENEFICIARY", "DATE", "DISTRICT"]
    },
    "Teacher_Services": {
        "priority": "MEDIUM",
        "processing_config": {
            "extract_notification_details": True,
            "parse_eligibility": True,
            "extract_vacancies": True,
            "parse_dates": True,
        },
        "entity_types": ["POST", "QUALIFICATION", "DATE", "VACANCY", "DISTRICT"]
    },
    "Academic": {
        "priority": "MEDIUM",
        "processing_config": {
            "parse_calendar": True,
            "extract_curriculum": True,
            "parse_guidelines": True,
        },
        "entity_types": ["DATE", "SUBJECT", "GRADE", "ACTIVITY"]
    },
    "Policy": {
        "priority": "MEDIUM",
        "processing_config": {
            "extract_policy_framework": True,
            "extract_recommendations": True,
            "parse_research": True,
        },
        "entity_types": ["POLICY", "OBJECTIVE", "RECOMMENDATION", "INSTITUTION"]
    },
    "National": {
        "priority": "MEDIUM",
        "processing_config": {
            "extract_standards": True,
            "map_adoption": True,
        },
        "entity_types": ["STANDARD", "FRAMEWORK", "GUIDELINE"]
    },
    "Uncategorized": {
        "priority": "LOW",
        "processing_config": {
            "basic_extraction": True,
        },
        "entity_types": []
    }
}


class VerticalProcessor:
    """Process documents from a specific vertical"""
    
    def __init__(self, vertical_name: str, base_dir: str, output_dir: str):
        self.vertical_name = vertical_name
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.vertical_dir = self.base_dir / vertical_name
        
        if vertical_name not in VERTICALS:
            raise ValueError(f"Unknown vertical: {vertical_name}")
        
        self.config = VERTICALS[vertical_name]
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.text_cleaner = TextCleaner()
        self.metadata_builder = EnhancedMetadataBuilder()
        self.entity_extractor = EntityExtractor()
        self.chunker = SemanticChunker()
        
        # Create output directory
        self.vertical_output = self.output_dir / vertical_name
        self.vertical_output.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized processor for vertical: {vertical_name}")
    
    def get_documents(self) -> List[Path]:
        """Get all PDF documents in the vertical"""
        if not self.vertical_dir.exists():
            raise FileNotFoundError(f"Vertical directory not found: {self.vertical_dir}")
        
        documents = list(self.vertical_dir.glob("*.pdf"))
        logger.info(f"Found {len(documents)} documents in {self.vertical_name}")
        return documents
    
    def process_document(self, doc_path: Path) -> Dict:
        """Process a single document"""
        logger.info(f"Processing: {doc_path.name}")
        
        try:
            # Extract text from PDF
            raw_text, metadata = self.pdf_extractor.extract(str(doc_path))
            
            # Clean text
            cleaned_text = self.text_cleaner.clean(raw_text)
            
            # Build enhanced metadata
            enhanced_metadata = self.metadata_builder.build(
                text=cleaned_text,
                base_metadata=metadata,
                vertical=self.vertical_name
            )
            
            # Extract entities
            entities = self.entity_extractor.extract(
                cleaned_text,
                entity_types=self.config.get("entity_types", [])
            )
            
            # Apply vertical-specific processing
            vertical_data = self._apply_vertical_processing(
                cleaned_text, 
                enhanced_metadata,
                entities
            )
            
            # Create chunks
            chunks = self.chunker.chunk(
                cleaned_text,
                metadata=enhanced_metadata
            )
            
            # Compile result
            result = {
                "document_id": doc_path.stem,
                "filename": doc_path.name,
                "vertical": self.vertical_name,
                "metadata": enhanced_metadata,
                "entities": entities,
                "vertical_data": vertical_data,
                "chunks": chunks,
                "processed_at": datetime.now().isoformat(),
                "processing_config": self.config["processing_config"]
            }
            
            logger.info(f"✓ Successfully processed: {doc_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Error processing {doc_path.name}: {str(e)}")
            return {
                "document_id": doc_path.stem,
                "filename": doc_path.name,
                "vertical": self.vertical_name,
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def _apply_vertical_processing(
        self, 
        text: str, 
        metadata: Dict, 
        entities: List[Dict]
    ) -> Dict:
        """Apply vertical-specific processing logic"""
        
        vertical_data = {}
        config = self.config["processing_config"]
        
        if self.vertical_name == "Legal":
            vertical_data = self._process_legal(text, metadata, entities, config)
        elif self.vertical_name == "Government_Orders":
            vertical_data = self._process_government_orders(text, metadata, entities, config)
        elif self.vertical_name == "Judicial":
            vertical_data = self._process_judicial(text, metadata, entities, config)
        elif self.vertical_name == "Data_Reports":
            vertical_data = self._process_data_reports(text, metadata, entities, config)
        elif self.vertical_name == "Schemes":
            vertical_data = self._process_schemes(text, metadata, entities, config)
        # Add other verticals as needed
        
        return vertical_data
    
    def _process_legal(self, text, metadata, entities, config):
        """Process Legal documents"""
        result = {}
        
        if config.get("extract_sections"):
            # Extract section numbers and hierarchies
            import re
            sections = re.findall(r'Section\s+(\d+[A-Z]?)', text, re.IGNORECASE)
            result["sections"] = list(set(sections))
        
        if config.get("extract_definitions"):
            # Find defined terms
            definitions = re.findall(
                r'"([^"]+)"\s+means\s+([^.;]+)',
                text,
                re.IGNORECASE
            )
            result["definitions"] = [
                {"term": term, "definition": defn} 
                for term, defn in definitions
            ]
        
        if config.get("track_amendments"):
            # Find amendment references
            amendments = re.findall(
                r'amended?\s+by\s+([^.;]+)',
                text,
                re.IGNORECASE
            )
            result["amendments"] = amendments
        
        return result
    
    def _process_government_orders(self, text, metadata, entities, config):
        """Process Government Orders"""
        result = {}
        
        if config.get("extract_go_number"):
            # Extract GO numbers
            import re
            go_numbers = re.findall(
                r'G\.O\.(?:Ms|Rt)\.?\s*No\.?\s*(\d+)',
                text,
                re.IGNORECASE
            )
            result["go_numbers"] = list(set(go_numbers))
        
        if config.get("track_supersession"):
            # Find supersession references
            supersedes = re.findall(
                r'supersed(?:e|ing)\s+G\.O\.(?:Ms|Rt)\.?\s*No\.?\s*(\d+)',
                text,
                re.IGNORECASE
            )
            result["supersedes"] = supersedes
        
        return result
    
    def _process_judicial(self, text, metadata, entities, config):
        """Process Judicial documents"""
        result = {}
        
        if config.get("extract_case_citation"):
            # Extract case numbers
            import re
            case_numbers = re.findall(
                r'(?:W\.?P\.?|WA|SLP|CIVIL\s+APPEAL)\s+(?:No\.?)?\s*(\d+/\d{4})',
                text,
                re.IGNORECASE
            )
            result["case_numbers"] = list(set(case_numbers))
        
        if config.get("extract_parties"):
            # Extract petitioner and respondent (simplified)
            petitioner = re.search(r'PETITIONER[:\s]+([^\n]+)', text, re.IGNORECASE)
            respondent = re.search(r'RESPONDENT[:\s]+([^\n]+)', text, re.IGNORECASE)
            
            result["parties"] = {
                "petitioner": petitioner.group(1).strip() if petitioner else None,
                "respondent": respondent.group(1).strip() if respondent else None
            }
        
        return result
    
    def _process_data_reports(self, text, metadata, entities, config):
        """Process Data Reports"""
        result = {}
        
        if config.get("extract_tables"):
            # Placeholder for table extraction
            result["tables_detected"] = "Table extraction requires specialized library"
        
        if config.get("parse_metrics"):
            # Extract numeric metrics (simplified)
            import re
            metrics = re.findall(
                r'(\w+(?:\s+\w+)*)\s*:\s*([0-9,]+(?:\.[0-9]+)?)',
                text
            )
            result["metrics"] = [
                {"name": name.strip(), "value": value} 
                for name, value in metrics[:20]  # Limit to first 20
            ]
        
        return result
    
    def _process_schemes(self, text, metadata, entities, config):
        """Process Schemes"""
        result = {}
        
        if config.get("extract_scheme_details"):
            # Extract scheme name
            import re
            scheme_match = re.search(
                r'(?:scheme|programme|program)[:\s]+([^\n.]+)',
                text,
                re.IGNORECASE
            )
            if scheme_match:
                result["scheme_name"] = scheme_match.group(1).strip()
        
        if config.get("track_budget"):
            # Extract budget amounts
            budget_matches = re.findall(
                r'(?:Rs\.?|INR|₹)\s*([0-9,]+(?:\.[0-9]+)?)\s*(crore|lakh|thousand)?',
                text,
                re.IGNORECASE
            )
            result["budget_allocations"] = [
                {"amount": amt, "unit": unit} 
                for amt, unit in budget_matches[:10]
            ]
        
        return result
    
    def process_all(self) -> Dict:
        """Process all documents in the vertical"""
        documents = self.get_documents()
        
        if not documents:
            logger.warning(f"No documents found in {self.vertical_name}")
            return {"vertical": self.vertical_name, "processed": 0, "results": []}
        
        logger.info(f"Starting batch processing of {len(documents)} documents")
        
        results = []
        successful = 0
        failed = 0
        
        for doc_path in documents:
            result = self.process_document(doc_path)
            results.append(result)
            
            if "error" in result:
                failed += 1
            else:
                successful += 1
                
                # Save individual result
                output_file = self.vertical_output / f"{doc_path.stem}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
        
        # Save batch summary
        summary = {
            "vertical": self.vertical_name,
            "priority": self.config["priority"],
            "total_documents": len(documents),
            "successful": successful,
            "failed": failed,
            "processing_config": self.config["processing_config"],
            "processed_at": datetime.now().isoformat(),
            "results": results
        }
        
        summary_file = self.vertical_output / "_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        logger.info(f"Results saved to: {self.vertical_output}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Process documents from organized verticals"
    )
    parser.add_argument(
        "--vertical",
        type=str,
        help="Name of the vertical to process (e.g., Legal, Judicial)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all verticals in priority order"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/organized_documents",
        help="Base directory containing organized documents"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_verticals",
        help="Output directory for processed results"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available verticals"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Verticals:\n")
        for name, config in sorted(VERTICALS.items(), key=lambda x: x[1]["priority"]):
            print(f"  {name:20s} - {config['priority']:10s}")
        print()
        return
    
    if not args.vertical and not args.all:
        parser.error("Either --vertical or --all must be specified")
    
    if args.all:
        # Process all verticals in priority order
        priority_order = {
            "CRITICAL": 1,
            "HIGH": 2,
            "MEDIUM": 3,
            "LOW": 4
        }
        
        verticals_sorted = sorted(
            VERTICALS.keys(),
            key=lambda v: (priority_order[VERTICALS[v]["priority"]], v)
        )
        
        logger.info(f"Processing all verticals in priority order: {verticals_sorted}")
        
        for vertical in verticals_sorted:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Vertical: {vertical}")
            logger.info(f"{'='*80}\n")
            
            try:
                processor = VerticalProcessor(vertical, args.base_dir, args.output_dir)
                summary = processor.process_all()
                
                print(f"\n{vertical}: {summary['successful']}/{summary['total_documents']} processed successfully")
            except Exception as e:
                logger.error(f"Failed to process vertical {vertical}: {e}")
    
    else:
        # Process single vertical
        processor = VerticalProcessor(args.vertical, args.base_dir, args.output_dir)
        summary = processor.process_all()
        
        print("\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)
        print(f"Vertical: {summary['vertical']}")
        print(f"Priority: {summary['priority']}")
        print(f"Total Documents: {summary['total_documents']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Output Directory: {args.output_dir}/{args.vertical}")
        print("="*80)


if __name__ == "__main__":
    main()

