"""
PDF text extraction with OCR fallback.
Handles both text-based and scanned PDFs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import PyPDF2
import pdfplumber
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.file_utils import write_json

logger = get_logger(__name__)


class PDFExtractor:
    """Extract text from PDF files with multiple strategies."""
    
    def __init__(self):
        self.ocr_available = self._check_ocr_availability()
    
    def _check_ocr_availability(self) -> bool:
        """Check if pytesseract is available for OCR."""
        try:
            import pytesseract
            # Test if tesseract is installed
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"OCR not available: {e}")
            return False
    
    def extract_text_pypdf2(self, pdf_path: Path) -> Tuple[str, Dict]:
        """
        Extract text using PyPDF2 (fast, for text-based PDFs).
        
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                metadata = {
                    "num_pages": len(pdf_reader.pages),
                    "method": "pypdf2",
                    "pdf_metadata": {}
                }
                
                # Extract PDF metadata
                if pdf_reader.metadata:
                    metadata["pdf_metadata"] = {
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "subject": pdf_reader.metadata.get('/Subject', ''),
                        "creator": pdf_reader.metadata.get('/Creator', ''),
                    }
                
                # Extract text from all pages
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                    except Exception as e:
                        logger.debug(f"Error extracting page {page_num}: {e}")
                        continue
                
                full_text = "\n".join(text_parts)
                
                return full_text, metadata
                
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return "", {}
    
    def extract_text_pdfplumber(self, pdf_path: Path) -> Tuple[str, Dict]:
        """
        Extract text using pdfplumber (better for complex layouts).
        
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = {
                    "num_pages": len(pdf.pages),
                    "method": "pdfplumber",
                    "has_tables": False
                }
                
                text_parts = []
                tables_found = 0
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                        
                        # Check for tables
                        tables = page.extract_tables()
                        if tables:
                            tables_found += len(tables)
                            metadata["has_tables"] = True
                            
                            # Convert tables to text representation
                            for table_num, table in enumerate(tables, 1):
                                text_parts.append(f"\n[Table {table_num} on Page {page_num}]")
                                for row in table:
                                    if row:
                                        text_parts.append(" | ".join([str(cell) if cell else "" for cell in row]))
                        
                    except Exception as e:
                        logger.debug(f"Error extracting page {page_num}: {e}")
                        continue
                
                metadata["tables_found"] = tables_found
                full_text = "\n".join(text_parts)
                
                return full_text, metadata
                
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return "", {}
    
    def extract_text_ocr(self, pdf_path: Path) -> Tuple[str, Dict]:
        """
        Extract text using OCR for scanned PDFs.
        
        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not self.ocr_available:
            logger.warning("OCR not available, skipping OCR extraction")
            return "", {"method": "ocr", "status": "unavailable"}
        
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            logger.info(f"Converting {pdf_path.name} to images for OCR...")
            images = convert_from_path(pdf_path, dpi=300)
            
            metadata = {
                "num_pages": len(images),
                "method": "ocr",
                "status": "success"
            }
            
            text_parts = []
            for page_num, image in enumerate(images, 1):
                try:
                    logger.debug(f"OCR processing page {page_num}/{len(images)}")
                    page_text = pytesseract.image_to_string(image)
                    if page_text and page_text.strip():
                        text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                except Exception as e:
                    logger.debug(f"Error in OCR for page {page_num}: {e}")
                    continue
            
            full_text = "\n".join(text_parts)
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return "", {"method": "ocr", "status": "failed", "error": str(e)}
    
    def extract(self, pdf_path: Path, prefer_ocr: bool = False) -> Dict:
        """
        Extract text from PDF with automatic fallback.
        
        Args:
            pdf_path: Path to PDF file
            prefer_ocr: If True, try OCR first
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Extracting text from: {pdf_path.name}")
        
        result = {
            "file_path": str(pdf_path),
            "filename": pdf_path.name,
            "text": "",
            "metadata": {},
            "extraction_method": None,
            "success": False,
            "word_count": 0,
            "char_count": 0
        }
        
        # Try pdfplumber first (best for most PDFs)
        if not prefer_ocr:
            text, metadata = self.extract_text_pdfplumber(pdf_path)
            
            # Check if extraction was successful
            if text and len(text.strip()) > 100:
                result["text"] = text
                result["metadata"] = metadata
                result["extraction_method"] = "pdfplumber"
                result["success"] = True
                logger.info(f"Successfully extracted using pdfplumber: {len(text)} characters")
            else:
                # Fallback to PyPDF2
                logger.info("pdfplumber extraction insufficient, trying PyPDF2...")
                text, metadata = self.extract_text_pypdf2(pdf_path)
                
                if text and len(text.strip()) > 100:
                    result["text"] = text
                    result["metadata"] = metadata
                    result["extraction_method"] = "pypdf2"
                    result["success"] = True
                    logger.info(f"Successfully extracted using PyPDF2: {len(text)} characters")
        
        # If still no text, try OCR (if available)
        if not result["success"] and self.ocr_available:
            logger.info("Standard extraction failed, attempting OCR...")
            text, metadata = self.extract_text_ocr(pdf_path)
            
            if text and len(text.strip()) > 100:
                result["text"] = text
                result["metadata"] = metadata
                result["extraction_method"] = "ocr"
                result["success"] = True
                logger.info(f"Successfully extracted using OCR: {len(text)} characters")
        
        # Calculate statistics
        if result["text"]:
            result["char_count"] = len(result["text"])
            result["word_count"] = len(result["text"].split())
        
        if not result["success"]:
            logger.warning(f"Failed to extract meaningful text from {pdf_path.name}")
        
        return result


def extract_pdf_batch(pdf_paths: List[Path], output_dir: Path) -> List[Dict]:
    """
    Batch extract text from multiple PDFs.
    
    Args:
        pdf_paths: List of PDF file paths
        output_dir: Directory to save extracted text
        
    Returns:
        List of extraction results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = PDFExtractor()
    results = []
    
    total = len(pdf_paths)
    for idx, pdf_path in enumerate(pdf_paths, 1):
        logger.info(f"Processing PDF {idx}/{total}: {pdf_path.name}")
        
        try:
            result = extractor.extract(pdf_path)
            results.append(result)
            
            # Save extracted text
            if result["success"]:
                output_file = output_dir / f"{pdf_path.stem}.json"
                write_json(result, output_file)
                logger.info(f"Saved extracted text to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            results.append({
                "file_path": str(pdf_path),
                "filename": pdf_path.name,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r.get("success"))
    logger.info(f"Extraction complete: {successful}/{total} PDFs successful")
    
    return results