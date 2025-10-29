"""
Text cleaning and normalization for extracted documents.
"""

import re
from typing import Dict, List, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Clean and normalize extracted text."""
    
    def __init__(self):
        """Initialize text cleaner."""
        pass
    
    def remove_page_headers_footers(self, text: str) -> str:
        """
        Remove common page headers and footers.
        
        Patterns:
        - Page numbers: "Page 1", "- 1 -", etc.
        - Headers: repeated text at top of pages
        """
        # Remove standalone page numbers
        text = re.sub(r'\n\s*[-–—]\s*\d+\s*[-–—]\s*\n', '\n', text)
        text = re.sub(r'\nPage \d+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\d+\s*of\s*\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace while preserving paragraph structure.
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove spaces at beginning/end of lines
        text = re.sub(r'^ +', '', text, flags=re.MULTILINE)
        text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def fix_hyphenation(self, text: str) -> str:
        """
        Fix words broken by hyphens at line endings.
        
        Example: "educa-\ntion" -> "education"
        """
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters.
        """
        # Replace various quote marks with standard ones
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Replace various dashes with standard hyphen/dash
        text = text.replace('–', '-').replace('—', '-')
        
        # Replace ellipsis
        text = text.replace('…', '...')
        
        return text
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters while keeping meaningful punctuation.
        """
        if keep_punctuation:
            # Remove only truly special characters, keep standard punctuation
            # Keep: letters, numbers, spaces, common punctuation
            text = re.sub(r'[^\w\s.,;:!?()\[\]{}"\'/-]', '', text)
        else:
            # Remove all non-alphanumeric except spaces
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in government documents.
        """
        # Common OCR mistakes
        replacements = {
            r'\bl\b': 'I',  # lowercase L mistaken for I
            r'\bO\b': '0',  # O mistaken for zero in numbers
            r'rn': 'm',     # rn mistaken for m in some cases
            r'\bGO\s+([A-Z])': r'GO \1',  # Fix spacing in GO references
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def standardize_go_references(self, text: str) -> str:
        """
        Standardize Government Order reference formats.
        
        Variations: G.O.Ms.No., GO MS No., G-O-Ms-No, etc.
        Standard: GO MS No.
        """
        # Standardize GO references
        patterns = [
            (r'G\.O\.Ms\.No\.?\s*', 'GO MS No. '),
            (r'G-O-Ms-No\.?\s*', 'GO MS No. '),
            (r'GO\s+MS\s+No\.?\s*', 'GO MS No. '),
            (r'G\.O\.\s*\(Ms\)\s*No\.?\s*', 'GO MS No. '),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def standardize_dates(self, text: str) -> str:
        """
        Standardize date formats to DD/MM/YYYY.
        
        Handles: DD-MM-YYYY, DD.MM.YYYY, DD MM YYYY
        """
        # Pattern: DD-MM-YYYY or DD.MM.YYYY -> DD/MM/YYYY
        text = re.sub(
            r'\b(\d{1,2})[-.](\d{1,2})[-.](\d{4})\b',
            r'\1/\2/\3',
            text
        )
        
        return text
    
    def standardize_section_references(self, text: str) -> str:
        """
        Standardize section/clause references.
        
        Variations: Section-12, Sec 12, § 12
        Standard: Section 12
        """
        # Standardize section references
        text = re.sub(r'Sec\.?\s+', 'Section ', text, flags=re.IGNORECASE)
        text = re.sub(r'Section-', 'Section ', text)
        text = re.sub(r'§\s*', 'Section ', text)
        
        return text
    
    def remove_artifacts(self, text: str) -> str:
        """
        Remove PDF extraction artifacts.
        """
        # Remove common artifacts
        artifacts = [
            r'\[Image: .*?\]',
            r'\[Chart: .*?\]',
            r'\[Graph: .*?\]',
            r'<.*?>',  # HTML tags if any
        ]
        
        for pattern in artifacts:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def clean(self, text: str, aggressive: bool = False) -> str:
        """
        Apply all cleaning steps to text.
        
        Args:
            text: Raw extracted text
            aggressive: If True, apply more aggressive cleaning
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        logger.debug(f"Cleaning text (length: {len(text)} chars)")
        
        # Step 1: Fix OCR errors
        text = self.fix_common_ocr_errors(text)
        
        # Step 2: Fix hyphenation
        text = self.fix_hyphenation(text)
        
        # Step 3: Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Step 4: Remove page headers/footers
        text = self.remove_page_headers_footers(text)
        
        # Step 5: Standardize references
        text = self.standardize_go_references(text)
        text = self.standardize_dates(text)
        text = self.standardize_section_references(text)
        
        # Step 6: Remove artifacts
        text = self.remove_artifacts(text)
        
        # Step 7: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 8: Optional aggressive cleaning
        if aggressive:
            text = self.remove_special_characters(text, keep_punctuation=False)
        
        logger.debug(f"Cleaned text (length: {len(text)} chars)")
        
        return text
