"""
Parse documents into logical sections based on document type.
"""

import re
from typing import List, Dict, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SectionParser:
    """Parse documents into logical sections."""
    
    def __init__(self):
        """Initialize section parser."""
        pass
    
    def parse_act(self, text: str) -> List[Dict]:
        """
        Parse legislative acts into sections.
        
        Pattern: "Section 1", "Section 2", etc.
        """
        sections = []
        
        # Split by section headers
        section_pattern = r'(Section\s+\d+[A-Z]?\.?\s*[-:]?\s*[^\n]*)'
        matches = list(re.finditer(section_pattern, text, re.IGNORECASE))
        
        if not matches:
            # No sections found, return whole document as single section
            return [{
                'section_id': 'full_text',
                'section_title': 'Full Document',
                'text': text,
                'char_count': len(text)
            }]
        
        for i, match in enumerate(matches):
            section_header = match.group(1).strip()
            section_start = match.end()
            
            # Find end of this section (start of next section or end of document)
            if i < len(matches) - 1:
                section_end = matches[i + 1].start()
            else:
                section_end = len(text)
            
            section_text = text[section_start:section_end].strip()
            
            # Extract section number
            section_num_match = re.search(r'Section\s+(\d+[A-Z]?)', section_header, re.IGNORECASE)
            section_id = section_num_match.group(1) if section_num_match else f'section_{i+1}'
            
            sections.append({
                'section_id': f'sec_{section_id}',
                'section_title': section_header,
                'text': section_text,
                'char_count': len(section_text)
            })
        
        logger.debug(f"Parsed act into {len(sections)} sections")
        return sections
    
    def parse_go(self, text: str) -> List[Dict]:
        """
        Parse Government Orders into sections.
        
        Typical structure:
        - Preamble
        - Orders/Directions
        - Signature block
        """
        sections = []
        
        # Try to identify main sections
        # Pattern 1: Numbered paragraphs (1., 2., 3., etc.)
        numbered_pattern = r'^\s*(\d+)\.\s+([^\n]+)'
        matches = list(re.finditer(numbered_pattern, text, re.MULTILINE))
        
        if len(matches) >= 2:
            # Document has numbered sections
            for i, match in enumerate(matches):
                section_num = match.group(1)
                section_title = match.group(2).strip()
                section_start = match.end()
                
                if i < len(matches) - 1:
                    section_end = matches[i + 1].start()
                else:
                    section_end = len(text)
                
                section_text = text[section_start:section_end].strip()
                
                sections.append({
                    'section_id': f'para_{section_num}',
                    'section_title': section_title[:100],  # Limit title length
                    'text': section_text,
                    'char_count': len(section_text)
                })
        else:
            # No clear sections, split by major headings or return as single section
            heading_pattern = r'^([A-Z][A-Z\s]{10,}):?\s*$'
            matches = list(re.finditer(heading_pattern, text, re.MULTILINE))
            
            if matches:
                for i, match in enumerate(matches):
                    heading = match.group(1).strip()
                    section_start = match.end()
                    
                    if i < len(matches) - 1:
                        section_end = matches[i + 1].start()
                    else:
                        section_end = len(text)
                    
                    section_text = text[section_start:section_end].strip()
                    
                    sections.append({
                        'section_id': f'heading_{i+1}',
                        'section_title': heading,
                        'text': section_text,
                        'char_count': len(section_text)
                    })
            else:
                # Return as single section
                sections.append({
                    'section_id': 'full_text',
                    'section_title': 'Government Order',
                    'text': text,
                    'char_count': len(text)
                })
        
        logger.debug(f"Parsed GO into {len(sections)} sections")
        return sections
    
    def parse_judicial(self, text: str) -> List[Dict]:
        """
        Parse judicial documents (court cases) into sections.
        
        Typical structure:
        - Facts
        - Arguments
        - Judgment/Order
        - Conclusion
        """
        sections = []
        
        # Common judicial section headers
        judicial_headers = [
            'FACTS',
            'SUBMISSIONS',
            'ARGUMENTS',
            'CONTENTIONS',
            'JUDGMENT',
            'ORDER',
            'HELD',
            'CONCLUSION',
            'DIRECTIONS'
        ]
        
        header_pattern = r'^\s*(' + '|'.join(judicial_headers) + r')\s*:?\s*$'
        matches = list(re.finditer(header_pattern, text, re.MULTILINE | re.IGNORECASE))
        
        if matches:
            for i, match in enumerate(matches):
                header = match.group(1).strip().upper()
                section_start = match.end()
                
                if i < len(matches) - 1:
                    section_end = matches[i + 1].start()
                else:
                    section_end = len(text)
                
                section_text = text[section_start:section_end].strip()
                
                sections.append({
                    'section_id': f'{header.lower()}',
                    'section_title': header,
                    'text': section_text,
                    'char_count': len(section_text)
                })
        else:
            # No clear sections, return as single section
            sections.append({
                'section_id': 'full_judgment',
                'section_title': 'Full Judgment',
                'text': text,
                'char_count': len(text)
            })
        
        logger.debug(f"Parsed judicial document into {len(sections)} sections")
        return sections
    
    def parse_data_report(self, text: str) -> List[Dict]:
        """
        Parse data reports into sections.
        
        Typically has chapters or numbered sections.
        """
        sections = []
        
        # Pattern: Chapter/Chapter headings
        chapter_pattern = r'(Chapter\s+\d+\s*[-:]?\s*[^\n]+)'
        matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))
        
        if matches:
            for i, match in enumerate(matches):
                chapter_header = match.group(1).strip()
                section_start = match.end()
                
                if i < len(matches) - 1:
                    section_end = matches[i + 1].start()
                else:
                    section_end = len(text)
                
                section_text = text[section_start:section_end].strip()
                
                # Extract chapter number
                chapter_num_match = re.search(r'Chapter\s+(\d+)', chapter_header, re.IGNORECASE)
                chapter_id = chapter_num_match.group(1) if chapter_num_match else f'ch_{i+1}'
                
                sections.append({
                    'section_id': f'chapter_{chapter_id}',
                    'section_title': chapter_header,
                    'text': section_text,
                    'char_count': len(section_text)
                })
        else:
            # Try numbered sections (1., 2., etc.)
            return self.parse_go(text)  # Reuse GO parsing logic
        
        logger.debug(f"Parsed data report into {len(sections)} sections")
        return sections
    
    def parse_generic(self, text: str, max_section_size: int = 5000) -> List[Dict]:
        """
        Generic parser for documents without clear structure.
        
        Splits by paragraphs or fixed size if too long.
        """
        sections = []
        
        # If document is short, return as single section
        if len(text) <= max_section_size:
            return [{
                'section_id': 'full_text',
                'section_title': 'Full Document',
                'text': text,
                'char_count': len(text)
            }]
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_section = []
        current_length = 0
        section_num = 1
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            if current_length + para_length > max_section_size and current_section:
                # Save current section
                section_text = '\n\n'.join(current_section)
                sections.append({
                    'section_id': f'section_{section_num}',
                    'section_title': f'Section {section_num}',
                    'text': section_text,
                    'char_count': len(section_text)
                })
                
                # Start new section
                current_section = [para]
                current_length = para_length
                section_num += 1
            else:
                current_section.append(para)
                current_length += para_length
        
        # Add final section
        if current_section:
            section_text = '\n\n'.join(current_section)
            sections.append({
                'section_id': f'section_{section_num}',
                'section_title': f'Section {section_num}',
                'text': section_text,
                'char_count': len(section_text)
            })
        
        logger.debug(f"Parsed generic document into {len(sections)} sections")
        return sections
    
    def parse(self, text: str, doc_type: str) -> List[Dict]:
        """
        Parse text into sections based on document type.
        
        Args:
            text: Cleaned document text
            doc_type: Document type (act, rule, government_order, etc.)
            
        Returns:
            List of section dictionaries
        """
        if not text or not text.strip():
            return []
        
        logger.debug(f"Parsing document of type: {doc_type}")
        
        # Route to appropriate parser
        if doc_type == 'act':
            sections = self.parse_act(text)
        elif doc_type == 'rule':
            sections = self.parse_act(text)  # Rules similar to acts
        elif doc_type == 'government_order':
            sections = self.parse_go(text)
        elif doc_type == 'judicial':
            sections = self.parse_judicial(text)
        elif doc_type == 'data_report':
            sections = self.parse_data_report(text)
        else:
            sections = self.parse_generic(text)
        
        # Filter out empty sections
        sections = [s for s in sections if s['text'].strip()]
        
        return sections