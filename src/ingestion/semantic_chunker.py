"""
SOTA Semantic Chunking for Policy Documents

This module implements intelligent document chunking that preserves semantic
units while optimizing for retrieval performance. Unlike fixed-size chunking,
this approach understands document structure and maintains context.

Key Features:
1. Legal-aware chunking (sections, articles, clauses)
2. Government Order structure preservation
3. Hierarchical context preservation
4. Adaptive chunk sizing based on content type
5. Cross-reference preservation
6. Metadata enrichment for better retrieval
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import spacy
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkType(Enum):
    """Types of semantic chunks"""
    LEGAL_SECTION = "legal_section"
    LEGAL_SUBSECTION = "legal_subsection"
    GO_PREAMBLE = "go_preamble"
    GO_ORDER = "go_order"
    GO_ANNEXURE = "go_annexure"
    JUDICIAL_FACT = "judicial_fact"
    JUDICIAL_HOLDING = "judicial_holding"
    DATA_TABLE = "data_table"
    DATA_SUMMARY = "data_summary"
    SCHEME_ELIGIBILITY = "scheme_eligibility"
    SCHEME_PROCEDURE = "scheme_procedure"
    GENERAL_PARAGRAPH = "general_paragraph"


@dataclass
class SemanticChunk:
    """Enhanced chunk with semantic metadata"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_type: ChunkType
    
    # Hierarchical structure
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    parent_section: Optional[str] = None
    subsection_number: Optional[str] = None
    
    # Context preservation
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None
    
    # Cross-references
    references: List[str] = None
    referenced_by: List[str] = None
    
    # Metadata
    word_count: int = 0
    sentence_count: int = 0
    importance_score: float = 0.0
    entity_mentions: List[str] = None
    
    # Position in document
    start_position: int = 0
    end_position: int = 0
    
    # Document metadata
    doc_type: str = ""
    doc_title: str = ""
    year: Optional[int] = None
    
    def __post_init__(self):
        if self.references is None:
            self.references = []
        if self.referenced_by is None:
            self.referenced_by = []
        if self.entity_mentions is None:
            self.entity_mentions = []
        
        # Calculate basic metrics
        if self.content:
            self.word_count = len(self.content.split())
            self.sentence_count = len([s for s in self.content.split('.') if s.strip()])


class SemanticChunker:
    """
    Advanced semantic chunker for policy documents.
    
    This chunker understands document structure and preserves semantic
    units while optimizing for retrieval performance.
    """
    
    # Legal document patterns
    LEGAL_PATTERNS = {
        'section': [
            r'^(\d+\.?\s*(?:\([a-z]\))?)\s*\.?\s*([^\n]+?)(?:\s*[-–—]\s*)?[\n:]',
            r'Section\s+(\d+(?:\([a-z]\))?)\s*\.?\s*([^\n]+)',
            r'Article\s+(\d+(?:[A-Z])?)\s*\.?\s*([^\n]+)',
            r'^(\d+\.)\s+([^\n]+)'
        ],
        'subsection': [
            r'^\s*\(([a-z]|i{1,3}|[0-9]+)\)\s*([^\n]+)',
            r'^\s*\(([0-9]+)\)\s*([^\n]+)',
            r'^\s*([a-z]\.)\s*([^\n]+)'
        ],
        'clause': [
            r'^\s*(i{1,3}\.?)\s*([^\n]+)',
            r'^\s*([a-z]{1,2}\.)\s*([^\n]+)'
        ]
    }
    
    # Government Order patterns
    GO_PATTERNS = {
        'go_header': [
            r'G\.?O\.?\s*(?:Ms\.?|Rt\.?)??\s*No\.?\s*(\d+)',
            r'Government\s+Order.*?No\.?\s*(\d+)',
            r'ORDER\s*No\.?\s*(\d+)'
        ],
        'preamble': [
            r'(?:WHEREAS|In exercise of|Having considered)',
            r'(?:Government.*?(?:pleased to|decided to|hereby))',
            r'(?:In.*?thereof|In.*?pursuance)'
        ],
        'order_directive': [
            r'(?:NOW\s+THEREFORE|Government.*?orders?|It is hereby ordered)',
            r'(?:The.*?(?:shall|will|is|are))',
            r'(?:All.*?(?:shall|will|must))'
        ],
        'supersession': [
            r'(?:supersedes?|in\s+supersession\s+of|replaces?)',
            r'(?:G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*\d+.*?(?:is|stands)\s+(?:superseded|cancelled))'
        ]
    }
    
    # Judicial patterns
    JUDICIAL_PATTERNS = {
        'case_citation': [
            r'(?:\d{4}\s+\(\d+\)\s+[A-Z]{2,5}\s+\d+)',
            r'(?:AIR\s+\d{4}\s+[A-Z]{2,5}\s+\d+)',
            r'(?:[A-Z]{2,5}\s+\d{4}\s+\([A-Z]+\)\s+\d+)'
        ],
        'holding': [
            r'(?:held|observed|ruled|decided|concluded)\s*[:.]',
            r'(?:It is|This Court|We)\s+(?:hold|observe|rule|decide)',
            r'(?:The.*?(?:held|observed|ruled))'
        ],
        'ratio': [
            r'(?:ratio|principle|law|rule)\s+(?:laid down|established|settled)',
            r'(?:legal principle|binding precedent)'
        ]
    }
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        """
        Initialize semantic chunker.
        
        Args:
            max_chunk_size: Maximum chunk size in characters
            overlap_size: Overlap between chunks for context preservation
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Try to load spaCy model for advanced NLP
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded for advanced semantic analysis")
        except OSError:
            logger.warning("SpaCy model not found. Using regex-based chunking only.")
        
        logger.info(f"Semantic chunker initialized: max_size={max_chunk_size}, overlap={overlap_size}")
    
    def chunk_document(
        self, 
        content: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[SemanticChunk]:
        """
        Main entry point for document chunking.
        
        Args:
            content: Document text content
            doc_metadata: Document metadata (type, title, year, etc.)
            
        Returns:
            List of semantic chunks
        """
        doc_type = doc_metadata.get('doc_type', '').lower()
        
        logger.info(f"Chunking document: type={doc_type}, length={len(content)}")
        
        # Route to specialized chunking based on document type
        if 'legal' in doc_type or 'act' in doc_type or 'rule' in doc_type:
            chunks = self._chunk_legal_document(content, doc_metadata)
        elif 'government_order' in doc_type or 'go' in doc_type:
            chunks = self._chunk_go_document(content, doc_metadata)
        elif 'judicial' in doc_type or 'judgment' in doc_type:
            chunks = self._chunk_judicial_document(content, doc_metadata)
        elif 'data' in doc_type or 'report' in doc_type:
            chunks = self._chunk_data_document(content, doc_metadata)
        else:
            chunks = self._chunk_general_document(content, doc_metadata)
        
        # Post-process chunks
        chunks = self._enrich_chunks(chunks, content, doc_metadata)
        chunks = self._optimize_chunk_sizes(chunks)
        chunks = self._extract_cross_references(chunks)
        
        logger.info(f"Generated {len(chunks)} semantic chunks")
        
        return chunks
    
    def _chunk_legal_document(
        self, 
        content: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[SemanticChunk]:
        """
        Chunk legal documents by preserving legal structure.
        
        Legal documents have hierarchical structure:
        - Articles/Sections
        - Sub-sections  
        - Clauses
        - Sub-clauses
        """
        chunks = []
        
        # Extract sections
        sections = self._extract_legal_sections(content)
        
        for section_data in sections:
            section_num = section_data['number']
            section_title = section_data['title']
            section_content = section_data['content']
            
            # If section is too large, break into subsections
            if len(section_content) > self.max_chunk_size:
                subsections = self._extract_subsections(section_content, section_num)
                
                for subsection in subsections:
                    chunk = SemanticChunk(
                        chunk_id=f"{doc_metadata['doc_id']}_sec_{section_num}_{subsection['number']}",
                        doc_id=doc_metadata['doc_id'],
                        content=f"Section {section_num}: {section_title}\n\n{subsection['content']}",
                        chunk_type=ChunkType.LEGAL_SUBSECTION,
                        section_number=section_num,
                        section_title=section_title,
                        subsection_number=subsection['number'],
                        **self._extract_doc_metadata(doc_metadata)
                    )
                    chunks.append(chunk)
            else:
                # Keep entire section as one chunk
                chunk = SemanticChunk(
                    chunk_id=f"{doc_metadata['doc_id']}_sec_{section_num}",
                    doc_id=doc_metadata['doc_id'],
                    content=f"Section {section_num}: {section_title}\n\n{section_content}",
                    chunk_type=ChunkType.LEGAL_SECTION,
                    section_number=section_num,
                    section_title=section_title,
                    **self._extract_doc_metadata(doc_metadata)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_go_document(
        self, 
        content: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[SemanticChunk]:
        """
        Chunk Government Orders by preserving GO structure.
        
        GO structure:
        - Header (GO number, date, department)
        - Preamble (background, legal basis)
        - Orders (specific directives)
        - Annexures (additional details)
        """
        chunks = []
        
        # Extract GO components
        go_data = self._parse_go_structure(content)
        
        # Preamble chunk (context)
        if go_data['preamble']:
            chunk = SemanticChunk(
                chunk_id=f"{doc_metadata['doc_id']}_preamble",
                doc_id=doc_metadata['doc_id'],
                content=f"{go_data['header']}\n\nPreamble:\n{go_data['preamble']}",
                chunk_type=ChunkType.GO_PREAMBLE,
                **self._extract_doc_metadata(doc_metadata)
            )
            chunks.append(chunk)
        
        # Order chunks (main directives)
        for i, order in enumerate(go_data['orders'], 1):
            # Include context from preamble
            context_snippet = go_data['preamble'][:200] + "..." if go_data['preamble'] else ""
            
            chunk = SemanticChunk(
                chunk_id=f"{doc_metadata['doc_id']}_order_{i}",
                doc_id=doc_metadata['doc_id'],
                content=f"Context: {context_snippet}\n\nOrder {i}: {order}",
                chunk_type=ChunkType.GO_ORDER,
                section_number=str(i),
                preceding_context=context_snippet,
                **self._extract_doc_metadata(doc_metadata)
            )
            chunks.append(chunk)
        
        # Annexure chunks
        for i, annexure in enumerate(go_data['annexures'], 1):
            chunk = SemanticChunk(
                chunk_id=f"{doc_metadata['doc_id']}_annexure_{i}",
                doc_id=doc_metadata['doc_id'],
                content=f"Annexure {i}:\n{annexure}",
                chunk_type=ChunkType.GO_ANNEXURE,
                section_number=f"annexure_{i}",
                **self._extract_doc_metadata(doc_metadata)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_judicial_document(
        self, 
        content: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[SemanticChunk]:
        """
        Chunk judicial documents preserving legal reasoning structure.
        """
        chunks = []
        
        # Extract key judicial components
        judicial_data = self._parse_judicial_structure(content)
        
        # Facts chunk
        if judicial_data['facts']:
            chunk = SemanticChunk(
                chunk_id=f"{doc_metadata['doc_id']}_facts",
                doc_id=doc_metadata['doc_id'],
                content=f"Facts:\n{judicial_data['facts']}",
                chunk_type=ChunkType.JUDICIAL_FACT,
                **self._extract_doc_metadata(doc_metadata)
            )
            chunks.append(chunk)
        
        # Holdings/Ratio chunks
        for i, holding in enumerate(judicial_data['holdings'], 1):
            chunk = SemanticChunk(
                chunk_id=f"{doc_metadata['doc_id']}_holding_{i}",
                doc_id=doc_metadata['doc_id'],
                content=f"Holding {i}:\n{holding}",
                chunk_type=ChunkType.JUDICIAL_HOLDING,
                section_number=str(i),
                **self._extract_doc_metadata(doc_metadata)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_data_document(
        self, 
        content: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[SemanticChunk]:
        """
        Chunk data/statistical documents preserving data relationships.
        """
        chunks = []
        
        # Extract tables and summaries
        data_components = self._parse_data_structure(content)
        
        # Summary chunks
        for i, summary in enumerate(data_components['summaries'], 1):
            chunk = SemanticChunk(
                chunk_id=f"{doc_metadata['doc_id']}_summary_{i}",
                doc_id=doc_metadata['doc_id'],
                content=summary,
                chunk_type=ChunkType.DATA_SUMMARY,
                section_number=str(i),
                **self._extract_doc_metadata(doc_metadata)
            )
            chunks.append(chunk)
        
        # Table chunks  
        for i, table in enumerate(data_components['tables'], 1):
            chunk = SemanticChunk(
                chunk_id=f"{doc_metadata['doc_id']}_table_{i}",
                doc_id=doc_metadata['doc_id'],
                content=f"Table {i}:\n{table}",
                chunk_type=ChunkType.DATA_TABLE,
                section_number=f"table_{i}",
                **self._extract_doc_metadata(doc_metadata)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_general_document(
        self, 
        content: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[SemanticChunk]:
        """
        Fallback chunking for documents that don't match specific types.
        Uses sentence-boundary aware chunking.
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_count = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk_count += 1
                chunk = SemanticChunk(
                    chunk_id=f"{doc_metadata['doc_id']}_para_{chunk_count}",
                    doc_id=doc_metadata['doc_id'],
                    content=current_chunk.strip(),
                    chunk_type=ChunkType.GENERAL_PARAGRAPH,
                    section_number=str(chunk_count),
                    **self._extract_doc_metadata(doc_metadata)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                current_chunk = current_chunk[-self.overlap_size:] + "\n\n" + paragraph
            else:
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
        
        # Final chunk
        if current_chunk.strip():
            chunk_count += 1
            chunk = SemanticChunk(
                chunk_id=f"{doc_metadata['doc_id']}_para_{chunk_count}",
                doc_id=doc_metadata['doc_id'],
                content=current_chunk.strip(),
                chunk_type=ChunkType.GENERAL_PARAGRAPH,
                section_number=str(chunk_count),
                **self._extract_doc_metadata(doc_metadata)
            )
            chunks.append(chunk)
        
        return chunks
    
    # ========== STRUCTURE EXTRACTION METHODS ==========
    
    def _extract_legal_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract legal sections with numbers and titles."""
        sections = []
        
        for pattern in self.LEGAL_PATTERNS['section']:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                section_num = match.group(1).strip()
                section_title = match.group(2).strip() if len(match.groups()) > 1 else ""
                
                # Find section content (until next section)
                start_pos = match.end()
                next_section_match = None
                
                for next_pattern in self.LEGAL_PATTERNS['section']:
                    next_match = re.search(next_pattern, content[start_pos:], re.MULTILINE | re.IGNORECASE)
                    if next_match:
                        if not next_section_match or next_match.start() < next_section_match.start():
                            next_section_match = next_match
                
                end_pos = start_pos + next_section_match.start() if next_section_match else len(content)
                section_content = content[start_pos:end_pos].strip()
                
                sections.append({
                    'number': section_num,
                    'title': section_title,
                    'content': section_content,
                    'start_pos': match.start(),
                    'end_pos': end_pos
                })
        
        # Sort by position and remove duplicates
        sections.sort(key=lambda x: x['start_pos'])
        
        return sections
    
    def _extract_subsections(self, content: str, parent_section: str) -> List[Dict[str, Any]]:
        """Extract subsections within a section."""
        subsections = []
        
        for pattern in self.LEGAL_PATTERNS['subsection']:
            matches = re.finditer(pattern, content, re.MULTILINE)
            
            for match in matches:
                subsection_num = match.group(1).strip()
                subsection_content = match.group(2).strip() if len(match.groups()) > 1 else ""
                
                # Find full subsection content
                start_pos = match.start()
                
                # Look for next subsection
                remaining_content = content[match.end():]
                next_match = re.search(pattern, remaining_content, re.MULTILINE)
                
                if next_match:
                    end_pos = match.end() + next_match.start()
                else:
                    end_pos = len(content)
                
                full_content = content[start_pos:end_pos].strip()
                
                subsections.append({
                    'number': subsection_num,
                    'content': full_content,
                    'parent_section': parent_section
                })
        
        return subsections
    
    def _parse_go_structure(self, content: str) -> Dict[str, Any]:
        """Parse Government Order structure."""
        go_data = {
            'header': '',
            'preamble': '',
            'orders': [],
            'annexures': []
        }
        
        lines = content.split('\n')
        current_section = 'header'
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section transitions
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.GO_PATTERNS['preamble']):
                if current_section == 'header':
                    go_data['header'] = '\n'.join(current_text)
                    current_text = []
                    current_section = 'preamble'
            
            elif any(re.search(pattern, line, re.IGNORECASE) for pattern in self.GO_PATTERNS['order_directive']):
                if current_section == 'preamble':
                    go_data['preamble'] = '\n'.join(current_text)
                    current_text = []
                    current_section = 'orders'
            
            elif re.search(r'annex|schedule|appendix', line, re.IGNORECASE):
                if current_section == 'orders':
                    go_data['orders'].append('\n'.join(current_text))
                    current_text = []
                    current_section = 'annexures'
            
            current_text.append(line)
        
        # Add final section
        if current_text:
            if current_section == 'preamble':
                go_data['preamble'] = '\n'.join(current_text)
            elif current_section == 'orders':
                go_data['orders'].append('\n'.join(current_text))
            elif current_section == 'annexures':
                go_data['annexures'].append('\n'.join(current_text))
        
        return go_data
    
    def _parse_judicial_structure(self, content: str) -> Dict[str, Any]:
        """Parse judicial document structure."""
        judicial_data = {
            'facts': '',
            'holdings': [],
            'ratio': []
        }
        
        # Extract holdings based on patterns
        for pattern in self.JUDICIAL_PATTERNS['holding']:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                # Extract sentence containing the holding
                sentence_start = content.rfind('.', 0, match.start()) + 1
                sentence_end = content.find('.', match.end())
                if sentence_end == -1:
                    sentence_end = len(content)
                
                holding = content[sentence_start:sentence_end].strip()
                if len(holding) > 50:  # Filter out short fragments
                    judicial_data['holdings'].append(holding)
        
        # Extract facts (simplified - first few paragraphs)
        paragraphs = content.split('\n\n')
        facts_paras = paragraphs[:3]  # First 3 paragraphs usually contain facts
        judicial_data['facts'] = '\n\n'.join(facts_paras)
        
        return judicial_data
    
    def _parse_data_structure(self, content: str) -> Dict[str, Any]:
        """Parse data document structure."""
        data_components = {
            'summaries': [],
            'tables': []
        }
        
        # Simple table detection (lines with multiple numbers/separators)
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Table detection heuristic
            if re.search(r'\d+.*\d+.*\d+', line) and ('|' in line or '\t' in line):
                # Found potential table row
                table_start = i
                table_end = i
                
                # Extend table
                for j in range(i + 1, min(len(lines), i + 20)):
                    if re.search(r'\d+.*\d+', lines[j]):
                        table_end = j
                    else:
                        break
                
                if table_end > table_start:
                    table_content = '\n'.join(lines[table_start:table_end + 1])
                    data_components['tables'].append(table_content)
        
        # Extract summaries (paragraphs with statistical terms)
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if any(term in para.lower() for term in ['percentage', 'ratio', 'statistics', 'data', 'enrollment', 'dropout']):
                data_components['summaries'].append(para)
        
        return data_components
    
    # ========== ENHANCEMENT METHODS ==========
    
    def _enrich_chunks(
        self, 
        chunks: List[SemanticChunk], 
        original_content: str, 
        doc_metadata: Dict[str, Any]
    ) -> List[SemanticChunk]:
        """Enrich chunks with additional metadata and context."""
        
        for chunk in chunks:
            # Extract entity mentions if spaCy is available
            if self.nlp:
                chunk.entity_mentions = self._extract_entities(chunk.content)
            
            # Calculate importance score
            chunk.importance_score = self._calculate_importance(chunk)
            
            # Add document position
            chunk.start_position = original_content.find(chunk.content[:100])
            chunk.end_position = chunk.start_position + len(chunk.content)
        
        return chunks
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities using spaCy."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'GPE', 'LAW', 'PERSON']:  # Relevant entity types
                entities.append(f"{ent.text}:{ent.label_}")
        
        return entities
    
    def _calculate_importance(self, chunk: SemanticChunk) -> float:
        """Calculate importance score based on content features."""
        score = 0.0
        content = chunk.content.lower()
        
        # Legal importance indicators
        legal_terms = ['shall', 'must', 'required', 'mandatory', 'prohibited', 'entitled']
        score += sum(1 for term in legal_terms if term in content) * 0.1
        
        # Specific document type importance
        if chunk.chunk_type in [ChunkType.LEGAL_SECTION, ChunkType.GO_ORDER]:
            score += 0.3
        elif chunk.chunk_type in [ChunkType.JUDICIAL_HOLDING, ChunkType.DATA_SUMMARY]:
            score += 0.2
        
        # Length-based scoring
        if 100 <= chunk.word_count <= 500:  # Optimal chunk size
            score += 0.2
        
        # Reference density
        reference_count = len(chunk.references) if chunk.references else 0
        score += min(reference_count * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _optimize_chunk_sizes(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Optimize chunk sizes by merging small chunks or splitting large ones."""
        optimized = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If chunk is too small, try to merge with next
            if current_chunk.word_count < 50 and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                # Only merge if same type and section
                if (current_chunk.chunk_type == next_chunk.chunk_type and 
                    current_chunk.section_number == next_chunk.section_number):
                    
                    merged_content = f"{current_chunk.content}\n\n{next_chunk.content}"
                    merged_chunk = SemanticChunk(
                        chunk_id=current_chunk.chunk_id,
                        doc_id=current_chunk.doc_id,
                        content=merged_content,
                        chunk_type=current_chunk.chunk_type,
                        section_number=current_chunk.section_number,
                        section_title=current_chunk.section_title,
                        **self._extract_doc_metadata({'doc_type': current_chunk.doc_type})
                    )
                    optimized.append(merged_chunk)
                    i += 2  # Skip next chunk
                    continue
            
            # If chunk is too large, split it
            elif current_chunk.word_count > self.max_chunk_size // 4:  # Word-based threshold
                split_chunks = self._split_large_chunk(current_chunk)
                optimized.extend(split_chunks)
            else:
                optimized.append(current_chunk)
            
            i += 1
        
        return optimized
    
    def _split_large_chunk(self, chunk: SemanticChunk) -> List[SemanticChunk]:
        """Split a large chunk into smaller ones while preserving context."""
        if len(chunk.content) <= self.max_chunk_size:
            return [chunk]
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', chunk.content)
        split_chunks = []
        current_content = ""
        part_num = 1
        
        for sentence in sentences:
            if len(current_content) + len(sentence) > self.max_chunk_size and current_content:
                # Create chunk
                split_chunk = SemanticChunk(
                    chunk_id=f"{chunk.chunk_id}_part_{part_num}",
                    doc_id=chunk.doc_id,
                    content=current_content.strip(),
                    chunk_type=chunk.chunk_type,
                    section_number=f"{chunk.section_number}_part_{part_num}",
                    section_title=chunk.section_title,
                    parent_section=chunk.section_number,
                    **self._extract_doc_metadata({'doc_type': chunk.doc_type})
                )
                split_chunks.append(split_chunk)
                
                # Start new chunk with overlap
                current_content = current_content[-self.overlap_size:] + " " + sentence
                part_num += 1
            else:
                current_content = current_content + " " + sentence if current_content else sentence
        
        # Final chunk
        if current_content.strip():
            split_chunk = SemanticChunk(
                chunk_id=f"{chunk.chunk_id}_part_{part_num}",
                doc_id=chunk.doc_id,
                content=current_content.strip(),
                chunk_type=chunk.chunk_type,
                section_number=f"{chunk.section_number}_part_{part_num}",
                section_title=chunk.section_title,
                parent_section=chunk.section_number,
                **self._extract_doc_metadata({'doc_type': chunk.doc_type})
            )
            split_chunks.append(split_chunk)
        
        return split_chunks
    
    def _extract_cross_references(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Extract cross-references between chunks."""
        
        # Build reference patterns
        reference_patterns = [
            r'Section\s+(\d+(?:\([a-z]\))?)',
            r'Article\s+(\d+)',
            r'Rule\s+(\d+)',
            r'G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
            r'Chapter\s+(\d+)',
            r'clause\s+\(([a-z])\)',
            r'sub-section\s+\((\d+)\)'
        ]
        
        # Extract references for each chunk
        for chunk in chunks:
            references = []
            
            for pattern in reference_patterns:
                matches = re.findall(pattern, chunk.content, re.IGNORECASE)
                references.extend(matches)
            
            chunk.references = list(set(references))  # Remove duplicates
        
        # Build reverse index (what references what)
        for chunk in chunks:
            for other_chunk in chunks:
                if chunk.chunk_id != other_chunk.chunk_id:
                    # Check if other_chunk references this chunk
                    if (chunk.section_number and 
                        chunk.section_number in other_chunk.references):
                        chunk.referenced_by.append(other_chunk.chunk_id)
        
        return chunks
    
    def _extract_doc_metadata(self, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract document-level metadata for chunks."""
        return {
            'doc_type': doc_metadata.get('doc_type', ''),
            'doc_title': doc_metadata.get('title', ''),
            'year': doc_metadata.get('year')
        }


# Utility functions for integration

def chunk_document_semantically(
    content: str,
    doc_metadata: Dict[str, Any],
    max_chunk_size: int = 1000,
    overlap_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Convenience function for semantic chunking.
    
    Returns chunks in the format expected by the existing pipeline.
    """
    chunker = SemanticChunker(max_chunk_size, overlap_size)
    semantic_chunks = chunker.chunk_document(content, doc_metadata)
    
    # Convert to existing format
    formatted_chunks = []
    for chunk in semantic_chunks:
        formatted_chunk = {
            'chunk_id': chunk.chunk_id,
            'doc_id': chunk.doc_id,
            'content': chunk.content,
            'metadata': {
                'chunk_type': chunk.chunk_type.value,
                'section_number': chunk.section_number,
                'section_title': chunk.section_title,
                'parent_section': chunk.parent_section,
                'subsection_number': chunk.subsection_number,
                'word_count': chunk.word_count,
                'sentence_count': chunk.sentence_count,
                'importance_score': chunk.importance_score,
                'entity_mentions': chunk.entity_mentions,
                'references': chunk.references,
                'referenced_by': chunk.referenced_by,
                'doc_type': chunk.doc_type,
                'doc_title': chunk.doc_title,
                'year': chunk.year
            }
        }
        formatted_chunks.append(formatted_chunk)
    
    return formatted_chunks


if __name__ == "__main__":
    print("Semantic Chunker module loaded successfully")
    
    # Test with sample content
    sample_content = """
    Section 12. Right of children to free and compulsory education.
    (1) Every child of the age of six to fourteen years shall have a right to free and compulsory elementary education in a neighbourhood school till completion of elementary education.
    (c) The appropriate Government may, subject to such conditions as may be prescribed, provide free elementary education to such child in a school specified under clause (n) of section 2.
    
    Section 13. Duties of the appropriate Government and local authority.
    The appropriate Government and the local authority shall ensure the establishment of a neighbourhood school.
    """
    
    chunker = SemanticChunker()
    test_metadata = {
        'doc_id': 'test_rte_act',
        'doc_type': 'legal_documents',
        'title': 'Right to Education Act 2009'
    }
    
    chunks = chunker.chunk_document(sample_content, test_metadata)
    print(f"Generated {len(chunks)} semantic chunks for test content")
    
    for chunk in chunks:
        print(f"- {chunk.chunk_type.value}: {chunk.section_number} ({chunk.word_count} words)")