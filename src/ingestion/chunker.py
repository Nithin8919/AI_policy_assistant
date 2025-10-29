"""
Smart text chunking with overlap and context preservation.
"""

import re
from typing import List, Dict, Optional
import hashlib

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SmartChunker:
    """
    Intelligent text chunking that preserves context and meaning.
    
    Strategies:
    - Respect paragraph boundaries
    - Add overlap between chunks for context
    - Preserve section metadata
    - Keep chunks within token limits
    """
    
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"Initialized chunker: size={chunk_size}, overlap={chunk_overlap}")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of tokens (words * 1.3 for subword tokens).
        """
        words = len(text.split())
        return int(words * 1.3)
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Handles common abbreviations (e.g., Dr., Mr., etc.)
        """
        # Simple sentence splitter
        # This is a basic implementation - could be enhanced with spaCy
        
        # Protect common abbreviations
        text = text.replace('Dr.', 'Dr<dot>')
        text = text.replace('Mr.', 'Mr<dot>')
        text = text.replace('Mrs.', 'Mrs<dot>')
        text = text.replace('Ms.', 'Ms<dot>')
        text = text.replace('No.', 'No<dot>')
        text = text.replace('Art.', 'Art<dot>')
        text = text.replace('Sec.', 'Sec<dot>')
        
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('<dot>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def create_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        return f"{doc_id}_chunk_{chunk_index:04d}"
    
    def chunk_section(
        self,
        section_text: str,
        doc_id: str,
        section_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk a document section into overlapping chunks.
        
        Args:
            section_text: Text of the section
            doc_id: Document ID
            section_id: Section identifier
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries
        """
        if not section_text or not section_text.strip():
            return []
        
        chunks = []
        
        # Split into paragraphs first
        paragraphs = self.split_by_paragraphs(section_text)
        
        if not paragraphs:
            return []
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If single paragraph exceeds chunk size, split it by sentences
            if para_length > self.chunk_size * 1.5:
                sentences = self.split_by_sentences(para)
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    
                    if current_length + sentence_length > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(self._create_chunk_dict(
                            chunk_text=chunk_text,
                            doc_id=doc_id,
                            section_id=section_id,
                            chunk_index=chunk_index,
                            metadata=metadata
                        ))
                        
                        # Start new chunk with overlap
                        overlap_text = ' '.join(current_chunk[-2:])  # Last 2 sentences
                        current_chunk = [overlap_text, sentence] if len(overlap_text) < self.chunk_overlap else [sentence]
                        current_length = len(' '.join(current_chunk))
                        chunk_index += 1
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
            
            # Normal paragraph handling
            elif current_length + para_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk_dict(
                    chunk_text=chunk_text,
                    doc_id=doc_id,
                    section_id=section_id,
                    chunk_index=chunk_index,
                    metadata=metadata
                ))
                
                # Start new chunk with overlap
                # Use last paragraph or portion of it as overlap
                overlap_text = current_chunk[-1][:self.chunk_overlap]
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_length = len('\n\n'.join(current_chunk))
                chunk_index += 1
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk_dict(
                chunk_text=chunk_text,
                doc_id=doc_id,
                section_id=section_id,
                chunk_index=chunk_index,
                metadata=metadata
            ))
        
        logger.debug(f"Created {len(chunks)} chunks from section {section_id}")
        
        return chunks
    
    def _create_chunk_dict(
        self,
        chunk_text: str,
        doc_id: str,
        section_id: Optional[str],
        chunk_index: int,
        metadata: Optional[Dict]
    ) -> Dict:
        """
        Create standardized chunk dictionary.
        
        Args:
            chunk_text: The actual chunk text
            doc_id: Document ID
            section_id: Section identifier
            chunk_index: Index of this chunk
            metadata: Document metadata
            
        Returns:
            Chunk dictionary ready for embedding
        """
        chunk_id = self.create_chunk_id(doc_id, chunk_index)
        
        # Calculate content hash for deduplication
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()
        
        chunk_dict = {
            'chunk_id': chunk_id,
            'doc_id': doc_id,
            'section_id': section_id,
            'chunk_index': chunk_index,
            'text': chunk_text,
            'char_count': len(chunk_text),
            'word_count': len(chunk_text.split()),
            'estimated_tokens': self.estimate_tokens(chunk_text),
            'content_hash': content_hash,
            'metadata': {
                'doc_type': metadata.get('doc_type') if metadata else None,
                'title': metadata.get('title') if metadata else None,
                'year': metadata.get('year') if metadata else None,
                'priority': metadata.get('priority') if metadata else None,
                'file_format': metadata.get('file_format') if metadata else None,
                'parent_folders': metadata.get('parent_folders') if metadata else [],
            },
            # Placeholder for entities (will be populated by pipeline)
            'entities': {},
            # Placeholder for bridge topics (will be populated by topic matcher)
            'bridge_topics': [],
            # Placeholder for temporal info
            'temporal': {},
            # Quality score (will be set during quality check)
            'quality_score': None
        }
        
        return chunk_dict
    
    def chunk_text_simple(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Simple chunking without section information.
        
        Args:
            text: Full text to chunk
            doc_id: Document ID
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries
        """
        return self.chunk_section(
            section_text=text,
            doc_id=doc_id,
            section_id='full_text',
            metadata=metadata
        )