"""
Enhanced Relation Extractor with Improved Pattern Matching.

Extracts semantic relationships between entities using:
- Improved regex patterns with context
- Dependency parsing for verb-object relations
- Multi-strategy approach (pattern + spaCy + heuristics)
- Better handling of Indian legal/government text
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy --break-system-packages")

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Relation:
    """Represents a relation between two entities."""
    relation_type: str
    source: str
    target: str
    source_chunk_id: str
    confidence: float = 1.0
    context: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class RelationExtractor:
    """
    Extract semantic relationships from text.
    
    Key Improvements:
    - Better regex patterns with word boundaries
    - Dependency parsing for implicit relations
    - Context-aware extraction
    - Handles Indian legal language patterns
    """
    
    def __init__(self):
        """Initialize relation extractor."""
        # Initialize spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Improved relation patterns
        self._init_relation_patterns()
        
        # Statistics
        self.stats = {
            "total_relations_extracted": 0,
            "by_type": defaultdict(int),
            "by_strategy": defaultdict(int)
        }
        
        logger.info("RelationExtractor initialized")
    
    def _init_relation_patterns(self):
        """Initialize improved regex patterns for relation extraction."""
        
        # CITES relations - improved patterns
        self.cites_patterns = [
            # "As per Section 5" or "Under Section 5"
            re.compile(
                r'(?:as per|under|in|pursuant to|according to|by virtue of)\s+'
                r'(Section|Sec\.|Article|Art\.|Rule|Clause)\s+'
                r'(\d+(?:\s*\([a-zA-Z0-9]+\))?)',
                re.IGNORECASE
            ),
            # "Section 5 provides" or "Section 5 states"
            re.compile(
                r'(Section|Sec\.|Article|Art\.|Rule)\s+'
                r'(\d+(?:\s*\([a-zA-Z0-9]+\))?)\s+'
                r'(?:provides|states|mandates|requires|specifies)',
                re.IGNORECASE
            ),
            # "in terms of Section 5"
            re.compile(
                r'in terms of\s+(Section|Sec\.|Article|Art\.|Rule)\s+'
                r'(\d+(?:\s*\([a-zA-Z0-9]+\))?)',
                re.IGNORECASE
            ),
            # "vide G.O.MS.No. 67"
            re.compile(
                r'vide\s+(?:G\.O\.|GO)(?:MS|RT)?\.?\s*No\.?\s*(\d+)',
                re.IGNORECASE
            )
        ]
        
        # IMPLEMENTS relations
        self.implements_patterns = [
            # "implements Section 5" or "implementing Section 5"
            re.compile(
                r'implement(?:s|ing)?\s+'
                r'(?:the provisions of\s+)?'
                r'(Section|Sec\.|Article|Rule)\s+'
                r'(\d+)',
                re.IGNORECASE
            ),
            # "in implementation of Act"
            re.compile(
                r'in implementation of\s+(?:the\s+)?'
                r'([A-Z][A-Za-z\s,]+(?:Act|Rule|Policy|Scheme))',
                re.IGNORECASE
            ),
            # "gives effect to Section"
            re.compile(
                r'gives effect to\s+'
                r'(Section|Sec\.|Article|Rule)\s+'
                r'(\d+)',
                re.IGNORECASE
            )
        ]
        
        # SUPERSEDES relations  
        self.supersedes_patterns = [
            # "supersedes G.O.MS.No. 45"
            re.compile(
                r'supersedes?\s+(?:the\s+)?'
                r'(?:G\.O\.|GO)(?:MS|RT)?\.?\s*No\.?\s*(\d+)',
                re.IGNORECASE
            ),
            # "in supersession of GO"
            re.compile(
                r'in supersession of\s+(?:the\s+)?'
                r'(?:G\.O\.|GO)(?:MS|RT)?\.?\s*No\.?\s*(\d+)',
                re.IGNORECASE
            ),
            # "hereby rescinded" or "hereby cancelled"
            re.compile(
                r'(?:G\.O\.|GO)(?:MS|RT)?\.?\s*No\.?\s*(\d+)'
                r'.*?(?:is hereby|are hereby)\s+'
                r'(?:rescinded|cancelled|withdrawn|revoked)',
                re.IGNORECASE
            ),
            # "replaces the earlier"
            re.compile(
                r'replaces?\s+(?:the\s+)?(?:earlier|previous)\s+'
                r'(?:G\.O\.|GO|order|notification)',
                re.IGNORECASE
            )
        ]
        
        # AMENDS relations
        self.amends_patterns = [
            # "amends Section 5"
            re.compile(
                r'amends?\s+(?:the\s+)?'
                r'(Section|Sec\.|Article|Rule)\s+'
                r'(\d+)',
                re.IGNORECASE
            ),
            # "amendment to Section"
            re.compile(
                r'amendment\s+to\s+'
                r'(Section|Sec\.|Article|Rule)\s+'
                r'(\d+)',
                re.IGNORECASE
            ),
            # "Section 5 is amended"
            re.compile(
                r'(Section|Sec\.|Article|Rule)\s+'
                r'(\d+)\s+'
                r'(?:is|shall be)\s+amended',
                re.IGNORECASE
            )
        ]
        
        # DEFINES relations
        self.defines_patterns = [
            # "Section 5 defines X as"
            re.compile(
                r'(Section|Sec\.|Article|Rule)\s+'
                r'(\d+)\s+'
                r'defines?\s+'
                r'"([^"]+)"',
                re.IGNORECASE
            ),
            # '"X" means (as defined in Section 5)'
            re.compile(
                r'"([^"]+)"\s+means\s+.*?'
                r'\(as defined in\s+(Section|Sec\.|Article|Rule)\s+(\d+)\)',
                re.IGNORECASE
            )
        ]
        
        # MANDATES relations
        self.mandates_patterns = [
            # "Section 5 mandates that"
            re.compile(
                r'(Section|Sec\.|Article|Rule)\s+'
                r'(\d+)\s+'
                r'(?:mandates?|requires?|stipulates?|prescribes?)',
                re.IGNORECASE
            ),
            # "shall be mandatory under Section 5"
            re.compile(
                r'shall be (?:mandatory|compulsory|required)\s+under\s+'
                r'(Section|Sec\.|Article|Rule)\s+'
                r'(\d+)',
                re.IGNORECASE
            )
        ]
        
        # ALLOCATES patterns (for schemes/budget)
        self.allocates_patterns = [
            # "Rs. 100 crore allocated for Nadu-Nedu"
            re.compile(
                r'(?:Rs|INR|₹)\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*'
                r'(?:crore|lakh|thousand)?\s+'
                r'(?:is\s+)?allocated\s+(?:for|to|under)\s+'
                r'([A-Z][A-Za-z\s-]+(?:Scheme|Programme|Project))',
                re.IGNORECASE
            )
        ]
        
        # APPLIES_TO patterns (for districts/categories)
        self.applies_to_patterns = [
            # "applicable to SC/ST students"
            re.compile(
                r'applicable\s+to\s+'
                r'(SC|ST|OBC|EWS|Minority|Girl|Rural|Urban)',
                re.IGNORECASE
            ),
            # "in Visakhapatnam district"
            re.compile(
                r'in\s+([A-Z][a-z]+)\s+district',
                re.IGNORECASE
            )
        ]
    
    def extract_cites_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract citation relationships."""
        relations = []
        
        for pattern in self.cites_patterns:
            for match in pattern.finditer(text):
                # Extract context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                # Build target reference
                if len(match.groups()) >= 2:
                    target = f"{match.group(1)} {match.group(2)}"
                else:
                    target = match.group(0)
                
                relation = Relation(
                    relation_type="cites",
                    source=chunk_id,
                    target=target.strip(),
                    source_chunk_id=chunk_id,
                    confidence=0.9,
                    context=context
                )
                relations.append(relation)
                self.stats["by_type"]["cites"] += 1
        
        return relations
    
    def extract_implements_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract implementation relationships."""
        relations = []
        
        for pattern in self.implements_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                # Build target
                if len(match.groups()) >= 2 and match.group(2):
                    target = f"{match.group(1)} {match.group(2)}"
                else:
                    target = match.group(1)
                
                relation = Relation(
                    relation_type="implements",
                    source=chunk_id,
                    target=target.strip(),
                    source_chunk_id=chunk_id,
                    confidence=0.85,
                    context=context
                )
                relations.append(relation)
                self.stats["by_type"]["implements"] += 1
        
        return relations
    
    def extract_supersedes_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract supersession relationships."""
        relations = []
        
        for pattern in self.supersedes_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                # Try to extract GO number
                go_match = re.search(r'(\d+)', match.group(0))
                if go_match:
                    target = f"G.O.MS.No. {go_match.group(1)}"
                else:
                    target = match.group(0)
                
                relation = Relation(
                    relation_type="supersedes",
                    source=chunk_id,
                    target=target.strip(),
                    source_chunk_id=chunk_id,
                    confidence=0.95,  # High confidence for explicit supersession
                    context=context
                )
                relations.append(relation)
                self.stats["by_type"]["supersedes"] += 1
        
        return relations
    
    def extract_amends_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract amendment relationships."""
        relations = []
        
        for pattern in self.amends_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                if len(match.groups()) >= 2:
                    target = f"{match.group(1)} {match.group(2)}"
                else:
                    target = match.group(0)
                
                relation = Relation(
                    relation_type="amends",
                    source=chunk_id,
                    target=target.strip(),
                    source_chunk_id=chunk_id,
                    confidence=0.9,
                    context=context
                )
                relations.append(relation)
                self.stats["by_type"]["amends"] += 1
        
        return relations
    
    def extract_defines_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract definition relationships."""
        relations = []
        
        for pattern in self.defines_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                # Extract defined term
                if len(match.groups()) >= 3:
                    term = match.group(1) if '"' in match.group(1) else match.group(3)
                    target = f"{match.group(2)} {match.group(3)}" if len(match.groups()) >= 3 else match.group(2)
                else:
                    term = match.group(1)
                    target = match.group(0)
                
                relation = Relation(
                    relation_type="defines",
                    source=chunk_id,
                    target=target.strip(),
                    source_chunk_id=chunk_id,
                    confidence=0.85,
                    context=context
                )
                relations.append(relation)
                self.stats["by_type"]["defines"] += 1
        
        return relations
    
    def extract_mandates_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract mandate relationships."""
        relations = []
        
        for pattern in self.mandates_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                if len(match.groups()) >= 2:
                    target = f"{match.group(1)} {match.group(2)}"
                else:
                    target = match.group(0)
                
                relation = Relation(
                    relation_type="mandates",
                    source=chunk_id,
                    target=target.strip(),
                    source_chunk_id=chunk_id,
                    confidence=0.85,
                    context=context
                )
                relations.append(relation)
                self.stats["by_type"]["mandates"] += 1
        
        return relations
    
    def extract_spacy_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract relations using spaCy dependency parsing."""
        if not self.nlp or not text or len(text) > 100000:
            return []
        
        relations = []
        
        try:
            # Process text
            doc = self.nlp(text[:10000])  # Limit to first 10k chars for performance
            
            # Look for verb-object relations
            for token in doc:
                # Look for key verbs
                if token.lemma_ in ["implement", "supersede", "amend", "cite", "define", "mandate"]:
                    # Find object
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            # Extract context
                            start = max(0, token.idx - 30)
                            end = min(len(text), child.idx + len(child.text) + 30)
                            context = text[start:end].strip()
                            
                            relation = Relation(
                                relation_type=token.lemma_,
                                source=chunk_id,
                                target=child.text,
                                source_chunk_id=chunk_id,
                                confidence=0.7,  # Lower confidence for dependency-based
                                context=context
                            )
                            relations.append(relation)
                            self.stats["by_type"][token.lemma_] += 1
                            self.stats["by_strategy"]["spacy"] += 1
            
        except Exception as e:
            logger.debug(f"Error in spaCy relation extraction: {e}")
        
        return relations
    
    def extract_all_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """
        Extract all types of relations from text.
        
        Args:
            text: Input text
            chunk_id: Chunk identifier
            
        Returns:
            List of extracted relations
        """
        if not text or not text.strip():
            return []
        
        all_relations = []
        
        try:
            # Pattern-based extraction (primary strategy)
            all_relations.extend(self.extract_cites_relations(text, chunk_id))
            all_relations.extend(self.extract_implements_relations(text, chunk_id))
            all_relations.extend(self.extract_supersedes_relations(text, chunk_id))
            all_relations.extend(self.extract_amends_relations(text, chunk_id))
            all_relations.extend(self.extract_defines_relations(text, chunk_id))
            all_relations.extend(self.extract_mandates_relations(text, chunk_id))
            
            # spaCy-based extraction (secondary strategy)
            if self.nlp:
                all_relations.extend(self.extract_spacy_relations(text, chunk_id))
            
            # Deduplicate
            all_relations = self._deduplicate_relations(all_relations)
            
            # Update stats
            self.stats["total_relations_extracted"] += len(all_relations)
            
            if len(all_relations) > 0:
                logger.debug(f"Extracted {len(all_relations)} relations from chunk {chunk_id}")
            
        except Exception as e:
            logger.error(f"Error extracting relations from chunk {chunk_id}: {e}")
        
        return all_relations
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate or highly similar relations."""
        if not relations:
            return []
        
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # Normalize target for comparison
            target_norm = re.sub(r'\s+', ' ', relation.target.lower().strip())
            
            # Create deduplication key
            key = (
                relation.relation_type,
                target_norm,
                relation.source_chunk_id
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def relations_to_dict(self, relations: List[Relation]) -> List[Dict]:
        """Convert relation objects to dictionaries."""
        return [r.to_dict() for r in relations]
    
    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        return dict(self.stats)


# Testing
if __name__ == "__main__":
    extractor = RelationExtractor()
    
    # Test with sample legal text
    test_texts = [
        "As per Section 5 of the Education Act, schools must maintain records.",
        "This GO implements the provisions of Section 10 of the Act.",
        "In supersession of G.O.MS.No. 45/2020, the following is notified.",
        "Section 12 mandates that all schools provide free education.",
        "Vide G.O.MS.No. 67 dated 15.04.2023, Nadu-Nedu scheme is implemented.",
        "Under Article 21A of the Constitution, education is a fundamental right."
    ]
    
    print("Testing Relation Extraction:\n")
    
    total_relations = 0
    for i, text in enumerate(test_texts):
        relations = extractor.extract_all_relations(text, f"test_chunk_{i}")
        print(f"Text: {text}")
        print(f"Relations found: {len(relations)}")
        for rel in relations:
            print(f"  - {rel.relation_type}: {rel.source} -> {rel.target}")
        print()
        total_relations += len(relations)
    
    print(f"Total relations extracted: {total_relations}")
    print(f"\nStatistics: {extractor.get_stats()}")