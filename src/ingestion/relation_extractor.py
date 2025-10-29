"""
Relation Extractor for education policy documents.

Extracts relationships between documents to build the knowledge graph:
- CITES relations: "as per Section 12 of RTE Act"
- IMPLEMENTS relations: "in pursuance of Section 12"
- SUPERSEDES relations: "in supersession of GO MS No. X"
- AMENDS relations: "this GO amends..."
- DEFINES relations: document defines a concept
- MANDATES relations: "Section 12 mandates..."
"""

import re
import spacy
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Relation:
    """Represents a relationship between documents or entities."""
    relation_type: str  # cites, implements, supersedes, amends, defines, mandates
    source_chunk_id: str
    target_reference: str  # The referenced entity (e.g., "Section 12, RTE Act")
    target_doc_id: Optional[str] = None  # Will be resolved later
    context: str = ""  # Surrounding sentence/phrase
    confidence: float = 1.0
    position: Tuple[int, int] = (0, 0)  # Start, end positions in text
    metadata: Dict = None


class RelationExtractor:
    """
    Extract semantic relationships between documents and legal entities.
    
    This feeds into the knowledge graph to enable queries like:
    - "What implements Section 12 of RTE Act?"
    - "Which GOs supersede GO MS No. 45/2018?"
    - "What does Article 21A mandate?"
    """
    
    def __init__(self):
        """Initialize relation extractor with patterns and NLP models."""
        self._compile_patterns()
        self._init_nlp_model()
        
        logger.info("RelationExtractor initialized with relationship patterns")
    
    def _init_nlp_model(self):
        """Initialize spaCy model for dependency parsing."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model for dependency parsing")
        except OSError:
            logger.warning("spaCy model not available. Using pattern-based extraction only.")
            self.nlp = None
    
    def _compile_patterns(self):
        """Compile regex patterns for different relation types."""
        
        # CITES relations - when a document cites another
        self.cites_patterns = [
            # "as per Section X"
            re.compile(r'\b(?:as per|according to|under|pursuant to)\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE),
            # "in accordance with"
            re.compile(r'\bin accordance with\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE),
            # "Section X mandates/provides/states"
            re.compile(r'\b((?:Section|Article|Rule)\s+\d+[^.!?]*?)\s+(?:mandates|provides|states|declares)', re.IGNORECASE),
            # "GO MS No. X provides"
            re.compile(r'\b(G\.?O\.?\s*(?:Ms\.?|MS\.?)\s*(?:No\.?)?\s*\d+[^.!?]*?)\s+(?:provides|mandates|states)', re.IGNORECASE)
        ]
        
        # IMPLEMENTS relations - when a document implements another
        self.implements_patterns = [
            # "in pursuance of"
            re.compile(r'\bin pursuance of\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE),
            # "for implementation of"
            re.compile(r'\bfor (?:the )?implementation of\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE),
            # "in exercise of powers under"
            re.compile(r'\bin exercise of (?:the )?powers? (?:under|conferred by)\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE),
            # "to implement"
            re.compile(r'\bto implement\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE)
        ]
        
        # SUPERSEDES relations - when a document supersedes another
        self.supersedes_patterns = [
            # "in supersession of"
            re.compile(r'\bin supersession of\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE),
            # "this order supersedes"
            re.compile(r'\bthis (?:order|GO|notification)\s+supersedes\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE),
            # "earlier GO ... is hereby cancelled"
            re.compile(r'\b(?:earlier|previous)\s+([^.!?]*?G\.?O\.?[^.!?]*?)\s+is hereby (?:cancelled|superseded)', re.IGNORECASE),
            # "stands superseded"
            re.compile(r'\b([^.!?]*?G\.?O\.?[^.!?]*?)\s+stands? (?:superseded|cancelled)', re.IGNORECASE)
        ]
        
        # AMENDS relations - when a document amends another
        self.amends_patterns = [
            # "this GO amends"
            re.compile(r'\bthis (?:GO|order|notification)\s+amends\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE),
            # "in modification of"
            re.compile(r'\bin modification of\s+([^.!?]+?)(?:\.|,|;|$)', re.IGNORECASE),
            # "is hereby amended"
            re.compile(r'\b([^.!?]+?)\s+is hereby amended', re.IGNORECASE),
            # "substituted by"
            re.compile(r'\b([^.!?]+?)\s+(?:is|shall be)\s+substituted by', re.IGNORECASE)
        ]
        
        # DEFINES relations - when a document defines a concept
        self.defines_patterns = [
            # "X means" or "X shall mean"
            re.compile(r'\b([^.!?"]+?)\s+(?:means?|shall mean)\s+([^.!?]+?)(?:\.|;|$)', re.IGNORECASE),
            # "for the purpose of this Act, X means"
            re.compile(r'\bfor the purposes? of [^,]+,\s*([^.!?]+?)\s+means?\s+([^.!?]+?)(?:\.|;|$)', re.IGNORECASE),
            # "X is defined as"
            re.compile(r'\b([^.!?]+?)\s+is defined as\s+([^.!?]+?)(?:\.|;|$)', re.IGNORECASE)
        ]
        
        # MANDATES relations - when a law/section mandates something
        self.mandates_patterns = [
            # "Section X mandates"
            re.compile(r'\b((?:Section|Article|Rule)\s+\d+[^.!?]*?)\s+mandates?\s+([^.!?]+?)(?:\.|;|$)', re.IGNORECASE),
            # "shall ensure/provide/maintain"
            re.compile(r'\b([^.!?]*?(?:school|teacher|student)[^.!?]*?)\s+shall\s+(?:ensure|provide|maintain)\s+([^.!?]+?)(?:\.|;|$)', re.IGNORECASE),
            # "it shall be mandatory"
            re.compile(r'\bit shall be mandatory (?:for|to)\s+([^.!?]+?)(?:\.|;|$)', re.IGNORECASE)
        ]
    
    def extract_cites_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract CITES relations from text."""
        relations = []
        
        for pattern in self.cites_patterns:
            for match in pattern.finditer(text):
                target_ref = match.group(1).strip()
                
                # Get surrounding context (sentence)
                context = self._get_sentence_context(text, match.start())
                
                relations.append(Relation(
                    relation_type="cites",
                    source_chunk_id=chunk_id,
                    target_reference=target_ref,
                    context=context,
                    confidence=0.9,
                    position=match.span()
                ))
        
        return relations
    
    def extract_implements_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract IMPLEMENTS relations from text."""
        relations = []
        
        for pattern in self.implements_patterns:
            for match in pattern.finditer(text):
                target_ref = match.group(1).strip()
                context = self._get_sentence_context(text, match.start())
                
                relations.append(Relation(
                    relation_type="implements",
                    source_chunk_id=chunk_id,
                    target_reference=target_ref,
                    context=context,
                    confidence=0.95,
                    position=match.span()
                ))
        
        return relations
    
    def extract_supersedes_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract SUPERSEDES relations from text."""
        relations = []
        
        for pattern in self.supersedes_patterns:
            for match in pattern.finditer(text):
                target_ref = match.group(1).strip()
                context = self._get_sentence_context(text, match.start())
                
                # Parse superseded GO details
                superseded_go = self._parse_go_reference(target_ref)
                
                relations.append(Relation(
                    relation_type="supersedes",
                    source_chunk_id=chunk_id,
                    target_reference=target_ref,
                    context=context,
                    confidence=0.95,
                    position=match.span(),
                    metadata={"superseded_go": superseded_go}
                ))
        
        return relations
    
    def extract_amends_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract AMENDS relations from text."""
        relations = []
        
        for pattern in self.amends_patterns:
            for match in pattern.finditer(text):
                target_ref = match.group(1).strip()
                context = self._get_sentence_context(text, match.start())
                
                relations.append(Relation(
                    relation_type="amends",
                    source_chunk_id=chunk_id,
                    target_reference=target_ref,
                    context=context,
                    confidence=0.9,
                    position=match.span()
                ))
        
        return relations
    
    def extract_defines_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract DEFINES relations from text."""
        relations = []
        
        for pattern in self.defines_patterns:
            for match in pattern.finditer(text):
                # For defines relations, we have both the term and definition
                term = match.group(1).strip()
                definition = match.group(2).strip() if match.lastindex > 1 else ""
                
                context = self._get_sentence_context(text, match.start())
                
                relations.append(Relation(
                    relation_type="defines",
                    source_chunk_id=chunk_id,
                    target_reference=term,
                    context=context,
                    confidence=0.95,
                    position=match.span(),
                    metadata={"definition": definition}
                ))
        
        return relations
    
    def extract_mandates_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """Extract MANDATES relations from text."""
        relations = []
        
        for pattern in self.mandates_patterns:
            for match in pattern.finditer(text):
                if match.lastindex >= 2:
                    source_ref = match.group(1).strip()
                    mandate = match.group(2).strip()
                    
                    context = self._get_sentence_context(text, match.start())
                    
                    relations.append(Relation(
                        relation_type="mandates",
                        source_chunk_id=chunk_id,
                        target_reference=source_ref,
                        context=context,
                        confidence=0.9,
                        position=match.span(),
                        metadata={"mandate": mandate}
                    ))
                else:
                    # Single group pattern
                    target_ref = match.group(1).strip()
                    context = self._get_sentence_context(text, match.start())
                    
                    relations.append(Relation(
                        relation_type="mandates",
                        source_chunk_id=chunk_id,
                        target_reference=target_ref,
                        context=context,
                        confidence=0.8,
                        position=match.span()
                    ))
        
        return relations
    
    def extract_spacy_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """
        Extract relations using spaCy dependency parsing.
        
        This can catch more complex semantic relationships that 
        regex patterns might miss.
        """
        if not self.nlp:
            return []
        
        relations = []
        
        try:
            doc = self.nlp(text)
            
            for sent in doc.sents:
                # Look for specific dependency patterns
                
                # Pattern: "X implements Y" (subject-verb-object)
                for token in sent:
                    if token.lemma_ in ["implement", "enforce", "execute"] and token.pos_ == "VERB":
                        # Find subject and object
                        subj = None
                        obj = None
                        
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                subj = child
                            elif child.dep_ in ["dobj", "pobj"]:
                                obj = child
                        
                        if subj and obj:
                            # Extract the full phrases
                            subj_phrase = self._extract_phrase(subj)
                            obj_phrase = self._extract_phrase(obj)
                            
                            relations.append(Relation(
                                relation_type="implements",
                                source_chunk_id=chunk_id,
                                target_reference=obj_phrase,
                                context=sent.text,
                                confidence=0.7,
                                position=(sent.start_char, sent.end_char),
                                metadata={"method": "spacy_dependency", "subject": subj_phrase}
                            ))
                
                # Pattern: "According to X" / "As per X"
                for token in sent:
                    if token.text.lower() in ["according", "per"] and token.head.pos_ == "ADP":
                        # Find the object of the preposition
                        for child in token.head.children:
                            if child.dep_ == "pobj":
                                ref_phrase = self._extract_phrase(child)
                                
                                relations.append(Relation(
                                    relation_type="cites",
                                    source_chunk_id=chunk_id,
                                    target_reference=ref_phrase,
                                    context=sent.text,
                                    confidence=0.8,
                                    position=(sent.start_char, sent.end_char),
                                    metadata={"method": "spacy_dependency"}
                                ))
        
        except Exception as e:
            logger.warning(f"Error in spaCy relation extraction: {e}")
        
        return relations
    
    def _extract_phrase(self, token) -> str:
        """Extract the full phrase for a token (including dependents)."""
        # Get all tokens in the subtree
        phrase_tokens = list(token.subtree)
        phrase_tokens.sort(key=lambda x: x.i)  # Sort by position
        
        return " ".join([t.text for t in phrase_tokens])
    
    def _get_sentence_context(self, text: str, position: int, window: int = 150) -> str:
        """Get the sentence containing the given position."""
        # Find sentence boundaries around the position
        start = max(0, position - window)
        end = min(len(text), position + window)
        
        context = text[start:end]
        
        # Try to find complete sentences
        sentences = re.split(r'[.!?]+', context)
        if len(sentences) > 1:
            # Return the middle sentence(s)
            middle = len(sentences) // 2
            return sentences[middle].strip()
        
        return context.strip()
    
    def _parse_go_reference(self, go_text: str) -> Dict:
        """Parse GO reference text to extract structured information."""
        go_pattern = re.compile(
            r'G\.?O\.?\s*(?:Ms\.?|MS\.?|Rt\.?|RT\.?)?\s*(?:No\.?)?\s*(\d+)(?:/(\d{4}))?',
            re.IGNORECASE
        )
        
        match = go_pattern.search(go_text)
        if match:
            return {
                "number": int(match.group(1)),
                "year": int(match.group(2)) if match.group(2) else None,
                "full_text": go_text
            }
        
        return {"full_text": go_text}
    
    def extract_all_relations(self, text: str, chunk_id: str) -> List[Relation]:
        """
        Extract all types of relations from text.
        
        Args:
            text: Input text to analyze
            chunk_id: ID of the source chunk
            
        Returns:
            List of all extracted relations
        """
        if not text or not text.strip():
            return []
        
        all_relations = []
        
        try:
            # Extract different types of relations
            all_relations.extend(self.extract_cites_relations(text, chunk_id))
            all_relations.extend(self.extract_implements_relations(text, chunk_id))
            all_relations.extend(self.extract_supersedes_relations(text, chunk_id))
            all_relations.extend(self.extract_amends_relations(text, chunk_id))
            all_relations.extend(self.extract_defines_relations(text, chunk_id))
            all_relations.extend(self.extract_mandates_relations(text, chunk_id))
            
            # Add spaCy-based relations
            all_relations.extend(self.extract_spacy_relations(text, chunk_id))
            
            # Deduplicate based on similar target references
            all_relations = self._deduplicate_relations(all_relations)
            
            logger.debug(f"Extracted {len(all_relations)} relations from chunk {chunk_id}")
            
        except Exception as e:
            logger.error(f"Error extracting relations from chunk {chunk_id}: {e}")
        
        return all_relations
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate or very similar relations."""
        if not relations:
            return []
        
        # Group by relation type and target reference
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # Create a key for deduplication
            key = (
                relation.relation_type,
                relation.target_reference.lower().strip(),
                relation.source_chunk_id
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def relations_to_dict(self, relations: List[Relation]) -> List[Dict]:
        """Convert relation objects to dictionaries for JSON serialization."""
        return [
            {
                "relation_type": rel.relation_type,
                "source_chunk_id": rel.source_chunk_id,
                "target_reference": rel.target_reference,
                "target_doc_id": rel.target_doc_id,
                "context": rel.context,
                "confidence": rel.confidence,
                "position": rel.position,
                "metadata": rel.metadata or {}
            }
            for rel in relations
        ]


# Example usage and testing
if __name__ == "__main__":
    # Test the relation extractor
    extractor = RelationExtractor()
    
    # Test text with various relation types
    test_text = """
    As per Section 12(1)(c) of the RTE Act, private schools shall reserve 25% seats 
    for EWS and DG children. In pursuance of Article 21A of the Constitution, 
    this notification is issued.
    
    In supersession of GO MS No. 45/2018 dated 10.03.2018, this order provides 
    new guidelines. This GO amends the earlier provisions regarding teacher recruitment.
    
    For the purpose of this Act, "disadvantaged group" means a group of people 
    who may be discriminated against. Section 21 mandates formation of School 
    Management Committees in all schools.
    """
    
    relations = extractor.extract_all_relations(test_text, "test_chunk_001")
    
    print(f"Extracted {len(relations)} relations:")
    for rel in relations:
        print(f"- {rel.relation_type}: {rel.target_reference}")
        print(f"  Context: {rel.context[:100]}...")
        print(f"  Confidence: {rel.confidence}")
        print()