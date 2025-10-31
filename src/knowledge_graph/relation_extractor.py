"""Relation extraction for building knowledge graph and bridge table"""
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    entity_type: str  # legal_ref, go_ref, scheme, metric, district, etc.
    entity_value: str  # The actual text
    confidence: float
    span_start: int
    span_end: int
    context: str  # Surrounding context

@dataclass
class ExtractedRelation:
    """Represents a relation between documents or entities"""
    relation_type: str  # cites, implements, amends, supersedes, etc.
    source_doc_id: str
    target_entity: str  # Could be doc_id or entity value
    confidence: float
    evidence: str  # Text evidence for the relation
    extraction_method: str  # regex, ner, semantic, etc.

@dataclass
class ChunkAnalysis:
    """Analysis results for a single chunk"""
    chunk_id: str
    doc_id: str
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    bridge_topic_matches: List[str]  # Bridge topic IDs this chunk relates to
    processing_notes: List[str]

class RelationExtractor:
    """Extract relations and entities from document chunks for knowledge graph construction"""
    
    def __init__(self, education_terms_path: Optional[str] = None):
        """
        Initialize the relation extractor
        
        Args:
            education_terms_path: Path to education terms dictionary
        """
        self.education_terms = {}
        self.nlp = None
        
        # Load education terms dictionary
        if education_terms_path:
            self.load_education_terms(education_terms_path)
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                # Try to load the model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Using basic tokenizer.")
                self.nlp = English()
        else:
            logger.warning("spaCy not available. Relation extraction will use regex patterns only.")
        
        # Compile regex patterns
        self._compile_patterns()
        
        # AP Districts list
        self.ap_districts = {
            'anantapur', 'chittoor', 'east godavari', 'guntur', 'kadapa', 'krishna',
            'kurnool', 'prakasam', 'srikakulam', 'visakhapatnam', 'vizianagaram',
            'west godavari', 'nellore', 'tirupati', 'annamayya', 'sri potti sriramulu nellore',
            'ysr', 'ysr kadapa', 'konaseema', 'eluru', 'nr', 'palnadu', 'bapatla',
            'alluri sitharama raju', 'kakinada', 'anakapalli'
        }
    
    def load_education_terms(self, terms_path: str):
        """Load education terms dictionary"""
        try:
            with open(terms_path, 'r', encoding='utf-8') as f:
                self.education_terms = json.load(f)
            logger.info(f"Loaded education terms from {terms_path}")
        except Exception as e:
            logger.error(f"Failed to load education terms: {e}")
            self.education_terms = {}
    
    def _compile_patterns(self):
        """Compile regex patterns for relation extraction"""
        
        # Legal reference patterns
        self.legal_patterns = [
            # Section patterns
            r'[Ss]ection\s+(\d+(?:\(\d+\))?(?:\([a-z]\))?)\s+of\s+(?:the\s+)?([A-Z][^.]{10,50}(?:Act|Rules))',
            r'[Ss]ections?\s+(\d+(?:\s*(?:and|,|to)\s*\d+)*)\s+of\s+(?:the\s+)?([A-Z][^.]{10,50}(?:Act|Rules))',
            
            # Article patterns
            r'[Aa]rticle\s+(\d+[A-Z]?)\s+of\s+(?:the\s+)?([A-Z][^.]{10,50}(?:Constitution|Charter))',
            
            # Rule patterns
            r'[Rr]ule\s+(\d+(?:\(\d+\))?)\s+of\s+(?:the\s+)?([A-Z][^.]{10,50}Rules)',
            
            # Generic references
            r'(?:as per|in accordance with|pursuant to|under)\s+([Ss]ection\s+\d+(?:\(\d+\))?(?:\([a-z]\))?)',
            r'(?:vide|as per|under)\s+([A-Z][^.]{5,30}(?:Act|Rules))',
        ]
        
        # GO reference patterns
        self.go_patterns = [
            # Standard GO patterns
            r'G\.?O\.?\s*M\.?S\.?\s*No\.?\s*(\d+)\s*(?:dated?\s*|dt\.?\s*|of\s*)?(\d{1,2}[.-]\d{1,2}[.-]\d{2,4})?',
            r'GO\s+(?:MS\s+)?No\.?\s*(\d+)(?:/(\d{4}))?(?:\s*dated?\s*(\d{1,2}[.-]\d{1,2}[.-]\d{2,4}))?',
            r'Government\s+Order\s+(?:MS\s+)?No\.?\s*(\d+)',
            r'vide\s+GO\s+(?:No\.?\s*)?(\d+)',
            
            # Proceedings patterns
            r'Proc\.?\s*(?:Rc\.?\s*)?No\.?\s*(\d+(?:[/-]\w+)*)',
            r'Proceedings\s+(?:Rc\.?\s*)?No\.?\s*(\d+(?:[/-]\w+)*)',
        ]
        
        # Supersession patterns
        self.supersession_patterns = [
            r'[Ii]n\s+supersession\s+of\s+(?:GO\s+(?:MS\s+)?No\.?\s*(\d+)(?:/(\d{4}))?|earlier\s+(?:order|GO))',
            r'[Tt]his\s+order\s+supersedes?\s+(?:GO\s+(?:MS\s+)?No\.?\s*(\d+)|earlier\s+(?:order|GO))',
            r'[Ii]n\s+(?:partial\s+)?modification\s+of\s+(?:GO\s+(?:MS\s+)?No\.?\s*(\d+))',
            r'[Ss]upersedes?\s+(?:the\s+)?(?:earlier\s+)?(?:order|GO)',
        ]
        
        # Implementation patterns
        self.implementation_patterns = [
            r'[Ii]n\s+pursuance\s+of\s+([Ss]ection\s+\d+(?:\(\d+\))?)',
            r'[Ff]or\s+implementation\s+of\s+(?:the\s+)?([A-Z][^.]{10,50}(?:Act|Rules|Scheme))',
            r'[Ii]n\s+accordance\s+with\s+(?:the\s+provisions\s+of\s+)?([A-Z][^.]{10,50}(?:Act|Rules))',
            r'[Uu]nder\s+(?:the\s+provisions\s+of\s+)?([A-Z][^.]{10,50}(?:Act|Rules))',
        ]
        
        # Citation patterns
        self.citation_patterns = [
            r'[Aa]s\s+(?:mentioned|stated|provided)\s+in\s+([A-Z][^.]{10,50}(?:Act|Rules|Order))',
            r'[Aa]s\s+per\s+(?:the\s+)?([A-Z][^.]{10,50}(?:Act|Rules|Order|Guidelines))',
            r'[Rr]eference\s+(?:is\s+)?(?:made\s+)?to\s+([A-Z][^.]{10,50}(?:Act|Rules))',
        ]
        
        # Compile all patterns
        self.compiled_legal = [re.compile(p, re.IGNORECASE) for p in self.legal_patterns]
        self.compiled_go = [re.compile(p, re.IGNORECASE) for p in self.go_patterns]
        self.compiled_supersession = [re.compile(p, re.IGNORECASE) for p in self.supersession_patterns]
        self.compiled_implementation = [re.compile(p, re.IGNORECASE) for p in self.implementation_patterns]
        self.compiled_citation = [re.compile(p, re.IGNORECASE) for p in self.citation_patterns]
    
    def extract_legal_references(self, text: str) -> List[ExtractedEntity]:
        """Extract legal references (sections, articles, rules)"""
        entities = []
        
        for pattern in self.compiled_legal:
            matches = pattern.finditer(text)
            for match in matches:
                entity = ExtractedEntity(
                    entity_type="legal_ref",
                    entity_value=match.group(0),
                    confidence=0.9,  # High confidence for regex matches
                    span_start=match.start(),
                    span_end=match.end(),
                    context=self._extract_context(text, match.start(), match.end())
                )
                entities.append(entity)
        
        return entities
    
    def extract_go_references(self, text: str) -> List[ExtractedEntity]:
        """Extract Government Order references"""
        entities = []
        
        for pattern in self.compiled_go:
            matches = pattern.finditer(text)
            for match in matches:
                entity = ExtractedEntity(
                    entity_type="go_ref",
                    entity_value=match.group(0),
                    confidence=0.9,
                    span_start=match.start(),
                    span_end=match.end(),
                    context=self._extract_context(text, match.start(), match.end())
                )
                entities.append(entity)
        
        return entities
    
    def extract_schemes(self, text: str) -> List[ExtractedEntity]:
        """Extract education schemes mentioned in text"""
        entities = []
        
        # Get schemes from education terms
        schemes = self.education_terms.get('schemes', {})
        
        for scheme_name, scheme_info in schemes.items():
            # Check main name
            if self._case_insensitive_search(scheme_name, text):
                positions = self._find_positions(scheme_name, text)
                for start, end in positions:
                    entity = ExtractedEntity(
                        entity_type="scheme",
                        entity_value=scheme_name,
                        confidence=0.95,
                        span_start=start,
                        span_end=end,
                        context=self._extract_context(text, start, end)
                    )
                    entities.append(entity)
            
            # Check aliases
            aliases = scheme_info.get('aliases', [])
            for alias in aliases:
                if self._case_insensitive_search(alias, text):
                    positions = self._find_positions(alias, text)
                    for start, end in positions:
                        entity = ExtractedEntity(
                            entity_type="scheme",
                            entity_value=scheme_name,  # Use canonical name
                            confidence=0.85,
                            span_start=start,
                            span_end=end,
                            context=self._extract_context(text, start, end)
                        )
                        entities.append(entity)
        
        return entities
    
    def extract_metrics(self, text: str) -> List[ExtractedEntity]:
        """Extract education metrics mentioned in text"""
        entities = []
        
        # Get metrics from education terms
        metrics = self.education_terms.get('metrics', {})
        
        for metric_name, metric_info in metrics.items():
            # Check main name
            if self._case_insensitive_search(metric_name, text):
                positions = self._find_positions(metric_name, text)
                for start, end in positions:
                    entity = ExtractedEntity(
                        entity_type="metric",
                        entity_value=metric_name,
                        confidence=0.9,
                        span_start=start,
                        span_end=end,
                        context=self._extract_context(text, start, end)
                    )
                    entities.append(entity)
            
            # Check aliases
            aliases = metric_info.get('aliases', [])
            for alias in aliases:
                if self._case_insensitive_search(alias, text):
                    positions = self._find_positions(alias, text)
                    for start, end in positions:
                        entity = ExtractedEntity(
                            entity_type="metric",
                            entity_value=metric_name,  # Use canonical name
                            confidence=0.8,
                            span_start=start,
                            span_end=end,
                            context=self._extract_context(text, start, end)
                        )
                        entities.append(entity)
        
        return entities
    
    def extract_districts(self, text: str) -> List[ExtractedEntity]:
        """Extract AP district mentions"""
        entities = []
        
        for district in self.ap_districts:
            if self._case_insensitive_search(district, text):
                positions = self._find_positions(district, text)
                for start, end in positions:
                    entity = ExtractedEntity(
                        entity_type="district",
                        entity_value=district,
                        confidence=0.8,
                        span_start=start,
                        span_end=end,
                        context=self._extract_context(text, start, end)
                    )
                    entities.append(entity)
        
        return entities
    
    def extract_supersession_relations(self, text: str, doc_id: str) -> List[ExtractedRelation]:
        """Extract supersession relations"""
        relations = []
        
        for pattern in self.compiled_supersession:
            matches = pattern.finditer(text)
            for match in matches:
                relation = ExtractedRelation(
                    relation_type="supersedes",
                    source_doc_id=doc_id,
                    target_entity=match.group(0),  # The referenced document
                    confidence=0.9,
                    evidence=self._extract_context(text, match.start(), match.end()),
                    extraction_method="regex"
                )
                relations.append(relation)
        
        return relations
    
    def extract_implementation_relations(self, text: str, doc_id: str) -> List[ExtractedRelation]:
        """Extract implementation relations"""
        relations = []
        
        for pattern in self.compiled_implementation:
            matches = pattern.finditer(text)
            for match in matches:
                relation = ExtractedRelation(
                    relation_type="implements",
                    source_doc_id=doc_id,
                    target_entity=match.group(1) if match.groups() else match.group(0),
                    confidence=0.85,
                    evidence=self._extract_context(text, match.start(), match.end()),
                    extraction_method="regex"
                )
                relations.append(relation)
        
        return relations
    
    def extract_citation_relations(self, text: str, doc_id: str) -> List[ExtractedRelation]:
        """Extract citation relations"""
        relations = []
        
        # Legal citations
        for pattern in self.compiled_citation:
            matches = pattern.finditer(text)
            for match in matches:
                relation = ExtractedRelation(
                    relation_type="cites",
                    source_doc_id=doc_id,
                    target_entity=match.group(1) if match.groups() else match.group(0),
                    confidence=0.8,
                    evidence=self._extract_context(text, match.start(), match.end()),
                    extraction_method="regex"
                )
                relations.append(relation)
        
        return relations
    
    def extract_ner_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using NER (if spaCy is available)"""
        entities = []
        
        if not self.nlp:
            return entities
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Map spaCy entity types to our entity types
                entity_type_mapping = {
                    'ORG': 'organization',
                    'PERSON': 'person',
                    'GPE': 'location',
                    'DATE': 'date',
                    'MONEY': 'financial',
                    'PERCENT': 'metric',
                    'CARDINAL': 'number'
                }
                
                entity_type = entity_type_mapping.get(ent.label_, 'other')
                
                entity = ExtractedEntity(
                    entity_type=f"ner_{entity_type}",
                    entity_value=ent.text,
                    confidence=0.7,  # Lower confidence for NER
                    span_start=ent.start_char,
                    span_end=ent.end_char,
                    context=self._extract_context(text, ent.start_char, ent.end_char)
                )
                entities.append(entity)
        
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
        
        return entities
    
    def analyze_chunk(self, chunk: Dict[str, Any], bridge_topics: Optional[Dict[str, Any]] = None) -> ChunkAnalysis:
        """
        Analyze a single chunk and extract all entities and relations
        
        Args:
            chunk: Chunk dictionary with content, doc_id, etc.
            bridge_topics: Bridge topics for matching
            
        Returns:
            ChunkAnalysis with extracted entities and relations
        """
        chunk_id = chunk.get('chunk_id', '')
        doc_id = chunk.get('doc_id', '')
        content = chunk.get('content', '')
        
        if not content:
            return ChunkAnalysis(
                chunk_id=chunk_id,
                doc_id=doc_id,
                entities=[],
                relations=[],
                bridge_topic_matches=[],
                processing_notes=["Empty content"]
            )
        
        processing_notes = []
        
        # Extract entities
        entities = []
        
        try:
            # Legal references
            legal_entities = self.extract_legal_references(content)
            entities.extend(legal_entities)
            
            # GO references
            go_entities = self.extract_go_references(content)
            entities.extend(go_entities)
            
            # Schemes
            scheme_entities = self.extract_schemes(content)
            entities.extend(scheme_entities)
            
            # Metrics
            metric_entities = self.extract_metrics(content)
            entities.extend(metric_entities)
            
            # Districts
            district_entities = self.extract_districts(content)
            entities.extend(district_entities)
            
            # NER entities (if available)
            ner_entities = self.extract_ner_entities(content)
            entities.extend(ner_entities)
            
        except Exception as e:
            processing_notes.append(f"Entity extraction error: {e}")
            logger.error(f"Entity extraction failed for {chunk_id}: {e}")
        
        # Extract relations
        relations = []
        
        try:
            # Supersession relations
            supersession_relations = self.extract_supersession_relations(content, doc_id)
            relations.extend(supersession_relations)
            
            # Implementation relations
            implementation_relations = self.extract_implementation_relations(content, doc_id)
            relations.extend(implementation_relations)
            
            # Citation relations
            citation_relations = self.extract_citation_relations(content, doc_id)
            relations.extend(citation_relations)
            
        except Exception as e:
            processing_notes.append(f"Relation extraction error: {e}")
            logger.error(f"Relation extraction failed for {chunk_id}: {e}")
        
        # Match bridge topics
        bridge_topic_matches = []
        if bridge_topics:
            bridge_topic_matches = self.match_bridge_topics(content, entities, bridge_topics)
        
        return ChunkAnalysis(
            chunk_id=chunk_id,
            doc_id=doc_id,
            entities=entities,
            relations=relations,
            bridge_topic_matches=bridge_topic_matches,
            processing_notes=processing_notes
        )
    
    def match_bridge_topics(self, content: str, entities: List[ExtractedEntity], bridge_topics: Dict[str, Any]) -> List[str]:
        """Match chunk to bridge topics based on keywords and entities"""
        matches = []
        
        content_lower = content.lower()
        
        for topic_id, topic_data in bridge_topics.items():
            keywords = topic_data.get('keywords', [])
            
            # Check keyword matches
            keyword_matches = 0
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    keyword_matches += 1
            
            # Check entity matches
            entity_matches = 0
            entity_values = [e.entity_value.lower() for e in entities]
            
            for keyword in keywords:
                if keyword.lower() in entity_values:
                    entity_matches += 1
            
            # Scoring: keyword matches + entity matches
            total_matches = keyword_matches + entity_matches
            
            # Threshold for matching (at least 2 keywords or 1 keyword + 1 entity)
            if total_matches >= 2 or (keyword_matches >= 1 and entity_matches >= 1):
                matches.append(topic_id)
        
        return matches
    
    def _case_insensitive_search(self, pattern: str, text: str) -> bool:
        """Case-insensitive search"""
        return pattern.lower() in text.lower()
    
    def _find_positions(self, pattern: str, text: str) -> List[Tuple[int, int]]:
        """Find all positions of pattern in text"""
        positions = []
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        
        start = 0
        while True:
            pos = text_lower.find(pattern_lower, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(pattern)))
            start = pos + 1
        
        return positions
    
    def _extract_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Extract context around the matched span"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end].strip()
    
    def get_extraction_stats(self, analyses: List[ChunkAnalysis]) -> Dict[str, Any]:
        """Get statistics about extraction results"""
        total_chunks = len(analyses)
        total_entities = sum(len(a.entities) for a in analyses)
        total_relations = sum(len(a.relations) for a in analyses)
        total_bridge_matches = sum(len(a.bridge_topic_matches) for a in analyses)
        
        # Entity type distribution
        entity_types = {}
        for analysis in analyses:
            for entity in analysis.entities:
                entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
        
        # Relation type distribution
        relation_types = {}
        for analysis in analyses:
            for relation in analysis.relations:
                relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'total_entities': total_entities,
            'total_relations': total_relations,
            'total_bridge_matches': total_bridge_matches,
            'avg_entities_per_chunk': total_entities / total_chunks if total_chunks > 0 else 0,
            'avg_relations_per_chunk': total_relations / total_chunks if total_chunks > 0 else 0,
            'avg_bridge_matches_per_chunk': total_bridge_matches / total_chunks if total_chunks > 0 else 0,
            'entity_type_distribution': entity_types,
            'relation_type_distribution': relation_types
        }




