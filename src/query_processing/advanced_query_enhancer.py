"""
SOTA Advanced Query Enhancement System

This module implements sophisticated query enhancement techniques that leverage:
1. Semantic chunking for better context understanding
2. Bridge table relationships for query expansion
3. Intent-based routing optimization
4. Multi-step reasoning for complex queries

Key Features:
1. Multi-level query understanding (syntactic, semantic, pragmatic)
2. Relationship-aware query expansion
3. Temporal reasoning (as of date X, latest version)
4. Comparative query handling (compare X vs Y)
5. Confidence-calibrated query routing
6. Context preservation across query reformulations
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, date
import spacy
from collections import defaultdict, Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph.bridge_table_builder import BridgeTableBuilder, RelationshipType, EntityType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryIntent(Enum):
    """Enhanced query intent categories"""
    LEGAL_INTERPRETATION = "legal_interpretation"
    POLICY_IMPLEMENTATION = "policy_implementation"
    FACTUAL_LOOKUP = "factual_lookup"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TEMPORAL_QUERY = "temporal_query"
    PROCEDURAL_GUIDANCE = "procedural_guidance"
    COMPLIANCE_CHECK = "compliance_check"
    STATUS_INQUIRY = "status_inquiry"
    DEFINITION_REQUEST = "definition_request"
    RELATIONSHIP_EXPLORATION = "relationship_exploration"


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"          # Single entity, direct lookup
    MODERATE = "moderate"      # Multiple entities, some reasoning
    COMPLEX = "complex"        # Cross-domain, temporal, comparative
    MULTI_STEP = "multi_step"  # Requires decomposition and synthesis


class TemporalScope(Enum):
    """Temporal scope of queries"""
    CURRENT = "current"        # Current/latest version
    HISTORICAL = "historical"  # As of specific date
    COMPARATIVE = "comparative" # Changes over time
    FUTURE = "future"          # Planned changes


@dataclass
class QueryEntity:
    """Enhanced entity with relationship context"""
    entity_id: str
    entity_type: EntityType
    entity_value: str
    confidence: float
    context: str
    relationships: List[Dict[str, Any]] = None
    temporal_context: Optional[str] = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []


@dataclass
class EnhancedQuery:
    """Comprehensive query representation"""
    original_query: str
    normalized_query: str
    
    # Intent analysis
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent]
    intent_confidence: float
    
    # Complexity assessment
    complexity: QueryComplexity
    reasoning_steps: List[str]
    
    # Entity analysis
    entities: List[QueryEntity]
    entity_relationships: Dict[str, List[Dict[str, Any]]]
    
    # Temporal analysis
    temporal_scope: TemporalScope
    temporal_expressions: List[str]
    reference_date: Optional[date]
    
    # Query expansion
    expanded_terms: List[str]
    semantic_expansions: List[str]
    relationship_expansions: List[str]
    
    # Routing hints
    suggested_agents: List[str]
    execution_strategy: str
    confidence_threshold: float
    
    # Context preservation
    conversation_context: Optional[str] = None
    previous_queries: List[str] = None
    
    def __post_init__(self):
        if self.previous_queries is None:
            self.previous_queries = []


class AdvancedQueryEnhancer:
    """
    Advanced query enhancement system that provides sophisticated
    query understanding, expansion, and routing optimization.
    """
    
    # Intent classification patterns
    INTENT_PATTERNS = {
        QueryIntent.LEGAL_INTERPRETATION: [
            r'(?:what\s+(?:is|does)|meaning\s+of|interpret|interpretation|legal\s+(?:meaning|significance))',
            r'(?:section|article|clause|provision).*?(?:means?|says?|states?)',
            r'(?:legal|statutory)\s+(?:definition|meaning|interpretation)',
            r'(?:under\s+the\s+law|legally|as\s+per\s+law)'
        ],
        
        QueryIntent.COMPARATIVE_ANALYSIS: [
            r'(?:compar[ei]|versus|vs\.?|difference|differ|between)',
            r'(?:what\s+(?:is|are)\s+the\s+difference|how\s+(?:do|does).*?differ)',
            r'(?:advantages?|disadvantages?|pros?|cons?|benefits?)\s+(?:of|over)',
            r'(?:better|worse|superior|inferior|more|less)\s+than'
        ],
        
        QueryIntent.TEMPORAL_QUERY: [
            r'(?:when|what\s+date|timeline|chronology|history)',
            r'(?:before|after|since|until|as\s+of|effective\s+from)',
            r'(?:latest|recent|current|updated|new|old|previous)',
            r'(?:changes?|amendments?|modifications?|updates?)',
            r'(?:superseded?|replaced?|cancelled?)'
        ],
        
        QueryIntent.PROCEDURAL_GUIDANCE: [
            r'(?:how\s+to|procedure|process|steps?|method)',
            r'(?:apply|application|implementation|execute)',
            r'(?:guidelines?|instructions?|directions?)',
            r'(?:what\s+(?:should|must|need)\s+(?:i|we|one))'
        ],
        
        QueryIntent.STATUS_INQUIRY: [
            r'(?:status|current\s+(?:state|position)|where\s+(?:do\s+)?we\s+stand)',
            r'(?:is\s+(?:it|this|that).*?(?:active|valid|effective|applicable))',
            r'(?:still\s+(?:valid|effective|applicable|in\s+force))',
            r'(?:has\s+(?:it|this|that)\s+been.*?(?:superseded|cancelled|amended))'
        ],
        
        QueryIntent.COMPLIANCE_CHECK: [
            r'(?:complian[ct]e|conform|accordance|adherence)',
            r'(?:mandatory|required|obligatory|compulsory)',
            r'(?:violation|breach|non[\-\s]?compliance)',
            r'(?:penalty|punishment|consequences?|sanctions?)'
        ]
    }
    
    # Temporal expression patterns
    TEMPORAL_PATTERNS = {
        'absolute_date': [
            r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
            r'(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4})',
            r'([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})'
        ],
        'relative_date': [
            r'(latest|recent|current|new|updated)',
            r'(as\s+of\s+(?:today|now|\d{4}))',
            r'(before|after|since|until)\s+(\d{4})',
            r'(effective\s+from|w\.?e\.?f\.?)\s+(.*?)',
            r'(superseded|replaced|cancelled)\s+(?:on|in|by)\s+(.*?)'
        ],
        'temporal_scope': [
            r'(historical|past|previous|earlier|former)',
            r'(current|present|now|today|latest)',
            r'(future|upcoming|planned|proposed)',
            r'(all\s+versions?|complete\s+history|chronological)'
        ]
    }
    
    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        QueryComplexity.SIMPLE: [
            r'(?:what\s+is|who\s+is|where\s+is|when\s+is)',
            r'(?:definition|meaning)\s+of\s+\w+',
            r'^\w+\s+(?:act|rule|section|go)\s+\d+$'
        ],
        
        QueryComplexity.MODERATE: [
            r'(?:how|why|which|whose)',
            r'(?:list|enumerate|identify)\s+(?:all|the)',
            r'(?:requirements?|conditions?|criteria)',
            r'(?:eligible|eligibility|qualification)'
        ],
        
        QueryComplexity.COMPLEX: [
            r'(?:analyze|analysis|evaluate|assessment)',
            r'(?:impact|effect|consequence|implication)',
            r'(?:relationship|connection|correlation)',
            r'(?:across|between|among).*?(?:and|or)',
            r'(?:multiple|various|different|several)'
        ],
        
        QueryComplexity.MULTI_STEP: [
            r'(?:first.*?then|step\s+by\s+step|sequential)',
            r'(?:if.*?then|in\s+case\s+of|depending\s+on)',
            r'(?:breakdown|decompose|step[\-\s]?wise)',
            r'(?:comprehensive|detailed|thorough)\s+(?:analysis|review)'
        ]
    }
    
    def __init__(self, bridge_table_path: str = "./bridge_table.db"):
        """
        Initialize advanced query enhancer.
        
        Args:
            bridge_table_path: Path to bridge table database
        """
        self.bridge_builder = BridgeTableBuilder(bridge_table_path)
        
        # Try to load spaCy model
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded for advanced query analysis")
        except OSError:
            logger.warning("SpaCy model not found. Using pattern-based analysis only.")
        
        # Semantic expansion cache
        self.semantic_cache = {}
        
        logger.info("Advanced Query Enhancer initialized")
    
    def enhance_query(
        self, 
        query: str,
        conversation_context: Optional[str] = None,
        previous_queries: Optional[List[str]] = None
    ) -> EnhancedQuery:
        """
        Main entry point for query enhancement.
        
        Args:
            query: Original user query
            conversation_context: Previous conversation context
            previous_queries: List of previous queries in session
            
        Returns:
            Comprehensive enhanced query object
        """
        logger.info(f"Enhancing query: {query}")
        
        # Step 1: Normalize and preprocess
        normalized_query = self._normalize_query(query)
        
        # Step 2: Intent classification
        primary_intent, secondary_intents, intent_confidence = self._classify_intent(normalized_query)
        
        # Step 3: Complexity assessment
        complexity, reasoning_steps = self._assess_complexity(normalized_query, primary_intent)
        
        # Step 4: Entity extraction with relationships
        entities, entity_relationships = self._extract_entities_with_relationships(normalized_query)
        
        # Step 5: Temporal analysis
        temporal_scope, temporal_expressions, reference_date = self._analyze_temporal_context(normalized_query)
        
        # Step 6: Query expansion
        expanded_terms, semantic_expansions, relationship_expansions = self._expand_query(
            normalized_query, entities, entity_relationships, primary_intent
        )
        
        # Step 7: Routing optimization
        suggested_agents, execution_strategy, confidence_threshold = self._optimize_routing(
            primary_intent, complexity, entities, temporal_scope
        )
        
        # Step 8: Build enhanced query
        enhanced_query = EnhancedQuery(
            original_query=query,
            normalized_query=normalized_query,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            intent_confidence=intent_confidence,
            complexity=complexity,
            reasoning_steps=reasoning_steps,
            entities=entities,
            entity_relationships=entity_relationships,
            temporal_scope=temporal_scope,
            temporal_expressions=temporal_expressions,
            reference_date=reference_date,
            expanded_terms=expanded_terms,
            semantic_expansions=semantic_expansions,
            relationship_expansions=relationship_expansions,
            suggested_agents=suggested_agents,
            execution_strategy=execution_strategy,
            confidence_threshold=confidence_threshold,
            conversation_context=conversation_context,
            previous_queries=previous_queries or []
        )
        
        logger.info(f"Query enhanced: Intent={primary_intent.value}, Complexity={complexity.value}, Entities={len(entities)}")
        
        return enhanced_query
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text for better processing."""
        # Basic normalization
        normalized = query.strip().lower()
        
        # Expand common abbreviations
        abbreviations = {
            r'\brte\b': 'right to education',
            r'\bgo\b': 'government order',
            r'\bptr\b': 'pupil teacher ratio',
            r'\bsse\b': 'sarva shiksha abhiyan',
            r'\bmid day meal\b': 'midday meal',
            r'\bap\b': 'andhra pradesh'
        }
        
        for abbrev, expansion in abbreviations.items():
            normalized = re.sub(abbrev, expansion, normalized, flags=re.IGNORECASE)
        
        # Clean up whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _classify_intent(self, query: str) -> Tuple[QueryIntent, List[QueryIntent], float]:
        """Classify query intent with confidence scoring."""
        intent_scores = defaultdict(float)
        
        # Pattern-based classification
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    intent_scores[intent] += 1.0
        
        # NLP-based enhancement if available
        if self.nlp:
            doc = self.nlp(query)
            
            # Verb analysis for intent
            for token in doc:
                if token.pos_ == 'VERB':
                    verb = token.lemma_.lower()
                    
                    if verb in ['compare', 'contrast', 'differ']:
                        intent_scores[QueryIntent.COMPARATIVE_ANALYSIS] += 0.5
                    elif verb in ['interpret', 'mean', 'define']:
                        intent_scores[QueryIntent.LEGAL_INTERPRETATION] += 0.5
                    elif verb in ['apply', 'implement', 'execute']:
                        intent_scores[QueryIntent.PROCEDURAL_GUIDANCE] += 0.5
                    elif verb in ['check', 'verify', 'ensure']:
                        intent_scores[QueryIntent.COMPLIANCE_CHECK] += 0.5
        
        # Default to factual lookup if no strong signals
        if not intent_scores:
            intent_scores[QueryIntent.FACTUAL_LOOKUP] = 0.5
        
        # Sort by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_intent = sorted_intents[0][0]
        primary_confidence = sorted_intents[0][1]
        
        # Secondary intents (score > 0.3)
        secondary_intents = [intent for intent, score in sorted_intents[1:] if score > 0.3]
        
        # Normalize confidence
        total_score = sum(intent_scores.values())
        normalized_confidence = min(primary_confidence / total_score, 1.0) if total_score > 0 else 0.5
        
        return primary_intent, secondary_intents, normalized_confidence
    
    def _assess_complexity(self, query: str, intent: QueryIntent) -> Tuple[QueryComplexity, List[str]]:
        """Assess query complexity and required reasoning steps."""
        complexity_scores = defaultdict(float)
        reasoning_steps = []
        
        # Pattern-based complexity assessment
        for complexity, patterns in self.COMPLEXITY_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    complexity_scores[complexity] += 1.0
        
        # Intent-based complexity adjustment
        if intent in [QueryIntent.COMPARATIVE_ANALYSIS, QueryIntent.RELATIONSHIP_EXPLORATION]:
            complexity_scores[QueryComplexity.COMPLEX] += 1.0
            reasoning_steps.append("Multi-entity comparison required")
        
        if intent == QueryIntent.TEMPORAL_QUERY:
            complexity_scores[QueryComplexity.MODERATE] += 0.5
            reasoning_steps.append("Temporal reasoning required")
        
        # Query structure analysis
        question_words = len(re.findall(r'\b(?:what|who|when|where|why|how|which)\b', query, re.IGNORECASE))
        if question_words > 1:
            complexity_scores[QueryComplexity.COMPLEX] += 0.5
            reasoning_steps.append("Multi-part question decomposition")
        
        # Conjunction analysis
        conjunctions = len(re.findall(r'\b(?:and|or|but|however|although)\b', query, re.IGNORECASE))
        if conjunctions > 0:
            complexity_scores[QueryComplexity.MODERATE] += 0.3 * conjunctions
            reasoning_steps.append("Multiple conditions or alternatives")
        
        # Default complexity
        if not complexity_scores:
            complexity_scores[QueryComplexity.SIMPLE] = 1.0
            reasoning_steps.append("Direct information lookup")
        
        # Select highest scoring complexity
        primary_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0]
        
        return primary_complexity, reasoning_steps
    
    def _extract_entities_with_relationships(
        self, 
        query: str
    ) -> Tuple[List[QueryEntity], Dict[str, List[Dict[str, Any]]]]:
        """Extract entities and their relationships from query."""
        entities = []
        entity_relationships = defaultdict(list)
        
        # Extract entities using bridge table patterns
        for entity_type, patterns in self.bridge_builder.ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                
                for match in matches:
                    entity_value = match.group(1).strip()
                    entity_id = f"{entity_type.value}_{entity_value.replace(' ', '_').replace('.', '')}"
                    
                    # Calculate confidence
                    confidence = self._calculate_entity_confidence(match, query)
                    
                    # Get context
                    context = query[max(0, match.start()-20):match.end()+20]
                    
                    # Create entity
                    entity = QueryEntity(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        entity_value=entity_value,
                        confidence=confidence,
                        context=context
                    )
                    
                    # Get relationships from bridge table
                    relationships = self.bridge_builder.get_entity_relationships(
                        entity_id, 
                        include_confidence=True
                    )
                    
                    entity.relationships = relationships
                    entity_relationships[entity_id] = relationships
                    
                    entities.append(entity)
        
        # Remove duplicates
        unique_entities = []
        seen_ids = set()
        
        for entity in entities:
            if entity.entity_id not in seen_ids:
                unique_entities.append(entity)
                seen_ids.add(entity.entity_id)
        
        return unique_entities, dict(entity_relationships)
    
    def _calculate_entity_confidence(self, match: re.Match, query: str) -> float:
        """Calculate confidence for entity extraction."""
        confidence = 0.8  # Base confidence
        
        # Position-based confidence (entities at start/end often more important)
        position = match.start() / len(query)
        if position < 0.2 or position > 0.8:
            confidence += 0.1
        
        # Context-based confidence
        context = query[max(0, match.start()-10):match.end()+10].lower()
        
        if any(word in context for word in ['section', 'act', 'rule', 'order']):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _analyze_temporal_context(
        self, 
        query: str
    ) -> Tuple[TemporalScope, List[str], Optional[date]]:
        """Analyze temporal aspects of the query."""
        temporal_expressions = []
        reference_date = None
        
        # Extract temporal expressions
        for pattern_type, patterns in self.TEMPORAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        temporal_expressions.extend([m for m in match if m])
                    else:
                        temporal_expressions.append(match)
        
        # Determine temporal scope
        temporal_scope = TemporalScope.CURRENT  # Default
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['historical', 'past', 'previous', 'earlier', 'before']):
            temporal_scope = TemporalScope.HISTORICAL
        elif any(word in query_lower for word in ['changes', 'evolution', 'over time', 'timeline']):
            temporal_scope = TemporalScope.COMPARATIVE
        elif any(word in query_lower for word in ['future', 'upcoming', 'planned', 'proposed']):
            temporal_scope = TemporalScope.FUTURE
        elif any(word in query_lower for word in ['latest', 'current', 'recent', 'now', 'today']):
            temporal_scope = TemporalScope.CURRENT
        
        # Parse reference date if available
        date_patterns = [
            r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
            r'(\d{4})',
            r'as\s+of\s+(\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query)
            if match:
                date_str = match.group(1)
                try:
                    if len(date_str) == 4:  # Year only
                        reference_date = date(int(date_str), 1, 1)
                    else:
                        # Try to parse date
                        for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y']:
                            try:
                                reference_date = datetime.strptime(date_str, fmt).date()
                                break
                            except ValueError:
                                continue
                except ValueError:
                    pass
                break
        
        return temporal_scope, temporal_expressions, reference_date
    
    def _expand_query(
        self, 
        query: str,
        entities: List[QueryEntity],
        entity_relationships: Dict[str, List[Dict[str, Any]]],
        intent: QueryIntent
    ) -> Tuple[List[str], List[str], List[str]]:
        """Expand query with synonyms, semantic terms, and relationships."""
        
        # 1. Basic term expansion
        expanded_terms = []
        
        # Synonyms for common terms
        synonyms = {
            'teacher': ['educator', 'instructor', 'faculty'],
            'student': ['pupil', 'learner', 'child'],
            'school': ['institution', 'educational institution'],
            'education': ['learning', 'instruction', 'schooling'],
            'policy': ['directive', 'guideline', 'regulation'],
            'implementation': ['execution', 'enforcement', 'application']
        }
        
        for term, syns in synonyms.items():
            if term in query.lower():
                expanded_terms.extend(syns)
        
        # 2. Semantic expansions based on intent
        semantic_expansions = []
        
        if intent == QueryIntent.LEGAL_INTERPRETATION:
            semantic_expansions.extend(['statutory', 'legal provision', 'constitutional', 'mandate'])
        elif intent == QueryIntent.PROCEDURAL_GUIDANCE:
            semantic_expansions.extend(['procedure', 'process', 'steps', 'guidelines', 'method'])
        elif intent == QueryIntent.COMPLIANCE_CHECK:
            semantic_expansions.extend(['compliance', 'adherence', 'conformity', 'requirement'])
        
        # 3. Relationship-based expansions
        relationship_expansions = []
        
        for entity in entities:
            if entity.relationships:
                for rel in entity.relationships[:3]:  # Top 3 relationships
                    # Add related entity values
                    related_value = rel.get('target_value') or rel.get('source_value')
                    if related_value and related_value not in query:
                        relationship_expansions.append(related_value)
                    
                    # Add relationship context
                    rel_type = rel.get('relationship_type', '')
                    if rel_type == 'supersedes':
                        relationship_expansions.append('superseded by')
                    elif rel_type == 'implements':
                        relationship_expansions.append('implementation of')
                    elif rel_type == 'references':
                        relationship_expansions.append('cross reference')
        
        return expanded_terms, semantic_expansions, relationship_expansions
    
    def _optimize_routing(
        self,
        intent: QueryIntent,
        complexity: QueryComplexity,
        entities: List[QueryEntity],
        temporal_scope: TemporalScope
    ) -> Tuple[List[str], str, float]:
        """Optimize agent routing based on query analysis."""
        
        suggested_agents = []
        execution_strategy = "sequential"
        confidence_threshold = 0.5
        
        # Intent-based agent selection
        intent_agent_mapping = {
            QueryIntent.LEGAL_INTERPRETATION: ['legal_agent'],
            QueryIntent.POLICY_IMPLEMENTATION: ['go_agent', 'legal_agent'],
            QueryIntent.PROCEDURAL_GUIDANCE: ['go_agent'],
            QueryIntent.COMPARATIVE_ANALYSIS: ['legal_agent', 'go_agent', 'data_agent'],
            QueryIntent.TEMPORAL_QUERY: ['legal_agent', 'go_agent'],
            QueryIntent.COMPLIANCE_CHECK: ['legal_agent'],
            QueryIntent.STATUS_INQUIRY: ['go_agent'],
            QueryIntent.FACTUAL_LOOKUP: ['general_agent']
        }
        
        base_agents = intent_agent_mapping.get(intent, ['general_agent'])
        suggested_agents.extend(base_agents)
        
        # Entity-based agent enhancement
        for entity in entities:
            if entity.entity_type == EntityType.GOVERNMENT_ORDER:
                if 'go_agent' not in suggested_agents:
                    suggested_agents.append('go_agent')
            elif entity.entity_type == EntityType.LEGAL_SECTION:
                if 'legal_agent' not in suggested_agents:
                    suggested_agents.append('legal_agent')
            elif entity.entity_type == EntityType.DISTRICT:
                if 'data_agent' not in suggested_agents:
                    suggested_agents.append('data_agent')
        
        # Complexity-based strategy adjustment
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_STEP]:
            execution_strategy = "parallel"
            confidence_threshold = 0.3  # Lower threshold for complex queries
            
            # Ensure comprehensive coverage for complex queries
            if len(suggested_agents) < 2:
                suggested_agents.append('general_agent')
        
        # Temporal scope adjustment
        if temporal_scope == TemporalScope.HISTORICAL:
            confidence_threshold = 0.4  # Slightly higher for historical queries
        elif temporal_scope == TemporalScope.COMPARATIVE:
            execution_strategy = "parallel"
            if 'data_agent' not in suggested_agents:
                suggested_agents.append('data_agent')
        
        # Remove duplicates while preserving order
        unique_agents = []
        for agent in suggested_agents:
            if agent not in unique_agents:
                unique_agents.append(agent)
        
        return unique_agents, execution_strategy, confidence_threshold
    
    def get_query_expansion_context(self, enhanced_query: EnhancedQuery) -> str:
        """Generate context string for query expansion."""
        context_parts = []
        
        # Add relationship context
        if enhanced_query.relationship_expansions:
            context_parts.append(f"Related concepts: {', '.join(enhanced_query.relationship_expansions[:5])}")
        
        # Add temporal context
        if enhanced_query.temporal_scope != TemporalScope.CURRENT:
            context_parts.append(f"Temporal scope: {enhanced_query.temporal_scope.value}")
        
        # Add entity relationships
        for entity in enhanced_query.entities[:3]:  # Top 3 entities
            if entity.relationships:
                rel_summary = f"{entity.entity_value} relationships: {len(entity.relationships)}"
                context_parts.append(rel_summary)
        
        return " | ".join(context_parts)
    
    def decompose_complex_query(self, enhanced_query: EnhancedQuery) -> List[str]:
        """Decompose complex queries into simpler sub-queries."""
        if enhanced_query.complexity not in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_STEP]:
            return [enhanced_query.original_query]
        
        sub_queries = []
        
        # For comparative queries
        if enhanced_query.primary_intent == QueryIntent.COMPARATIVE_ANALYSIS:
            entities_mentioned = [e.entity_value for e in enhanced_query.entities]
            if len(entities_mentioned) >= 2:
                for entity in entities_mentioned:
                    sub_queries.append(f"What is {entity}?")
                sub_queries.append(f"Compare {' and '.join(entities_mentioned)}")
        
        # For temporal queries
        elif enhanced_query.temporal_scope == TemporalScope.COMPARATIVE:
            if enhanced_query.entities:
                entity = enhanced_query.entities[0].entity_value
                sub_queries.extend([
                    f"What is the current status of {entity}?",
                    f"What was the historical status of {entity}?",
                    f"What changes have occurred in {entity}?"
                ])
        
        # For multi-step procedural queries
        elif enhanced_query.primary_intent == QueryIntent.PROCEDURAL_GUIDANCE:
            if enhanced_query.entities:
                entity = enhanced_query.entities[0].entity_value
                sub_queries.extend([
                    f"What are the requirements for {entity}?",
                    f"What is the procedure for {entity}?",
                    f"What are the steps to implement {entity}?"
                ])
        
        # Fallback: return original query
        if not sub_queries:
            sub_queries = [enhanced_query.original_query]
        
        return sub_queries


# Convenience functions

def enhance_query_advanced(
    query: str,
    bridge_table_path: str = "./bridge_table.db",
    conversation_context: Optional[str] = None
) -> EnhancedQuery:
    """Convenience function for query enhancement."""
    
    enhancer = AdvancedQueryEnhancer(bridge_table_path)
    return enhancer.enhance_query(query, conversation_context)


if __name__ == "__main__":
    print("Advanced Query Enhancer module loaded successfully")
    
    # Test with sample queries
    enhancer = AdvancedQueryEnhancer("./test_bridge.db")
    
    test_queries = [
        "What is Section 12(1)(c) of RTE Act?",
        "Compare Nadu-Nedu scheme with teacher qualification requirements",
        "What changes have been made to GO 123 since 2023?",
        "How to implement midday meal scheme in government schools?"
    ]
    
    for query in test_queries:
        enhanced = enhancer.enhance_query(query)
        print(f"Query: {query}")
        print(f"  Intent: {enhanced.primary_intent.value}")
        print(f"  Complexity: {enhanced.complexity.value}")
        print(f"  Entities: {len(enhanced.entities)}")
        print(f"  Suggested agents: {enhanced.suggested_agents}")
        print(f"  Execution: {enhanced.execution_strategy}")
        print()