"""
Query Processing Pipeline Orchestrator

Coordinates all query processing stages:
1. Normalization (spelling, acronyms, canonicalization)
2. Entity Extraction (districts, schemes, metrics, dates, legal refs)
3. Intent Classification (primary + secondary intents)
4. Query Expansion (synonyms, legal terms, variations)
5. Context Injection (conversation history, entity resolution)

Produces structured payload for downstream agent router.
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .normalizer import QueryNormalizer, process_query_normalization
from .entity_extractor import QueryEntityExtractor, process_entity_extraction
from .intent_classifier import QueryIntentClassifier, process_intent_classification
from .query_expander import QueryExpander, process_query_expansion
from .context_injector import ContextInjector, process_context_injection
from .validator import EntityValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Complete processed query package"""
    # Input
    original_query: str
    session_id: str
    
    # Normalization
    normalized_query: str
    normalization_transforms: Dict
    
    # Entities
    entities: Dict[str, List]
    entity_summary: str
    
    # Intent
    primary_intent: str
    secondary_intents: List[str]
    intent_confidence: float
    query_complexity: str
    
    # Expansion
    query_expansions: List[Dict]
    
    # Context
    has_context: bool
    context_summary: str
    resolved_query: str
    
    # Routing hints
    suggested_verticals: List[str]
    query_type: str
    
    # Metadata
    processing_time_ms: float
    timestamp: str


class QueryProcessingPipeline:
    """
    Orchestrates complete query processing workflow.
    """
    
    def __init__(self, dictionaries_path: str = "data/dictionaries",
                 enable_validation: bool = True):
        """
        Initialize pipeline with all components.
        
        Args:
            dictionaries_path: Path to dictionaries directory
            enable_validation: Whether to validate extracted entities
        """
        self.normalizer = QueryNormalizer(dictionaries_path)
        self.entity_extractor = QueryEntityExtractor(dictionaries_path)
        self.intent_classifier = QueryIntentClassifier()
        self.query_expander = QueryExpander()
        self.context_injector = ContextInjector()
        self.validator = EntityValidator() if enable_validation else None
        self.enable_validation = enable_validation
        
        logger.info(f"QueryProcessingPipeline initialized (validation={'ON' if enable_validation else 'OFF'})")
    
    def process(self, query: str, 
                session_id: str = "default",
                conversation_history: Optional[List[str]] = None,
                expand_query: bool = True,
                inject_context: bool = True) -> ProcessedQuery:
        """
        Process query through complete pipeline.
        
        Args:
            query: Raw user query
            session_id: Session identifier for context
            conversation_history: Optional conversation history (legacy)
            expand_query: Whether to generate query expansions
            inject_context: Whether to inject conversation context
            
        Returns:
            ProcessedQuery with all extracted information
        """
        start_time = datetime.now()
        
        logger.info(f"Processing query: '{query}'")
        
        # Stage 1: Normalization
        norm_result = self.normalizer.normalize(query)
        normalized_query = norm_result["normalized"]
        
        logger.debug(f"Normalized: '{normalized_query}'")
        
        # Stage 2: Entity Extraction
        entities = self.entity_extractor.extract(normalized_query)
        
        # Stage 2.5: Entity Validation (optional)
        validation_result = None
        if self.enable_validation and self.validator:
            validation_result = self.validator.validate(entities, normalized_query)
            entities = validation_result.validated_entities
            
            if not validation_result.is_valid:
                logger.warning(f"Validation issues: {self.validator.get_validation_summary(validation_result)}")
        
        entity_summary = self._summarize_entities(entities)
        
        logger.debug(f"Extracted entities: {entity_summary}")
        
        # Stage 3: Intent Classification
        intent_result = self.intent_classifier.classify(normalized_query, entities)
        primary_intent = intent_result["primary"]["name"]
        secondary_intents = [i["name"] for i in intent_result["secondary"]]
        
        logger.debug(f"Classified intent: primary={primary_intent}, secondary={secondary_intents}")
        
        # Stage 4: Query Expansion
        query_expansions = []
        if expand_query:
            expansion_context = {
                "entities": entities,
                "intent": intent_result
            }
            expansions = self.query_expander.expand_with_context(normalized_query, expansion_context)
            query_expansions = [
                {
                    "text": exp.text,
                    "type": exp.expansion_type,
                    "weight": exp.weight
                }
                for exp in expansions
            ]
            
            logger.debug(f"Generated {len(query_expansions)} expansions")
        
        # Stage 5: Context Injection
        has_context = False
        context_summary = "No context"
        resolved_query = normalized_query
        
        if inject_context:
            context_result = self.context_injector.inject_context(
                normalized_query,
                session_id=session_id,
                entities=entities,
                intent=intent_result
            )
            has_context = context_result["has_context"]
            context_summary = context_result["context_summary"]
            resolved_query = context_result["resolved_query"]
            
            logger.debug(f"Context injected: {context_summary}")
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Build final package
        processed = ProcessedQuery(
            original_query=query,
            session_id=session_id,
            normalized_query=normalized_query,
            normalization_transforms=norm_result["transformations"],
            entities=self.entity_extractor.to_dict(entities),
            entity_summary=entity_summary,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            intent_confidence=intent_result["primary"]["confidence"],
            query_complexity=intent_result["complexity"],
            query_expansions=query_expansions,
            has_context=has_context,
            context_summary=context_summary,
            resolved_query=resolved_query,
            suggested_verticals=intent_result["suggested_verticals"],
            query_type=intent_result["query_type"],
            processing_time_ms=processing_time_ms,
            timestamp=end_time.isoformat()
        )
        
        logger.info(f"Query processed in {processing_time_ms:.2f}ms")
        return processed
    
    def _summarize_entities(self, entities: Dict[str, List]) -> str:
        """Create human-readable entity summary"""
        summary_parts = []
        
        for entity_type, entity_list in entities.items():
            if entity_list:
                count = len(entity_list)
                if count == 1:
                    entity = entity_list[0]
                    canonical = entity.canonical if hasattr(entity, 'canonical') else str(entity)
                    summary_parts.append(f"{entity_type}: {canonical}")
                else:
                    summary_parts.append(f"{entity_type}: {count} items")
        
        return "; ".join(summary_parts) if summary_parts else "No entities"
    
    def process_batch(self, queries: List[str], 
                     session_id: str = "default") -> List[ProcessedQuery]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of queries to process
            session_id: Session identifier
            
        Returns:
            List of processed queries
        """
        results = []
        
        for query in queries:
            try:
                result = self.process(query, session_id=session_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                # Return minimal result
                results.append(ProcessedQuery(
                    original_query=query,
                    session_id=session_id,
                    normalized_query=query,
                    normalization_transforms={},
                    entities={},
                    entity_summary="Error",
                    primary_intent="unknown",
                    secondary_intents=[],
                    intent_confidence=0.0,
                    query_complexity="unknown",
                    query_expansions=[],
                    has_context=False,
                    context_summary="Error",
                    resolved_query=query,
                    suggested_verticals=[],
                    query_type="unknown",
                    processing_time_ms=0.0,
                    timestamp=datetime.now().isoformat()
                ))
        
        return results
    
    def to_dict(self, processed: ProcessedQuery) -> Dict:
        """Convert ProcessedQuery to dictionary"""
        return asdict(processed)
    
    def to_json(self, processed: ProcessedQuery, indent: int = 2) -> str:
        """Convert ProcessedQuery to JSON string"""
        return json.dumps(asdict(processed), indent=indent)


# Convenience function for backwards compatibility
def process_query(query: str, dictionaries: dict = None, conversation_history: list = None) -> dict:
    """Process query through all stages (backwards compatible)"""
    pipeline = QueryProcessingPipeline()
    processed = pipeline.process(query)
    
    # Convert to simpler format
    return {
        "original": processed.original_query,
        "normalized": processed.normalized_query,
        "entities": processed.entities,
        "intent": {
            "primary": processed.primary_intent,
            "secondary": processed.secondary_intents
        },
        "expanded": [exp["text"] for exp in processed.query_expansions]
    }


# Main API function
def process_query_pipeline(query: str,
                          session_id: str = "default",
                          expand_query: bool = True,
                          inject_context: bool = True) -> Dict[str, any]:
    """
    Process query through complete pipeline.
    
    Args:
        query: Raw user query
        session_id: Session identifier
        expand_query: Whether to generate expansions
        inject_context: Whether to inject context
        
    Returns:
        Complete processed query as dictionary
    """
    pipeline = QueryProcessingPipeline()
    processed = pipeline.process(
        query,
        session_id=session_id,
        expand_query=expand_query,
        inject_context=inject_context
    )
    return pipeline.to_dict(processed)


# Export key classes
__all__ = [
    'QueryProcessingPipeline',
    'ProcessedQuery',
    'process_query',
    'process_query_pipeline'
]
