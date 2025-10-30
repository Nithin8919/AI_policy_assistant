"""
Enhanced Agent Router with Vector Database Integration and Query Orchestration
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType
from src.query_processing.pipeline import QueryProcessingPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

class QueryComplexity(Enum):
    SIMPLE = "simple"      # Single vertical, direct lookup
    MODERATE = "moderate"  # Multi-vertical or complex entities
    COMPLEX = "complex"    # Cross-vertical synthesis required

@dataclass
class AgentSelection:
    """Agent selection result"""
    agent_name: str
    doc_type: DocumentType
    confidence: float
    reasoning: str

@dataclass
class RetrievalResult:
    """Result from agent retrieval"""
    agent_name: str
    doc_type: str
    chunks: List[Dict[str, Any]]
    total_results: int
    processing_time: float
    confidence_score: float

@dataclass
class RouterResponse:
    """Complete router response"""
    query: str
    complexity: QueryComplexity
    selected_agents: List[AgentSelection]
    retrieval_results: List[RetrievalResult]
    total_processing_time: float
    needs_synthesis: bool

class EnhancedRouter:
    """Enhanced Agent Router with Vector Database Integration"""
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        """Initialize enhanced router"""
        logger.info("Initializing Enhanced Agent Router...")
        
        # Initialize vector store and embedder
        self.config = VectorStoreConfig(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        self.vector_store = VectorStore(self.config)
        self.embedder = Embedder()
        
        # Initialize query processing pipeline
        self.query_processor = QueryProcessingPipeline()
        
        # Agent configuration for each vertical
        self.agent_configs = {
            'legal_agent': {
                'doc_type': DocumentType.LEGAL_DOCUMENTS,
                'specialization': ['acts', 'rules', 'legal framework', 'constitution', 'amendments'],
                'keywords': ['act', 'rule', 'law', 'legal', 'section', 'amendment', 'constitution', 'rte', 'education act'],
                'confidence_boost': 0.2  # Boost for legal queries
            },
            'go_agent': {
                'doc_type': DocumentType.GOVERNMENT_ORDERS,
                'specialization': ['government orders', 'policy implementation', 'administrative orders'],
                'keywords': ['go', 'government order', 'order', 'ms', 'circular', 'notification', 'implementation'],
                'confidence_boost': 0.15
            },
            'judicial_agent': {
                'doc_type': DocumentType.JUDICIAL_DOCUMENTS,
                'specialization': ['case law', 'court decisions', 'judgments'],
                'keywords': ['case', 'judgment', 'court', 'judicial', 'supreme court', 'high court', 'precedent'],
                'confidence_boost': 0.1
            },
            'data_agent': {
                'doc_type': DocumentType.DATA_REPORTS,
                'specialization': ['statistics', 'reports', 'metrics', 'data analysis'],
                'keywords': ['ratio', 'metric', 'statistics', 'data', 'report', 'enrollment', 'dropout', 'ptr'],
                'confidence_boost': 0.1
            },
            'general_agent': {
                'doc_type': DocumentType.EXTERNAL_SOURCES,
                'specialization': ['all processed documents', 'cross-vertical search'],
                'keywords': [],  # Fallback agent
                'confidence_boost': 0.0
            }
        }
        
        logger.info(f"Enhanced Router initialized with {len(self.agent_configs)} specialized agents")
    
    def route_query(self, query: str, top_k: int = 10) -> RouterResponse:
        """Route query to appropriate agents and retrieve results"""
        import time
        start_time = time.time()
        
        logger.info(f"Routing query: {query}")
        
        # Step 1: Process query for intent and entities
        processed_query = self.query_processor.process(query)
        
        # Step 2: Determine query complexity
        complexity = self._assess_complexity(processed_query)
        
        # Step 3: Select appropriate agents
        selected_agents = self._select_agents(processed_query, complexity)
        
        # Step 4: Execute retrieval across selected agents
        retrieval_results = []
        for agent_selection in selected_agents:
            result = self._execute_agent_retrieval(
                query, 
                agent_selection, 
                top_k=top_k
            )
            if result:
                retrieval_results.append(result)
        
        # Step 5: Determine if synthesis is needed
        needs_synthesis = len(retrieval_results) > 1 or complexity == QueryComplexity.COMPLEX
        
        total_time = time.time() - start_time
        
        response = RouterResponse(
            query=query,
            complexity=complexity,
            selected_agents=selected_agents,
            retrieval_results=retrieval_results,
            total_processing_time=total_time,
            needs_synthesis=needs_synthesis
        )
        
        logger.info(f"Query routed in {total_time:.3f}s - {len(selected_agents)} agents, {len(retrieval_results)} successful retrievals")
        
        return response
    
    def _assess_complexity(self, processed_query) -> QueryComplexity:
        """Assess query complexity for routing strategy"""
        
        # Check for multiple intents
        multiple_intents = len(processed_query.suggested_verticals) > 1
        
        # Check for complex entities
        complex_entities = (
            len(processed_query.entity_summary.split(';')) > 3 or
            'compare' in processed_query.normalized_query.lower() or
            'versus' in processed_query.normalized_query.lower() or
            'across' in processed_query.normalized_query.lower()
        )
        
        # Check intent confidence
        low_confidence = processed_query.intent_confidence < 0.7
        
        if multiple_intents and complex_entities:
            return QueryComplexity.COMPLEX
        elif multiple_intents or complex_entities or low_confidence:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _select_agents(self, processed_query, complexity: QueryComplexity) -> List[AgentSelection]:
        """Select appropriate agents based on query analysis"""
        
        selections = []
        
        # Get base confidence from intent classification
        base_confidence = processed_query.intent_confidence
        
        # Iterate through each agent and calculate selection confidence
        for agent_name, config in self.agent_configs.items():
            confidence = self._calculate_agent_confidence(
                processed_query, 
                config, 
                base_confidence
            )
            
            if confidence > 0.3:  # Threshold for agent selection
                selections.append(AgentSelection(
                    agent_name=agent_name,
                    doc_type=config['doc_type'],
                    confidence=confidence,
                    reasoning=self._get_selection_reasoning(processed_query, config)
                ))
        
        # Sort by confidence and select top agents
        selections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Selection strategy based on complexity
        if complexity == QueryComplexity.SIMPLE:
            # Select top 1-2 agents
            selected = selections[:2]
        elif complexity == QueryComplexity.MODERATE:
            # Select top 2-3 agents
            selected = selections[:3]
        else:  # COMPLEX
            # Select all relevant agents + general agent
            selected = selections[:4]
            # Always include general agent for complex queries
            if not any(s.agent_name == 'general_agent' for s in selected):
                general_selection = next(
                    (s for s in selections if s.agent_name == 'general_agent'), 
                    None
                )
                if general_selection:
                    selected.append(general_selection)
        
        logger.info(f"Selected {len(selected)} agents: {[s.agent_name for s in selected]}")
        
        return selected
    
    def _calculate_agent_confidence(self, processed_query, agent_config: Dict, base_confidence: float) -> float:
        """Calculate confidence score for agent selection"""
        
        confidence = 0.0
        query_lower = processed_query.normalized_query.lower()
        
        # Keyword matching
        keyword_matches = sum(1 for keyword in agent_config['keywords'] 
                             if keyword in query_lower)
        if agent_config['keywords']:  # Only if agent has keywords
            keyword_score = (keyword_matches / len(agent_config['keywords'])) * 0.4
            confidence += keyword_score
        
        # Vertical suggestion matching
        agent_vertical = agent_config['doc_type'].value
        if any(agent_vertical in vertical.lower() for vertical in processed_query.suggested_verticals):
            confidence += 0.3
        
        # Intent-based matching
        intent_score = self._get_intent_agent_match(processed_query.primary_intent, agent_config)
        confidence += intent_score
        
        # Apply confidence boost
        confidence += agent_config['confidence_boost']
        
        # Base confidence factor
        confidence *= base_confidence
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _get_intent_agent_match(self, intent: str, agent_config: Dict) -> float:
        """Get intent-to-agent matching score"""
        
        intent_mappings = {
            'legal_interpretation': {'legal_agent': 0.4, 'general_agent': 0.1},
            'data_query': {'data_agent': 0.4, 'general_agent': 0.2},
            'scheme_inquiry': {'go_agent': 0.3, 'general_agent': 0.2},
            'factual_lookup': {'general_agent': 0.3, 'legal_agent': 0.1, 'go_agent': 0.1},
            'listing': {'data_agent': 0.2, 'general_agent': 0.3}
        }
        
        agent_name = next(name for name, config in self.agent_configs.items() 
                         if config == agent_config)
        
        return intent_mappings.get(intent, {}).get(agent_name, 0.0)
    
    def _get_selection_reasoning(self, processed_query, agent_config: Dict) -> str:
        """Generate reasoning for agent selection"""
        
        reasons = []
        query_lower = processed_query.normalized_query.lower()
        
        # Check keyword matches
        keyword_matches = [kw for kw in agent_config['keywords'] if kw in query_lower]
        if keyword_matches:
            reasons.append(f"Keywords: {', '.join(keyword_matches)}")
        
        # Check vertical matches
        agent_vertical = agent_config['doc_type'].value
        vertical_matches = [v for v in processed_query.suggested_verticals 
                           if agent_vertical in v.lower()]
        if vertical_matches:
            reasons.append(f"Vertical: {', '.join(vertical_matches)}")
        
        # Check intent match
        intent_score = self._get_intent_agent_match(processed_query.primary_intent, agent_config)
        if intent_score > 0.1:
            reasons.append(f"Intent: {processed_query.primary_intent}")
        
        return "; ".join(reasons) if reasons else "General relevance"
    
    def _execute_agent_retrieval(self, query: str, agent_selection: AgentSelection, top_k: int) -> Optional[RetrievalResult]:
        """Execute retrieval for a specific agent"""
        import time
        start_time = time.time()
        
        try:
            # Generate query embedding
            embedding_result = self.embedder.embed_single(query)
            query_embedding = embedding_result.embedding
            
            # Search in the agent's collection
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                doc_type=agent_selection.doc_type,
                limit=top_k
            )
            
            # Convert search results to chunks
            chunks = []
            for result in search_results:
                chunk = {
                    'chunk_id': result.chunk_id,
                    'doc_id': result.doc_id,
                    'text': result.content,
                    'score': result.score,
                    'metadata': result.payload,
                    'vertical': agent_selection.agent_name
                }
                chunks.append(chunk)
            
            processing_time = time.time() - start_time
            
            # Calculate confidence score based on results
            confidence_score = self._calculate_retrieval_confidence(search_results, agent_selection.confidence)
            
            return RetrievalResult(
                agent_name=agent_selection.agent_name,
                doc_type=agent_selection.doc_type.value,
                chunks=chunks,
                total_results=len(chunks),
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error in {agent_selection.agent_name} retrieval: {str(e)}")
            return None
    
    def _calculate_retrieval_confidence(self, search_results: List, selection_confidence: float) -> float:
        """Calculate confidence score for retrieval results"""
        
        if not search_results:
            return 0.0
        
        # Average score of top results
        avg_score = sum(result.score for result in search_results[:3]) / min(3, len(search_results))
        
        # Combine with selection confidence
        combined_confidence = (avg_score * 0.7) + (selection_confidence * 0.3)
        
        return combined_confidence
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents and collections"""
        
        status = {
            'router_status': 'operational',
            'total_agents': len(self.agent_configs),
            'agents': {}
        }
        
        for agent_name, config in self.agent_configs.items():
            try:
                collection_name = self.vector_store.get_collection_name(config['doc_type'])
                collection_info = self.vector_store.client.get_collection(collection_name)
                
                agent_status = {
                    'doc_type': config['doc_type'].value,
                    'collection': collection_name,
                    'embeddings_count': getattr(collection_info, 'points_count', 0),
                    'status': 'operational',
                    'specialization': config['specialization']
                }
            except Exception as e:
                agent_status = {
                    'doc_type': config['doc_type'].value,
                    'status': 'error',
                    'error': str(e)
                }
            
            status['agents'][agent_name] = agent_status
        
        return status