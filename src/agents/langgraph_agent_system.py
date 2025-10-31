"""
LangGraph-based Multi-Agent System for AI Policy Assistant

This implementation uses LangGraph to orchestrate multiple specialized agents
in a graph-based workflow, providing better control, error handling, and
structured execution compared to the current approach.

Key improvements:
1. Graph-based agent orchestration with explicit state management
2. Structured error handling and fallback mechanisms
3. Agent specialization with clear responsibilities
4. Memory and context preservation across agent interactions
5. Parallel and sequential execution patterns
6. Built-in logging and observability
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Annotated, Optional, Union
from dataclasses import dataclass
import logging
import time
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
# from langgraph.checkpoint.sqlite import SqliteSaver  # Removed for compatibility
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

# Internal imports
from src.embeddings.vector_store import VectorStore, VectorStoreConfig, DocumentType
from src.embeddings.embedder import Embedder
from src.query_processing.pipeline import QueryProcessingPipeline
from src.query_processing.qa_pipeline_multi_llm import MultiLLMAnswerGenerator, CitationValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AgentType(Enum):
    """Specialized agent types for different document domains"""
    LEGAL = "legal"
    GOVERNMENT_ORDER = "government_order"
    JUDICIAL = "judicial"
    DATA_ANALYSIS = "data_analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


class QueryComplexity(Enum):
    """Query complexity levels affecting routing strategy"""
    SIMPLE = "simple"        # Single domain, direct retrieval
    MODERATE = "moderate"    # Multi-domain or entity resolution needed
    COMPLEX = "complex"      # Cross-domain synthesis, temporal reasoning


@dataclass
class AgentResult:
    """Result from individual agent execution"""
    agent_type: AgentType
    success: bool
    chunks: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class AgentState(TypedDict):
    """Global state shared across all agents in the graph"""
    # Input
    query: str
    original_query: str
    
    # Processing metadata
    complexity: QueryComplexity
    processed_query: Dict[str, Any]
    
    # Agent results
    legal_result: Optional[AgentResult]
    go_result: Optional[AgentResult]
    judicial_result: Optional[AgentResult]
    data_result: Optional[AgentResult]
    
    # Synthesis
    combined_chunks: List[Dict[str, Any]]
    synthesized_answer: Optional[str]
    
    # Final output
    final_answer: str
    confidence_score: float
    citations: Dict[str, Any]
    metadata: Dict[str, Any]
    
    # Messages for conversation tracking
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Error handling
    errors: List[str]
    fallback_used: bool


class LangGraphPolicyAgent:
    """
    LangGraph-based multi-agent system for policy document Q&A.
    
    This class orchestrates specialized agents in a graph-based workflow,
    providing structured execution with error handling and state management.
    """
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        llm_provider: str = "gemini",
        llm_api_key: Optional[str] = None,
        checkpoint_path: str = "./checkpoints.sqlite"
    ):
        """
        Initialize the LangGraph-based agent system.
        
        Args:
            qdrant_url: Qdrant vector store URL
            qdrant_api_key: Qdrant API key
            llm_provider: LLM provider ("gemini" or "groq")
            llm_api_key: LLM API key
            checkpoint_path: Path for state checkpointing
        """
        logger.info("Initializing LangGraph Policy Agent System...")
        
        # Initialize core components
        self.vector_store = VectorStore(VectorStoreConfig(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        ))
        self.embedder = Embedder()
        self.query_processor = QueryProcessingPipeline()
        self.llm_generator = MultiLLMAnswerGenerator(llm_provider, llm_api_key)
        self.citation_validator = CitationValidator()
        
        # Build the agent graph (without checkpointing for now)
        self.graph = self._build_agent_graph()
        
        logger.info("LangGraph Policy Agent System initialized successfully")
    
    def _build_agent_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow graph with specialized agents.
        
        Graph Structure:
        START → query_analysis → routing_decision → [parallel_agents] → synthesis → validation → END
        """
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("query_analysis", self._query_analysis_node)
        workflow.add_node("routing_decision", self._routing_decision_node)
        workflow.add_node("legal_agent", self._legal_agent_node)
        workflow.add_node("go_agent", self._go_agent_node)
        workflow.add_node("judicial_agent", self._judicial_agent_node)
        workflow.add_node("data_agent", self._data_agent_node)
        workflow.add_node("synthesis_agent", self._synthesis_agent_node)
        workflow.add_node("validation_agent", self._validation_agent_node)
        workflow.add_node("fallback_agent", self._fallback_agent_node)
        
        # Define entry point
        workflow.add_edge(START, "query_analysis")
        
        # Query analysis flows to routing
        workflow.add_edge("query_analysis", "routing_decision")
        
        # Conditional routing based on query complexity and type
        workflow.add_conditional_edges(
            "routing_decision",
            self._should_route_to_agents,
            {
                "legal": "legal_agent",
                "government_order": "go_agent", 
                "judicial": "judicial_agent",
                "data": "data_agent",
                "multi_agent": "legal_agent",  # Start with legal for complex queries
                "fallback": "fallback_agent"
            }
        )
        
        # Agent completion flows
        workflow.add_conditional_edges(
            "legal_agent",
            self._after_agent_execution,
            {
                "continue_go": "go_agent",
                "continue_judicial": "judicial_agent", 
                "continue_data": "data_agent",
                "synthesis": "synthesis_agent"
            }
        )
        
        workflow.add_conditional_edges(
            "go_agent",
            self._after_agent_execution,
            {
                "continue_judicial": "judicial_agent",
                "continue_data": "data_agent", 
                "synthesis": "synthesis_agent"
            }
        )
        
        workflow.add_conditional_edges(
            "judicial_agent",
            self._after_agent_execution,
            {
                "continue_data": "data_agent",
                "synthesis": "synthesis_agent"
            }
        )
        
        workflow.add_edge("data_agent", "synthesis_agent")
        workflow.add_edge("fallback_agent", "synthesis_agent")
        
        # Synthesis flows to validation
        workflow.add_edge("synthesis_agent", "validation_agent")
        
        # Validation is the final step
        workflow.add_edge("validation_agent", END)
        
        # Compile the graph
        return workflow.compile()
    
    # ========== NODE IMPLEMENTATIONS ==========
    
    def _query_analysis_node(self, state: AgentState) -> AgentState:
        """
        Analyze the incoming query to understand intent, entities, and complexity.
        """
        logger.info(f"Analyzing query: {state['query']}")
        
        try:
            # Process query using existing pipeline
            processed_query = self.query_processor.process(state['query'])
            
            # Assess complexity based on multiple factors
            complexity = self._assess_query_complexity(processed_query)
            
            # Update state
            state.update({
                "processed_query": {
                    "intent": processed_query.primary_intent,
                    "entities": processed_query.entity_summary,
                    "verticals": processed_query.suggested_verticals,
                    "confidence": processed_query.intent_confidence,
                    "normalized": processed_query.normalized_query
                },
                "complexity": complexity,
                "messages": state["messages"] + [
                    SystemMessage(content=f"Query analyzed: Intent={processed_query.primary_intent}, Complexity={complexity.value}")
                ]
            })
            
            logger.info(f"Query analysis complete: Intent={processed_query.primary_intent}, Complexity={complexity.value}")
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["errors"].append(f"Query analysis error: {str(e)}")
            state["complexity"] = QueryComplexity.SIMPLE  # Default fallback
        
        return state
    
    def _routing_decision_node(self, state: AgentState) -> AgentState:
        """
        Decide which agents should be executed based on query analysis.
        """
        logger.info("Making routing decisions...")
        
        processed = state.get("processed_query", {})
        complexity = state.get("complexity", QueryComplexity.SIMPLE)
        
        # Determine routing strategy
        routing_metadata = {
            "primary_vertical": None,
            "secondary_verticals": [],
            "execution_mode": "sequential",
            "confidence_threshold": 0.3
        }
        
        # Extract primary vertical from intent and entities
        intent = processed.get("intent", "")
        verticals = processed.get("verticals", [])
        
        if "legal" in intent.lower() or any("legal" in v.lower() for v in verticals):
            routing_metadata["primary_vertical"] = "legal"
        elif "government_order" in intent.lower() or any("go" in v.lower() for v in verticals):
            routing_metadata["primary_vertical"] = "government_order"
        elif "judicial" in intent.lower() or any("judicial" in v.lower() for v in verticals):
            routing_metadata["primary_vertical"] = "judicial"
        elif "data" in intent.lower() or any("data" in v.lower() for v in verticals):
            routing_metadata["primary_vertical"] = "data"
        
        # For complex queries, plan multi-agent execution
        if complexity == QueryComplexity.COMPLEX:
            routing_metadata["execution_mode"] = "parallel"
            routing_metadata["secondary_verticals"] = [v for v in ["legal", "government_order", "data"] 
                                                     if v != routing_metadata["primary_vertical"]]
        
        state["metadata"] = routing_metadata
        
        logger.info(f"Routing decision: Primary={routing_metadata['primary_vertical']}, Mode={routing_metadata['execution_mode']}")
        
        return state
    
    def _legal_agent_node(self, state: AgentState) -> AgentState:
        """
        Execute legal document retrieval and processing.
        """
        logger.info("Executing Legal Agent...")
        
        try:
            result = self._execute_specialized_retrieval(
                state["query"],
                DocumentType.LEGAL_DOCUMENTS,
                AgentType.LEGAL
            )
            state["legal_result"] = result
            
            state["messages"].append(
                AIMessage(content=f"Legal Agent: Retrieved {len(result.chunks)} legal documents (confidence: {result.confidence:.2f})")
            )
            
        except Exception as e:
            logger.error(f"Legal agent failed: {e}")
            state["errors"].append(f"Legal agent error: {str(e)}")
            state["legal_result"] = AgentResult(
                agent_type=AgentType.LEGAL,
                success=False,
                chunks=[],
                confidence=0.0,
                processing_time=0.0,
                error_message=str(e)
            )
        
        return state
    
    def _go_agent_node(self, state: AgentState) -> AgentState:
        """
        Execute government order retrieval and processing.
        """
        logger.info("Executing Government Order Agent...")
        
        try:
            result = self._execute_specialized_retrieval(
                state["query"],
                DocumentType.GOVERNMENT_ORDERS,
                AgentType.GOVERNMENT_ORDER
            )
            state["go_result"] = result
            
            state["messages"].append(
                AIMessage(content=f"GO Agent: Retrieved {len(result.chunks)} government orders (confidence: {result.confidence:.2f})")
            )
            
        except Exception as e:
            logger.error(f"Government Order agent failed: {e}")
            state["errors"].append(f"GO agent error: {str(e)}")
            state["go_result"] = AgentResult(
                agent_type=AgentType.GOVERNMENT_ORDER,
                success=False,
                chunks=[],
                confidence=0.0,
                processing_time=0.0,
                error_message=str(e)
            )
        
        return state
    
    def _judicial_agent_node(self, state: AgentState) -> AgentState:
        """
        Execute judicial document retrieval and processing.
        """
        logger.info("Executing Judicial Agent...")
        
        try:
            result = self._execute_specialized_retrieval(
                state["query"],
                DocumentType.JUDICIAL_DOCUMENTS,
                AgentType.JUDICIAL
            )
            state["judicial_result"] = result
            
            state["messages"].append(
                AIMessage(content=f"Judicial Agent: Retrieved {len(result.chunks)} judicial documents (confidence: {result.confidence:.2f})")
            )
            
        except Exception as e:
            logger.error(f"Judicial agent failed: {e}")
            state["errors"].append(f"Judicial agent error: {str(e)}")
            state["judicial_result"] = AgentResult(
                agent_type=AgentType.JUDICIAL,
                success=False,
                chunks=[],
                confidence=0.0,
                processing_time=0.0,
                error_message=str(e)
            )
        
        return state
    
    def _data_agent_node(self, state: AgentState) -> AgentState:
        """
        Execute data/statistics retrieval and processing.
        """
        logger.info("Executing Data Analysis Agent...")
        
        try:
            result = self._execute_specialized_retrieval(
                state["query"],
                DocumentType.DATA_REPORTS,
                AgentType.DATA_ANALYSIS
            )
            state["data_result"] = result
            
            state["messages"].append(
                AIMessage(content=f"Data Agent: Retrieved {len(result.chunks)} data reports (confidence: {result.confidence:.2f})")
            )
            
        except Exception as e:
            logger.error(f"Data agent failed: {e}")
            state["errors"].append(f"Data agent error: {str(e)}")
            state["data_result"] = AgentResult(
                agent_type=AgentType.DATA_ANALYSIS,
                success=False,
                chunks=[],
                confidence=0.0,
                processing_time=0.0,
                error_message=str(e)
            )
        
        return state
    
    def _synthesis_agent_node(self, state: AgentState) -> AgentState:
        """
        Synthesize results from all executed agents into a coherent answer.
        """
        logger.info("Executing Synthesis Agent...")
        
        try:
            # Collect all successful results
            all_chunks = []
            agent_results = []
            
            for result_key in ["legal_result", "go_result", "judicial_result", "data_result"]:
                result = state.get(result_key)
                if result and result.success and result.chunks:
                    all_chunks.extend(result.chunks)
                    agent_results.append(result)
            
            state["combined_chunks"] = all_chunks
            
            if not all_chunks:
                # No successful retrievals - use fallback
                state["synthesized_answer"] = "I couldn't find relevant information in the knowledge base to answer this query. Please try rephrasing or check if this topic is covered in our documents."
                state["confidence_score"] = 0.0
                state["fallback_used"] = True
            else:
                # Generate answer using LLM
                context = self._format_context_for_llm(all_chunks)
                
                llm_response = self.llm_generator.generate_answer(
                    state["query"],
                    context,
                    mode="normal_qa"
                )
                
                if llm_response.get("success", False):
                    state["synthesized_answer"] = llm_response["answer"]
                    state["confidence_score"] = self._calculate_confidence(agent_results, llm_response)
                else:
                    state["synthesized_answer"] = f"Error generating answer: {llm_response.get('error', 'Unknown error')}"
                    state["confidence_score"] = 0.0
            
            state["messages"].append(
                AIMessage(content=f"Synthesis complete: Generated answer from {len(all_chunks)} chunks")
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            state["errors"].append(f"Synthesis error: {str(e)}")
            state["synthesized_answer"] = f"Error during answer synthesis: {str(e)}"
            state["confidence_score"] = 0.0
        
        return state
    
    def _validation_agent_node(self, state: AgentState) -> AgentState:
        """
        Validate the synthesized answer and extract citations.
        """
        logger.info("Executing Validation Agent...")
        
        try:
            answer = state.get("synthesized_answer", "")
            chunks = state.get("combined_chunks", [])
            
            # Validate citations
            citations = self.citation_validator.validate_citations(answer, chunks)
            
            # Final answer processing
            state.update({
                "final_answer": answer,
                "citations": citations,
                "metadata": {
                    **state.get("metadata", {}),
                    "total_chunks": len(chunks),
                    "agents_executed": [k for k in ["legal_result", "go_result", "judicial_result", "data_result"] 
                                      if state.get(k) and state[k].success],
                    "errors_encountered": len(state.get("errors", [])),
                    "validation_complete": True
                }
            })
            
            state["messages"].append(
                AIMessage(content=f"Validation complete: {citations.get('total_citations', 0)} citations validated")
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            state["errors"].append(f"Validation error: {str(e)}")
            state["final_answer"] = state.get("synthesized_answer", "Error in validation")
            state["citations"] = {"error": str(e)}
        
        return state
    
    def _fallback_agent_node(self, state: AgentState) -> AgentState:
        """
        Fallback agent for when routing fails or no specific agents are suitable.
        """
        logger.info("Executing Fallback Agent...")
        
        try:
            # Try general search across all collections
            result = self._execute_specialized_retrieval(
                state["query"],
                DocumentType.EXTERNAL_SOURCES,
                AgentType.SYNTHESIS,
                top_k=15  # Cast wider net
            )
            
            state["combined_chunks"] = result.chunks
            state["fallback_used"] = True
            
            state["messages"].append(
                AIMessage(content=f"Fallback Agent: Retrieved {len(result.chunks)} documents from general search")
            )
            
        except Exception as e:
            logger.error(f"Fallback agent failed: {e}")
            state["errors"].append(f"Fallback agent error: {str(e)}")
            state["combined_chunks"] = []
        
        return state
    
    # ========== HELPER METHODS ==========
    
    def _execute_specialized_retrieval(
        self,
        query: str,
        doc_type: DocumentType,
        agent_type: AgentType,
        top_k: int = 10
    ) -> AgentResult:
        """
        Execute retrieval for a specific document type.
        """
        start_time = time.time()
        
        try:
            # Generate embedding
            embedding_result = self.embedder.embed_single(query)
            query_embedding = embedding_result.embedding
            
            # Search specific collection
            collection_name = self.vector_store.get_collection_name(doc_type)
            search_results = self.vector_store.search(
                query_vector=query_embedding,
                collection_names=[collection_name],
                limit=top_k
            )
            
            # Convert to chunks
            chunks = []
            for result in search_results:
                chunk = {
                    'chunk_id': result.chunk_id,
                    'doc_id': result.doc_id,
                    'content': result.content,
                    'score': result.score,
                    'metadata': result.payload,
                    'agent_type': agent_type.value
                }
                chunks.append(chunk)
            
            # Calculate confidence
            confidence = self._calculate_retrieval_confidence(search_results)
            
            processing_time = time.time() - start_time
            
            return AgentResult(
                agent_type=agent_type,
                success=True,
                chunks=chunks,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "collection": collection_name,
                    "total_results": len(chunks)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return AgentResult(
                agent_type=agent_type,
                success=False,
                chunks=[],
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _assess_query_complexity(self, processed_query) -> QueryComplexity:
        """
        Assess query complexity based on intent, entities, and structure.
        """
        # Multiple verticals suggested = complex
        if len(processed_query.suggested_verticals) > 2:
            return QueryComplexity.COMPLEX
        
        # Low confidence = moderate
        if processed_query.intent_confidence < 0.6:
            return QueryComplexity.MODERATE
        
        # Complex entity structure
        entities = processed_query.entity_summary.lower()
        if any(word in entities for word in ['compare', 'versus', 'across', 'between', 'all']):
            return QueryComplexity.COMPLEX
        
        # Multiple entity types
        if len(entities.split(';')) > 3:
            return QueryComplexity.MODERATE
        
        return QueryComplexity.SIMPLE
    
    def _format_context_for_llm(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into LLM-friendly context.
        """
        context_parts = []
        
        for idx, chunk in enumerate(chunks[:10], 1):  # Limit to top 10
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', chunk.get('text', ''))
            
            context_parts.append(f"""
[Source {idx}]
Document: {metadata.get('title', 'Unknown')}
Type: {metadata.get('doc_type', 'unknown')}
Section: {metadata.get('section_id', 'N/A')}
Score: {chunk.get('score', 0.0):.3f}

Content:
{content}

{'='*80}
""")
        
        return "\n".join(context_parts)
    
    def _calculate_retrieval_confidence(self, search_results: List) -> float:
        """
        Calculate confidence based on retrieval scores.
        """
        if not search_results:
            return 0.0
        
        top_scores = [result.score for result in search_results[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, avg_score + 0.5))  # Adjust baseline
    
    def _calculate_confidence(self, agent_results: List[AgentResult], llm_response: Dict) -> float:
        """
        Calculate overall confidence score.
        """
        if not agent_results:
            return 0.0
        
        # Average agent confidence
        avg_agent_confidence = sum(r.confidence for r in agent_results) / len(agent_results)
        
        # LLM success factor
        llm_factor = 1.0 if llm_response.get("success", False) else 0.5
        
        # Number of successful agents factor
        agent_factor = min(len(agent_results) / 2, 1.0)  # Normalize to 1.0 at 2+ agents
        
        combined = (avg_agent_confidence * 0.6 + llm_factor * 0.3 + agent_factor * 0.1)
        
        return round(combined, 2)
    
    # ========== ROUTING LOGIC ==========
    
    def _should_route_to_agents(self, state: AgentState) -> str:
        """
        Determine which agent(s) to route to based on analysis.
        """
        metadata = state.get("metadata", {})
        primary = metadata.get("primary_vertical")
        complexity = state.get("complexity", QueryComplexity.SIMPLE)
        
        if not primary:
            return "fallback"
        
        if complexity == QueryComplexity.COMPLEX:
            return "multi_agent"
        
        return primary
    
    def _after_agent_execution(self, state: AgentState) -> str:
        """
        Determine next step after an agent completes.
        """
        metadata = state.get("metadata", {})
        complexity = state.get("complexity", QueryComplexity.SIMPLE)
        
        if complexity != QueryComplexity.COMPLEX:
            return "synthesis"
        
        # For complex queries, execute multiple agents in sequence
        if not state.get("go_result") and "government_order" in metadata.get("secondary_verticals", []):
            return "continue_go"
        
        if not state.get("judicial_result") and "judicial" in metadata.get("secondary_verticals", []):
            return "continue_judicial"
        
        if not state.get("data_result") and "data" in metadata.get("secondary_verticals", []):
            return "continue_data"
        
        return "synthesis"
    
    # ========== PUBLIC API ==========
    
    def answer_query(
        self,
        query: str,
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point for answering queries using the LangGraph agent system.
        
        Args:
            query: The user's question
            thread_id: Optional thread ID for conversation continuity
            **kwargs: Additional configuration options
            
        Returns:
            Complete response with answer, citations, and metadata
        """
        logger.info(f"Processing query with LangGraph: {query}")
        
        # Initialize state
        initial_state = {
            "query": query,
            "original_query": query,
            "complexity": QueryComplexity.SIMPLE,
            "processed_query": {},
            "legal_result": None,
            "go_result": None,
            "judicial_result": None,
            "data_result": None,
            "combined_chunks": [],
            "synthesized_answer": None,
            "final_answer": "",
            "confidence_score": 0.0,
            "citations": {},
            "metadata": {},
            "messages": [HumanMessage(content=query)],
            "errors": [],
            "fallback_used": False
        }
        
        # Configure execution (simplified without checkpointing)
        config = RunnableConfig() if not thread_id else RunnableConfig(
            {"configurable": {"thread_id": thread_id}}
        )
        
        start_time = time.time()
        
        try:
            # Execute the graph
            final_state = self.graph.invoke(initial_state, config)
            
            processing_time = time.time() - start_time
            
            # Format response
            response = {
                "query": query,
                "answer": final_state.get("final_answer", "No answer generated"),
                "confidence": final_state.get("confidence_score", 0.0),
                "citations": final_state.get("citations", {}),
                "metadata": {
                    **final_state.get("metadata", {}),
                    "processing_time": processing_time,
                    "complexity": final_state.get("complexity", QueryComplexity.SIMPLE).value,
                    "agents_used": [k.replace("_result", "") for k in ["legal_result", "go_result", "judicial_result", "data_result"] 
                                   if final_state.get(k) and final_state[k].success],
                    "fallback_used": final_state.get("fallback_used", False),
                    "errors": final_state.get("errors", [])
                },
                "success": True
            }
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error(f"Query processing failed: {e}")
            
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "confidence": 0.0,
                "citations": {},
                "metadata": {
                    "processing_time": processing_time,
                    "error": str(e)
                },
                "success": False
            }


# Convenience function for testing
def create_langgraph_agent(
    qdrant_url: str = None,
    qdrant_api_key: str = None,
    llm_provider: str = "gemini",
    llm_api_key: str = None
) -> LangGraphPolicyAgent:
    """
    Create a LangGraph agent with environment variables as defaults.
    """
    return LangGraphPolicyAgent(
        qdrant_url=qdrant_url or os.getenv("QDRANT_URL"),
        qdrant_api_key=qdrant_api_key or os.getenv("QDRANT_API_KEY"),
        llm_provider=llm_provider,
        llm_api_key=llm_api_key or os.getenv("GOOGLE_API_KEY" if llm_provider == "gemini" else "GROQ_API_KEY")
    )


if __name__ == "__main__":
    print("LangGraph Policy Agent System module loaded successfully")
    
    # Test basic functionality
    try:
        agent = create_langgraph_agent()
        print("✅ Agent created successfully")
        
        # Test query
        result = agent.answer_query("What is Section 12 of RTE Act?")
        print(f"✅ Test query completed: {result['success']}")
        print(f"Answer preview: {result['answer'][:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")