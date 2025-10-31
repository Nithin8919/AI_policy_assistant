"""
Complete Question-Answering Pipeline with LLM Integration
Connects retrieval system with Claude for natural language answers
with enhanced error handling, retry logic, and usage tracking.
"""

import os
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import anthropic
import re
import logging
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class QAResponse:
    """Complete Q&A response with metadata"""
    query: str
    answer: str
    citations: Dict[str, Any]
    retrieval_stats: Dict[str, Any]
    llm_stats: Dict[str, Any]
    mode: str
    confidence_score: float
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# Utility: Retry Logic with Exponential Backoff
# ============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries} retry attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# Usage Tracker for LLM Calls
# ============================================================================

class UsageTracker:
    """
    Tracks LLM API usage for monitoring and cost estimation.
    Persists to disk for long-term tracking.
    """
    
    def __init__(self, log_file: str = "logs/llm_usage.jsonl"):
        """
        Initialize usage tracker.
        
        Args:
            log_file: Path to usage log file (JSONL format)
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory counters
        self.session_stats = {
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "errors": 0,
            "by_mode": {},
            "session_start": datetime.now().isoformat()
        }
        
        logger.info(f"Usage tracker initialized. Logging to {self.log_file}")
    
    def log_call(
        self,
        query: str,
        mode: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        success: bool,
        processing_time: float,
        error: Optional[str] = None
    ) -> None:
        """
        Log an LLM API call.
        
        Args:
            query: User query
            mode: Answer generation mode
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model used
            success: Whether call succeeded
            processing_time: Processing time in seconds
            error: Error message if failed
        """
        # Calculate cost (Claude Sonnet pricing as of 2024)
        # Input: $3 per million tokens, Output: $15 per million tokens
        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        total_cost = input_cost + output_cost
        
        # Update session stats
        self.session_stats["total_calls"] += 1
        if success:
            self.session_stats["total_input_tokens"] += input_tokens
            self.session_stats["total_output_tokens"] += output_tokens
            self.session_stats["total_cost_usd"] += total_cost
            
            if mode not in self.session_stats["by_mode"]:
                self.session_stats["by_mode"][mode] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            self.session_stats["by_mode"][mode]["calls"] += 1
            self.session_stats["by_mode"][mode]["tokens"] += input_tokens + output_tokens
            self.session_stats["by_mode"][mode]["cost"] += total_cost
        else:
            self.session_stats["errors"] += 1
        
        # Log to file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Truncate for privacy
            "mode": mode,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": total_cost,
            "processing_time": processing_time,
            "success": success,
            "error": error
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write usage log: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return self.session_stats.copy()
    
    def get_cost_summary(self) -> str:
        """Get formatted cost summary"""
        stats = self.session_stats
        summary = f"""
Usage Summary:
  Total Calls: {stats['total_calls']}
  Input Tokens: {stats['total_input_tokens']:,}
  Output Tokens: {stats['total_output_tokens']:,}
  Total Tokens: {stats['total_input_tokens'] + stats['total_output_tokens']:,}
  Estimated Cost: ${stats['total_cost_usd']:.4f}
  Errors: {stats['errors']}
"""
        return summary


# ============================================================================
# Context Assembler
# ============================================================================

class ContextAssembler:
    """Assembles retrieved chunks into LLM-friendly context"""
    
    def __init__(self, max_chunks: int = 10, max_tokens: int = 8000):
        """
        Initialize context assembler.
        
        Args:
            max_chunks: Maximum number of chunks to include
            max_tokens: Approximate token limit for context
        """
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens
    
    def assemble_context(
        self, 
        retrieval_results: List[Dict[str, Any]],
        max_chunks: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format retrieved chunks for LLM consumption.
        
        Args:
            retrieval_results: Retrieved chunks with metadata
            max_chunks: Override max chunks (optional)
            max_tokens: Override max tokens (optional)
            
        Returns:
            Tuple of (formatted_context_string, source_list)
        """
        max_chunks = max_chunks or self.max_chunks
        max_tokens = max_tokens or self.max_tokens
        
        context_parts = []
        sources = []
        total_tokens = 0
        
        for idx, result in enumerate(retrieval_results[:max_chunks]):
            chunk_text = result.get('content', result.get('text', ''))
            metadata = result.get('metadata', {})
            
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            chunk_tokens = len(chunk_text) // 4
            
            if total_tokens + chunk_tokens > max_tokens:
                logger.warning(f"Context token limit reached at chunk {idx}")
                break
            
            # Format source
            source_info = {
                'index': idx + 1,
                'document': metadata.get('title', 'Unknown Document'),
                'section': metadata.get('section_id', 'N/A'),
                'year': metadata.get('year', 'N/A'),
                'doc_type': metadata.get('doc_type', 'unknown'),
                'score': result.get('score', 0.0),
                'chunk_id': result.get('chunk_id', ''),
                'file_path': metadata.get('file_path', ''),
            }
            sources.append(source_info)
            
            # Format for LLM
            context_parts.append(f"""
[Source {idx + 1}]
Document: {source_info['document']}
Type: {source_info['doc_type']}
Section: {source_info['section']}
Year: {source_info['year']}
Relevance: {source_info['score']:.2f}

Content:
{chunk_text}

{'='*80}
""")
            
            total_tokens += chunk_tokens
        
        formatted_context = "\n".join(context_parts)
        logger.info(f"Assembled context with {len(sources)} sources (~{total_tokens} tokens)")
        
        return formatted_context, sources


# ============================================================================
# Claude Answer Generator
# ============================================================================

class ClaudeAnswerGenerator:
    """Claude API integration for answer generation with retry logic"""
    
    # System prompts for different modes
    SYSTEM_PROMPTS = {
        "normal_qa": """You are an AI Policy Assistant for Andhra Pradesh education policy.
Your role is to provide accurate, citation-based answers to policy questions.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. ALWAYS cite sources using [Source X] notation
3. If information is not in context, say "I don't have information about this in the available documents"
4. Structure answers clearly with bullet points or numbered lists for readability
5. Include relevant section numbers, GO numbers, and document names
6. Be concise but comprehensive
7. Never make up information or cite sources that don't exist
8. If asked for opinions or predictions, politely decline and stick to facts

FORMAT GUIDELINES:
- Use markdown formatting for clarity
- Bold important terms and section numbers
- Use bullet points for lists
- Always end with a "Sources" section listing all citations""",

        "detailed": """You are an AI Policy Analyst for Andhra Pradesh education system.
Provide comprehensive, detailed analysis with:

1. **Legal Framework:** Cite relevant Acts, Rules, and their sections
2. **Implementation Guidelines:** Reference Government Orders and circulars
3. **Cross-References:** Link related provisions and amendments
4. **Historical Context:** Mention when laws were enacted/amended if relevant
5. **Practical Implications:** Explain how policies apply in practice

ALWAYS cite sources meticulously using [Source X] notation.""",

        "concise": """You are an AI Policy Assistant providing quick, accurate answers.
- Be extremely concise (2-3 sentences max)
- Focus on the direct answer only
- Still include citations [Source X]
- Skip elaboration unless critical""",

        "comparative": """You are an AI Policy Analyst comparing different provisions.
When comparing:
1. Present both/all items clearly
2. Highlight similarities and differences
3. Use tables or side-by-side format where appropriate
4. Cite sources for each claim
5. Note any conflicts or contradictions"""
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        usage_tracker: Optional[UsageTracker] = None
    ):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
            model: Claude model to use
            usage_tracker: Optional usage tracker instance
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Claude API key required. Set ANTHROPIC_API_KEY or pass api_key parameter"
            )
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.usage_tracker = usage_tracker
        logger.info(f"Claude Answer Generator initialized with model {self.model}")
    
    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(anthropic.APIError, anthropic.RateLimitError)
    )
    def generate_answer(
        self,
        query: str,
        context: str,
        mode: str = "normal_qa",
        max_tokens: int = 2000,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate natural language answer using Claude.
        
        Args:
            query: User's original question
            context: Assembled context from retrieval
            mode: Answer generation mode (normal_qa, detailed, concise, comparative)
            max_tokens: Maximum tokens for answer
            temperature: Temperature for generation (0.0 = deterministic)
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Get system prompt for mode
        system_prompt = self.SYSTEM_PROMPTS.get(mode, self.SYSTEM_PROMPTS["normal_qa"])
        
        # Construct user prompt
        user_prompt = f"""Based on the following context from official Andhra Pradesh education policy documents, 
please answer this question:

**QUESTION:** {query}

**CONTEXT FROM DOCUMENTS:**

{context}

Please provide a clear, well-structured answer with proper citations using [Source X] notation."""
        
        try:
            # Call Claude API (with retry logic from decorator)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            answer_text = response.content[0].text
            processing_time = time.time() - start_time
            
            result = {
                "answer": answer_text,
                "model": self.model,
                "mode": mode,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "processing_time": processing_time,
                "stop_reason": response.stop_reason,
                "success": True
            }
            
            # Log usage
            if self.usage_tracker:
                self.usage_tracker.log_call(
                    query=query,
                    mode=mode,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=self.model,
                    success=True,
                    processing_time=processing_time
                )
            
            logger.info(
                f"Answer generated in {processing_time:.2f}s using "
                f"{result['total_tokens']} tokens"
            )
            
            return result
            
        except anthropic.APIError as e:
            error_msg = f"Claude API error: {str(e)}"
            logger.error(error_msg)
            
            if self.usage_tracker:
                self.usage_tracker.log_call(
                    query=query,
                    mode=mode,
                    input_tokens=0,
                    output_tokens=0,
                    model=self.model,
                    success=False,
                    processing_time=time.time() - start_time,
                    error=error_msg
                )
            
            return {
                "answer": f"Error generating answer: {str(e)}",
                "success": False,
                "error": str(e),
                "error_type": "api_error"
            }
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            
            if self.usage_tracker:
                self.usage_tracker.log_call(
                    query=query,
                    mode=mode,
                    input_tokens=0,
                    output_tokens=0,
                    model=self.model,
                    success=False,
                    processing_time=time.time() - start_time,
                    error=error_msg
                )
            
            return {
                "answer": f"Unexpected error: {str(e)}",
                "success": False,
                "error": str(e),
                "error_type": "unexpected_error"
            }


# ============================================================================
# Citation Validator
# ============================================================================

class CitationValidator:
    """Validates and extracts citations from LLM answers"""
    
    def validate_citations(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract and validate [Source X] citations.
        
        Args:
            answer: LLM-generated answer with citations
            sources: List of source documents provided to LLM
            
        Returns:
            Citation analysis dictionary
        """
        # Find all [Source X] citations
        citation_pattern = r'\[Source (\d+)\]'
        citations_found = re.findall(citation_pattern, answer)
        
        citation_details = {}
        invalid_citations = []
        
        for cite_num_str in set(citations_found):
            cite_num = int(cite_num_str)
            source_idx = cite_num - 1  # Convert to 0-indexed
            
            if source_idx < len(sources):
                # Valid citation
                citation_details[cite_num_str] = sources[source_idx]
            else:
                # Invalid citation (LLM hallucinated a source number)
                invalid_citations.append(cite_num_str)
                logger.warning(f"Invalid citation found: [Source {cite_num_str}]")
        
        total_citations = len(citations_found)
        unique_sources_cited = len(citation_details)
        all_valid = len(invalid_citations) == 0
        
        # Calculate citation density
        word_count = len(answer.split())
        citation_density = total_citations / max(word_count, 1)
        
        return {
            "total_citations": total_citations,
            "unique_sources_cited": unique_sources_cited,
            "citation_details": citation_details,
            "invalid_citations": invalid_citations,
            "all_citations_valid": all_valid,
            "citation_density": citation_density,
            "sources_used_percentage": (unique_sources_cited / len(sources) * 100) if sources else 0,
            "word_count": word_count
        }


# ============================================================================
# Main QA Pipeline
# ============================================================================

class QAPipeline:
    """
    Complete Question-Answering Pipeline integrating retrieval + LLM.
    
    This is the main class that ties everything together:
    1. Takes a natural language query
    2. Uses EnhancedRouter for retrieval
    3. Assembles context for LLM
    4. Generates answer with Claude
    5. Validates citations
    6. Returns complete response
    """
    
    def __init__(
        self,
        router,  # EnhancedRouter instance
        claude_api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        enable_usage_tracking: bool = True,
        usage_log_file: str = "logs/llm_usage.jsonl"
    ):
        """
        Initialize QA Pipeline.
        
        Args:
            router: EnhancedRouter instance (already initialized with Qdrant)
            claude_api_key: Anthropic API key
            model: Claude model to use
            enable_usage_tracking: Whether to track LLM usage
            usage_log_file: Path to usage log file
        """
        self.router = router
        self.assembler = ContextAssembler()
        
        # Initialize usage tracker
        self.usage_tracker = UsageTracker(usage_log_file) if enable_usage_tracking else None
        
        # Initialize generator with tracker
        self.generator = ClaudeAnswerGenerator(
            api_key=claude_api_key,
            model=model,
            usage_tracker=self.usage_tracker
        )
        
        self.validator = CitationValidator()
        
        logger.info("QA Pipeline initialized")
    
    def answer_query(
        self,
        query: str,
        mode: str = "normal_qa",
        top_k: int = 10
    ) -> QAResponse:
        """
        Complete pipeline: Query → Retrieval → LLM → Answer.
        
        Args:
            query: Natural language question
            mode: Answer generation mode (normal_qa, detailed, concise, comparative)
            top_k: Number of chunks to retrieve
            
        Returns:
            Complete QAResponse object
        """
        pipeline_start = time.time()
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Route query and retrieve relevant chunks
            logger.info("Step 1: Retrieving relevant documents...")
            retrieval_response = self.router.route_query(query, top_k=top_k)
            
            # Flatten all chunks from all agents
            all_chunks = []
            for result in retrieval_response.retrieval_results:
                all_chunks.extend(result.chunks)
            
            if not all_chunks:
                logger.warning("No relevant documents found")
                return QAResponse(
                    query=query,
                    answer="I couldn't find any relevant documents to answer this question. Please try rephrasing or check if this topic is covered in the knowledge base.",
                    citations={
                        "total_citations": 0,
                        "unique_sources_cited": 0,
                        "all_citations_valid": True
                    },
                    retrieval_stats={"chunks_retrieved": 0, "agents_queried": 0},
                    llm_stats={},
                    mode=mode,
                    confidence_score=0.0,
                    processing_time=time.time() - pipeline_start
                )
            
            # Step 2: Assemble context for LLM
            logger.info("Step 2: Assembling context...")
            context, sources = self.assembler.assemble_context(all_chunks, max_chunks=top_k)
            
            # Step 3: Generate answer using Claude
            logger.info("Step 3: Generating answer with Claude...")
            llm_response = self.generator.generate_answer(query, context, mode)
            
            if not llm_response.get('success', False):
                logger.error("Answer generation failed")
                return QAResponse(
                    query=query,
                    answer=llm_response.get('answer', 'Error generating answer'),
                    citations={},
                    retrieval_stats={"chunks_retrieved": len(all_chunks)},
                    llm_stats=llm_response,
                    mode=mode,
                    confidence_score=0.0,
                    processing_time=time.time() - pipeline_start
                )
            
            # Step 4: Validate citations
            logger.info("Step 4: Validating citations...")
            citations = self.validator.validate_citations(llm_response['answer'], sources)
            
            # Step 5: Calculate confidence score
            confidence = self._calculate_confidence(
                retrieval_response,
                citations,
                len(all_chunks)
            )
            
            total_time = time.time() - pipeline_start
            
            logger.info(f"Query answered in {total_time:.2f}s with confidence {confidence:.2f}")
            
            # Return complete response
            return QAResponse(
                query=query,
                answer=llm_response['answer'],
                citations=citations,
                retrieval_stats={
                    "chunks_retrieved": len(all_chunks),
                    "agents_queried": len(retrieval_response.retrieval_results),
                    "agents_used": [sel.agent_name for sel in retrieval_response.selected_agents],
                    "query_complexity": retrieval_response.complexity.value,
                    "retrieval_time": retrieval_response.total_processing_time
                },
                llm_stats={
                    "model": llm_response['model'],
                    "input_tokens": llm_response['input_tokens'],
                    "output_tokens": llm_response['output_tokens'],
                    "total_tokens": llm_response['total_tokens'],
                    "llm_time": llm_response['processing_time']
                },
                mode=mode,
                confidence_score=confidence,
                processing_time=total_time
            )
        
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            return QAResponse(
                query=query,
                answer=f"An error occurred while processing your query: {str(e)}",
                citations={},
                retrieval_stats={},
                llm_stats={},
                mode=mode,
                confidence_score=0.0,
                processing_time=time.time() - pipeline_start
            )
    
    def _calculate_confidence(
        self,
        retrieval_response: Any,
        citations: Dict[str, Any],
        num_chunks: int
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        Factors:
        - Retrieval quality (top chunk scores)
        - Citation validity
        - Number of sources used
        
        Args:
            retrieval_response: Response from router
            citations: Citation validation results
            num_chunks: Number of chunks retrieved
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence from top retrieval scores
        top_scores = []
        for result in retrieval_response.retrieval_results:
            if result.chunks:
                top_scores.append(result.chunks[0].get('score', 0.0))
        
        avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
        
        # Citation quality factor
        citation_factor = 1.0 if citations.get('all_citations_valid', False) else 0.7
        
        # Source coverage factor (used at least 30% of retrieved chunks)
        coverage = citations.get('unique_sources_cited', 0) / max(num_chunks, 1)
        coverage_factor = min(coverage * 3, 1.0)  # Scale to 1.0 at 33% usage
        
        # Combined confidence
        confidence = (avg_top_score * 0.5 + citation_factor * 0.3 + coverage_factor * 0.2)
        
        return round(confidence, 2)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current session usage statistics"""
        if self.usage_tracker:
            return self.usage_tracker.get_session_stats()
        return {}
    
    def get_cost_summary(self) -> str:
        """Get formatted cost summary"""
        if self.usage_tracker:
            return self.usage_tracker.get_cost_summary()
        return "Usage tracking not enabled"


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # This would be used like:
    """
    from src.agents.enhanced_router import EnhancedRouter
    
    # Initialize router (already built)
    router = EnhancedRouter(
        qdrant_url="YOUR_QDRANT_URL",
        qdrant_api_key="YOUR_QDRANT_KEY"
    )
    
    # Initialize QA Pipeline
    qa_pipeline = QAPipeline(
        router=router,
        claude_api_key="YOUR_CLAUDE_KEY"
    )
    
    # Ask a question in natural language
    response = qa_pipeline.answer_query(
        "What are the responsibilities of School Management Committees under RTE Act?"
    )
    
    # Print answer
    print(f"Question: {response.query}")
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nConfidence: {response.confidence_score}")
    print(f"\nCitations: {response.citations['unique_sources_cited']} unique sources used")
    print(f"\nProcessing time: {response.processing_time:.2f}s")
    
    # Get usage stats
    print("\n" + qa_pipeline.get_cost_summary())
    """
    print("QA Pipeline module loaded successfully")

