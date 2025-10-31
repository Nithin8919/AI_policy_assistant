"""
Complete Question-Answering Pipeline with Multiple LLM Support
Supports Google Gemini and Groq for natural language answers
"""

import os
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import re
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)


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


class ContextAssembler:
    """Assembles retrieved chunks into LLM-friendly context"""
    
    def assemble_context(
        self, 
        retrieval_results: List[Dict],
        max_chunks: int = 10,
        max_tokens: int = 8000
    ) -> tuple[str, List[Dict]]:
        """
        Format retrieved chunks for LLM consumption.
        
        Args:
            retrieval_results: Retrieved chunks with metadata
            max_chunks: Maximum number of chunks to include
            max_tokens: Approximate token limit for context
            
        Returns:
            (formatted_context_string, source_list)
        """
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
                'chunk_id': result.get('chunk_id', '')
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


class MultiLLMAnswerGenerator:
    """Multi-LLM integration supporting Gemini and Groq"""
    
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
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        """
        Initialize Multi-LLM client.
        
        Args:
            provider: LLM provider ("gemini" or "groq")
            api_key: API key (or uses environment variables)
        """
        self.provider = provider.lower()
        
        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
            
            self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
            if not self.api_key:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY or pass api_key parameter")
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.model_name = "gemini-2.5-flash"
            
        elif self.provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("groq package not installed. Run: pip install groq")
            
            self.api_key = api_key or os.getenv('GROQ_API_KEY')
            if not self.api_key:
                raise ValueError("Groq API key required. Set GROQ_API_KEY or pass api_key parameter")
            
            self.client = Groq(api_key=self.api_key)
            self.model_name = "llama-3.1-8b-instant"  # Updated to current available model
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'groq'")
        
        logger.info(f"{self.provider.title()} Answer Generator initialized with model: {self.model_name}")
    
    def generate_answer(
        self,
        query: str,
        context: str,
        mode: str = "normal_qa",
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate natural language answer using selected LLM.
        
        Args:
            query: User's original question
            context: Assembled context from retrieval
            mode: Answer generation mode (normal_qa, detailed, concise, comparative)
            max_tokens: Maximum tokens for answer
            
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
            if self.provider == "gemini":
                return self._generate_gemini_answer(system_prompt, user_prompt, max_tokens, start_time, mode)
            elif self.provider == "groq":
                return self._generate_groq_answer(system_prompt, user_prompt, max_tokens, start_time, mode)
                
        except Exception as e:
            logger.error(f"{self.provider.title()} API error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "success": False,
                "error": str(e),
                "provider": self.provider
            }
    
    def _generate_gemini_answer(self, system_prompt: str, user_prompt: str, max_tokens: int, start_time: float, mode: str) -> Dict[str, Any]:
        """Generate answer using Google Gemini"""
        
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Configure generation
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.0,  # Deterministic for policy answers
        )
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # Handle different response formats with better error handling
        try:
            answer_text = response.text
        except Exception as e:
            # Fallback for complex responses
            try:
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        if candidate.content.parts and len(candidate.content.parts) > 0:
                            answer_text = candidate.content.parts[0].text
                        else:
                            raise ValueError("Gemini response has no content parts")
                    else:
                        raise ValueError("Gemini response has invalid content structure")
                else:
                    raise ValueError("Gemini returned no candidates")
            except Exception as fallback_error:
                logger.error(f"Failed to extract Gemini response: {fallback_error}")
                raise ValueError(f"Gemini response extraction failed: {str(fallback_error)}")
        
        processing_time = time.time() - start_time
        
        # Estimate token usage (Gemini doesn't provide exact counts in all cases)
        input_tokens = len(full_prompt) // 4  # Rough estimate
        output_tokens = len(answer_text) // 4
        
        logger.info(f"Answer generated with Gemini in {processing_time:.2f}s (~{input_tokens + output_tokens} tokens)")
        
        return {
            "answer": answer_text,
            "model": self.model_name,
            "provider": "gemini",
            "mode": mode,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "processing_time": processing_time,
            "finish_reason": response.candidates[0].finish_reason.name if response.candidates else "unknown",
            "success": True
        }
    
    def _generate_groq_answer(self, system_prompt: str, user_prompt: str, max_tokens: int, start_time: float, mode: str) -> Dict[str, Any]:
        """Generate answer using Groq"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministic for policy answers
            stream=False
        )
        
        answer_text = response.choices[0].message.content
        processing_time = time.time() - start_time
        
        logger.info(f"Answer generated with Groq in {processing_time:.2f}s using {response.usage.total_tokens} tokens")
        
        return {
            "answer": answer_text,
            "model": self.model_name,
            "provider": "groq",
            "mode": mode,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "processing_time": processing_time,
            "finish_reason": response.choices[0].finish_reason,
            "success": True
        }


class CitationValidator:
    """Validates and extracts citations from LLM answers"""
    
    def validate_citations(
        self,
        answer: str,
        sources: List[Dict]
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
        
        return {
            "total_citations": total_citations,
            "unique_sources_cited": unique_sources_cited,
            "citation_details": citation_details,
            "invalid_citations": invalid_citations,
            "all_citations_valid": all_valid,
            "citation_rate": total_citations / max(len(answer.split()), 1),  # Citations per word
            "sources_used_percentage": (unique_sources_cited / len(sources) * 100) if sources else 0
        }


class QAPipeline:
    """
    Complete Question-Answering Pipeline with Multi-LLM support.
    
    This is the main class that ties everything together:
    1. Takes a natural language query
    2. Uses EnhancedRouter for retrieval
    3. Assembles context for LLM
    4. Generates answer with Gemini/Groq
    5. Validates citations
    6. Returns complete response
    """
    
    def __init__(
        self,
        router,  # EnhancedRouter instance
        llm_provider: str = "gemini",
        api_key: Optional[str] = None
    ):
        """
        Initialize QA Pipeline.
        
        Args:
            router: EnhancedRouter instance (already initialized with Qdrant)
            llm_provider: LLM provider ("gemini" or "groq")
            api_key: API key for the selected provider
        """
        self.router = router
        self.assembler = ContextAssembler()
        self.generator = MultiLLMAnswerGenerator(llm_provider, api_key)
        self.validator = CitationValidator()
        
        logger.info(f"QA Pipeline initialized with {llm_provider}")
    
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
                citations={"total_citations": 0, "unique_sources_cited": 0, "all_citations_valid": True},
                retrieval_stats={"chunks_retrieved": 0, "agents_queried": 0},
                llm_stats={},
                mode=mode,
                confidence_score=0.0,
                processing_time=time.time() - pipeline_start
            )
        
        # Step 2: Assemble context for LLM
        logger.info("Step 2: Assembling context...")
        context, sources = self.assembler.assemble_context(all_chunks, max_chunks=top_k)
        
        # Step 3: Generate answer using selected LLM
        logger.info(f"Step 3: Generating answer with {self.generator.provider}...")
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
                "provider": llm_response['provider'],
                "input_tokens": llm_response['input_tokens'],
                "output_tokens": llm_response['output_tokens'],
                "total_tokens": llm_response['total_tokens'],
                "llm_time": llm_response['processing_time']
            },
            mode=mode,
            confidence_score=confidence,
            processing_time=total_time
        )
    
    def _calculate_confidence(
        self,
        retrieval_response,
        citations: Dict,
        num_chunks: int
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        Factors:
        - Retrieval quality (top chunk scores)
        - Citation validity
        - Number of sources used
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


# Convenience function for quick testing
def test_llm_providers():
    """Test available LLM providers"""
    results = {}
    
    # Test Gemini
    if GEMINI_AVAILABLE and os.getenv('GOOGLE_API_KEY'):
        try:
            generator = MultiLLMAnswerGenerator("gemini")
            test_response = generator.generate_answer(
                "What is the capital of India?",
                "[Source 1]\nDocument: Test\nContent: New Delhi is the capital of India.",
                mode="concise",
                max_tokens=100
            )
            results['gemini'] = test_response.get('success', False)
        except Exception as e:
            results['gemini'] = f"Error: {e}"
    else:
        results['gemini'] = "Not available (missing package or API key)"
    
    # Test Groq
    if GROQ_AVAILABLE and os.getenv('GROQ_API_KEY'):
        try:
            generator = MultiLLMAnswerGenerator("groq")
            test_response = generator.generate_answer(
                "What is the capital of India?",
                "[Source 1]\nDocument: Test\nContent: New Delhi is the capital of India.",
                mode="concise",
                max_tokens=100
            )
            results['groq'] = test_response.get('success', False)
        except Exception as e:
            results['groq'] = f"Error: {e}"
    else:
        results['groq'] = "Not available (missing package or API key)"
    
    return results


# Example usage
if __name__ == "__main__":
    print("Multi-LLM QA Pipeline module loaded successfully")
    
    # Test provider availability
    print("\nTesting LLM providers...")
    test_results = test_llm_providers()
    for provider, result in test_results.items():
        print(f"{provider.title()}: {result}")