# Answer Synthesis Module

Complete Question-Answering pipeline with LLM integration for the AI Policy Assistant.

## Overview

The synthesis module orchestrates the complete flow from natural language query to citation-backed answer:

1. **Query Processing** → Routes query to appropriate retrieval agents
2. **Context Assembly** → Formats retrieved chunks for LLM consumption
3. **Answer Generation** → Uses Claude to generate natural language answers
4. **Citation Validation** → Validates and extracts source citations
5. **Response Packaging** → Returns complete response with metadata

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          QA Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. EnhancedRouter                                             │
│     ↓                                                          │
│  2. ContextAssembler                                           │
│     ↓                                                          │
│  3. ClaudeAnswerGenerator                                      │
│     ↓                                                          │
│  4. CitationValidator                                          │
│     ↓                                                          │
│  5. QAResponse                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Main Components

### 1. QAPipeline

The main orchestrator class that ties everything together.

```python
from src.agents.enhanced_router import EnhancedRouter
from src.synthesis import QAPipeline

# Initialize router
router = EnhancedRouter(
    qdrant_url="YOUR_QDRANT_URL",
    qdrant_api_key="YOUR_QDRANT_KEY"
)

# Initialize QA Pipeline
qa_pipeline = QAPipeline(
    router=router,
    claude_api_key="YOUR_CLAUDE_KEY",
    enable_usage_tracking=True
)

# Ask a question
response = qa_pipeline.answer_query(
    query="What are the responsibilities of School Management Committees?",
    mode="normal_qa",
    top_k=10
)

print(response.answer)
print(f"Confidence: {response.confidence_score}")
print(f"Sources: {response.citations['unique_sources_cited']}")
```

**Key Features:**
- Automatic error handling and retry logic
- Usage tracking and cost estimation
- Multiple answer modes (normal_qa, detailed, concise, comparative)
- Comprehensive metadata and metrics

### 2. ContextAssembler

Formats retrieved document chunks into LLM-ready context.

```python
from src.synthesis import ContextAssembler

assembler = ContextAssembler(
    max_chunks=10,
    max_tokens=8000
)

context, sources = assembler.assemble_context(
    retrieval_results=chunks,
    max_chunks=10
)
```

**Features:**
- Token-aware context assembly
- Source metadata extraction
- Relevance-based ordering
- Truncation at token limits

### 3. ClaudeAnswerGenerator

Generates natural language answers using Claude API.

```python
from src.synthesis import ClaudeAnswerGenerator

generator = ClaudeAnswerGenerator(
    api_key="YOUR_CLAUDE_KEY",
    model="claude-sonnet-4-20250514"
)

response = generator.generate_answer(
    query="What is the RTE Act?",
    context=formatted_context,
    mode="detailed",
    max_tokens=2000
)
```

**Answer Modes:**

1. **normal_qa** - Balanced, citation-based answers
2. **detailed** - Comprehensive analysis with legal framework
3. **concise** - Quick, 2-3 sentence answers
4. **comparative** - Side-by-side comparisons

**Features:**
- Automatic retry with exponential backoff
- Temperature control (0.0 for deterministic)
- Usage tracking
- Error handling

### 4. CitationValidator

Validates and extracts citations from LLM-generated answers.

```python
from src.synthesis import CitationValidator

validator = CitationValidator()

citations = validator.validate_citations(
    answer=llm_response['answer'],
    sources=source_list
)

print(f"Valid citations: {citations['all_citations_valid']}")
print(f"Sources cited: {citations['unique_sources_cited']}")
print(f"Citation density: {citations['citation_density']:.2%}")
```

**Validation Checks:**
- Extracts [Source X] citations
- Validates citation numbers against provided sources
- Detects hallucinated citations
- Calculates citation density and coverage

### 5. UsageTracker

Tracks LLM API usage for monitoring and cost estimation.

```python
from src.synthesis import UsageTracker

tracker = UsageTracker(log_file="logs/llm_usage.jsonl")

# Automatically logs each call
tracker.log_call(
    query="Sample query",
    mode="normal_qa",
    input_tokens=1500,
    output_tokens=500,
    model="claude-sonnet-4",
    success=True,
    processing_time=2.5
)

# Get statistics
stats = tracker.get_session_stats()
print(tracker.get_cost_summary())
```

**Tracking Features:**
- Per-call logging to JSONL
- Session statistics
- Cost estimation (input: $3/M tokens, output: $15/M tokens)
- Mode-based breakdowns
- Error tracking

## Data Models

### QAResponse

Complete response object with all metadata.

```python
@dataclass
class QAResponse:
    query: str                      # Original query
    answer: str                     # Generated answer
    citations: Dict[str, Any]       # Citation analysis
    retrieval_stats: Dict[str, Any] # Retrieval metrics
    llm_stats: Dict[str, Any]       # LLM usage metrics
    mode: str                       # Answer mode used
    confidence_score: float         # 0.0 to 1.0
    processing_time: float          # Total time in seconds
```

**Methods:**
- `to_dict()` - Convert to dictionary
- `to_json()` - Convert to JSON string

## Advanced Features

### Retry Logic

Built-in retry with exponential backoff for Claude API calls:

```python
@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    exceptions=(anthropic.APIError, anthropic.RateLimitError)
)
def generate_answer(...):
    # API call with automatic retry
    pass
```

**Retry Parameters:**
- `max_retries`: Maximum retry attempts (default: 3)
- `initial_delay`: Initial wait time (default: 1.0s)
- `backoff_factor`: Multiplier for each retry (default: 2.0)
- Waits: 1s → 2s → 4s

### Confidence Scoring

Multi-factor confidence calculation:

```python
confidence = (
    avg_retrieval_score * 0.5 +    # Retrieval quality
    citation_validity * 0.3 +       # Citation correctness
    source_coverage * 0.2           # % of sources used
)
```

**Interpretation:**
- **0.7 - 1.0**: High confidence (strong retrieval, valid citations)
- **0.4 - 0.7**: Medium confidence (moderate retrieval or some issues)
- **0.0 - 0.4**: Low confidence (weak retrieval or citation problems)

### Error Handling

Comprehensive error handling at each stage:

```python
try:
    response = qa_pipeline.answer_query(query)
except anthropic.APIError as e:
    # Claude API error (automatically retried)
    logger.error(f"API error: {e}")
except Exception as e:
    # Other errors
    logger.error(f"Pipeline error: {e}")
```

All errors are:
1. Logged with full context
2. Tracked in usage statistics
3. Returned with error details in response

## Usage Examples

### Basic Query

```python
from src.agents.enhanced_router import EnhancedRouter
from src.synthesis import QAPipeline

# Setup
router = EnhancedRouter(
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY")
)

pipeline = QAPipeline(
    router=router,
    claude_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Query
response = pipeline.answer_query(
    "What are the teacher qualification requirements?"
)

print(response.answer)
```

### Detailed Analysis Mode

```python
response = pipeline.answer_query(
    query="Explain the provisions for inclusive education",
    mode="detailed",
    top_k=15
)

# Access detailed metadata
print(f"Processing time: {response.processing_time:.2f}s")
print(f"Sources retrieved: {response.retrieval_stats['chunks_retrieved']}")
print(f"Tokens used: {response.llm_stats['total_tokens']}")
print(f"Confidence: {response.confidence_score}")
```

### Comparative Analysis

```python
response = pipeline.answer_query(
    query="Compare G.O.Ms.No. 123 and G.O.Ms.No. 456",
    mode="comparative",
    top_k=20
)

# Citations include both documents
for cite_num, details in response.citations['citation_details'].items():
    print(f"Source {cite_num}: {details['document']}")
```

### Quick Answers

```python
response = pipeline.answer_query(
    query="What is the age limit for primary school admission?",
    mode="concise",
    top_k=5
)

# Concise answer (2-3 sentences)
print(response.answer)
```

### Usage Monitoring

```python
# After multiple queries
stats = pipeline.get_usage_stats()

print(f"Total queries: {stats['total_calls']}")
print(f"Total tokens: {stats['total_input_tokens'] + stats['total_output_tokens']}")
print(f"Estimated cost: ${stats['total_cost_usd']:.4f}")

# By mode breakdown
for mode, data in stats['by_mode'].items():
    print(f"{mode}: {data['calls']} calls, ${data['cost']:.4f}")

# Full summary
print(pipeline.get_cost_summary())
```

### Export Response

```python
# To JSON
json_data = response.to_json()
with open('response.json', 'w') as f:
    f.write(json_data)

# To dict
data = response.to_dict()

# Access specific fields
print(data['citations']['unique_sources_cited'])
print(data['llm_stats']['model'])
```

## Configuration

### Environment Variables

Required:
```bash
ANTHROPIC_API_KEY=your_claude_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

Optional:
```bash
# Custom usage log location
LLM_USAGE_LOG=logs/custom_usage.jsonl
```

### Pipeline Configuration

```python
pipeline = QAPipeline(
    router=router,
    claude_api_key=api_key,
    model="claude-sonnet-4-20250514",  # Claude model
    enable_usage_tracking=True,         # Track usage
    usage_log_file="logs/llm_usage.jsonl"
)
```

### Context Assembly Configuration

```python
assembler = ContextAssembler(
    max_chunks=10,      # Maximum chunks to include
    max_tokens=8000     # Token limit for context
)
```

### Answer Generation Configuration

```python
response = generator.generate_answer(
    query=query,
    context=context,
    mode="detailed",
    max_tokens=2000,     # Max answer length
    temperature=0.0      # 0.0 = deterministic
)
```

## Performance Considerations

### Token Management

- **Context Assembly**: ~8,000 tokens default (configurable)
- **Answer Generation**: ~2,000 tokens default (configurable)
- **Total per query**: ~10,000 tokens typical

**Cost per query** (Claude Sonnet 4):
- Input: 8,000 tokens × $3/M = $0.024
- Output: 500 tokens × $15/M = $0.0075
- **Total: ~$0.03 per query**

### Processing Time

Typical breakdown:
- Retrieval: 0.5-2.0s
- Context assembly: <0.1s
- LLM generation: 2-5s
- Citation validation: <0.1s
- **Total: 3-7s per query**

### Retry Strategy

With 3 retries and exponential backoff:
- Success rate: >99% (handles transient failures)
- Max wait time: 7s (1s + 2s + 4s)
- Total timeout: ~15s maximum

## Logging

All components use Python's standard logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/synthesis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('src.synthesis')
```

**Log levels:**
- `INFO`: Normal operation, query processing
- `WARNING`: Retries, token limits, invalid citations
- `ERROR`: API errors, failures
- `DEBUG`: Detailed step-by-step processing

## Testing

```python
# Test basic functionality
from src.synthesis import QAPipeline

def test_qa_pipeline():
    pipeline = QAPipeline(router=router, claude_api_key=api_key)
    
    response = pipeline.answer_query("Test query")
    
    assert response.query == "Test query"
    assert response.answer is not None
    assert response.confidence_score >= 0.0
    assert response.confidence_score <= 1.0
    assert response.processing_time > 0

# Test error handling
def test_error_handling():
    pipeline = QAPipeline(router=router, claude_api_key="invalid_key")
    
    response = pipeline.answer_query("Test query")
    
    assert "error" in response.answer.lower()
    assert response.confidence_score == 0.0
```

## Troubleshooting

### Issue: "Claude API key required"

**Solution**: Set `ANTHROPIC_API_KEY` environment variable:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Issue: Rate limit errors

**Solution**: Reduce query frequency or use backoff:
```python
pipeline = QAPipeline(
    router=router,
    claude_api_key=api_key,
    # Automatic retry handles rate limits
)
```

### Issue: Low confidence scores

**Possible causes:**
1. Query doesn't match knowledge base content
2. Weak retrieval results (low similarity scores)
3. Few or invalid citations
4. Low source coverage

**Solution**: 
- Rephrase query for better matches
- Increase `top_k` for more sources
- Check retrieval stats in response

### Issue: High token usage

**Solution**: Adjust limits:
```python
assembler = ContextAssembler(
    max_chunks=5,       # Reduce from 10
    max_tokens=4000     # Reduce from 8000
)

response = pipeline.answer_query(
    query=query,
    mode="concise",     # Use concise mode
    top_k=5             # Reduce sources
)
```

### Issue: Slow response times

**Optimization strategies:**
1. Use `concise` mode for faster answers
2. Reduce `top_k` to retrieve fewer chunks
3. Lower `max_tokens` for shorter context
4. Check network latency to Qdrant/Claude

## Related Modules

- **[agents/enhanced_router.py](../agents/enhanced_router.py)** - Multi-agent retrieval system
- **[query_processing/](../query_processing/)** - Query normalization and enhancement
- **[retrieval/](../retrieval/)** - Vector search and ranking
- **[evaluation/](../evaluation/)** - Quality metrics and testing

## API Reference

See individual class docstrings for detailed API documentation:

```python
help(QAPipeline)
help(ClaudeAnswerGenerator)
help(ContextAssembler)
help(CitationValidator)
help(UsageTracker)
```

## Contributing

When adding new features:

1. **Add type hints** to all functions
2. **Include error handling** with appropriate logging
3. **Update tests** for new functionality
4. **Document** in docstrings and this README
5. **Track usage** if adding LLM calls

## License

Part of the AI Policy Assistant project.

