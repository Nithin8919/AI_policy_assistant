# Technical Architecture

## System Architecture

The AI Policy Assistant uses a multi-layered architecture combining:
- Vector search (semantic similarity)
- Keyword filtering (metadata-based)
- Knowledge graph (document relations)
- Multi-agent retrieval (specialized agents)

## Components

### 1. Data Ingestion Layer
- PDF extraction and text cleaning
- Section parsing and chunking
- Metadata extraction
- Entity recognition

### 2. Embedding Layer
- Vector generation using all-MiniLM-L6-v2
- Batch processing for efficiency
- Qdrant vector store integration

### 3. Query Processing Layer
- Query normalization
- Entity extraction
- Intent classification
- Query expansion
- Context injection

### 4. Agent System
- Legal Agent (Acts & Rules)
- GO Agent (Government Orders)
- Judicial Agent (Case Law)
- Data Agent (Statistics)
- Internet Agent (Web Search)

### 5. Retrieval Layer
- Hybrid search (vector + keyword)
- Reranking for relevance and diversity
- Bridge table lookups

### 6. Synthesis Layer
- Answer generation
- Citation formatting
- Confidence scoring
- Fact verification

## Operational Modes

### Mode 1: Normal QA
- Fast, precise answers
- Trusted corpus only
- Single-agent focused retrieval

### Mode 2: Brainstorming
- Broad, exploratory answers
- Multiple agents
- External sources included

### Mode 3: Pro Deepthink
- Comprehensive analysis
- All agents active
- Extended context
- Fact verification

## Data Flow

1. User query → Query Processing Pipeline
2. Mode detection → Agent Router
3. Agent selection → Parallel retrieval
4. Results aggregation → Synthesis
5. Answer generation → Response


