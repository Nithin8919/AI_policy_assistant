# AI Policy Assistant 📚

An intelligent question-answering system for Andhra Pradesh education policies, laws, and government orders. Get accurate, citation-backed answers from official documents using advanced retrieval and AI-powered synthesis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

The AI Policy Assistant helps policymakers, educators, administrators, and researchers quickly find accurate information from thousands of pages of education policy documents. Instead of manually searching through PDFs, users can ask natural language questions and receive comprehensive, citation-backed answers.

### Key Features

- 🔍 **Natural Language Queries** - Ask questions in plain English
- 📚 **Citation-Based Answers** - Every claim backed by source documents
- 🤖 **Multi-Agent Retrieval** - Specialized agents for different document types
- 💡 **Multiple Answer Modes** - From quick answers to detailed analysis
- 📊 **Rich Metadata** - Confidence scores, processing times, token usage
- 🎨 **Interactive UI** - Beautiful Streamlit interface with visualizations
- 🚀 **REST API** - Programmatic access for integrations
- 💰 **Usage Tracking** - Monitor LLM calls and costs

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit UI (ui/streamlit_app.py)  │  REST API (api/main.py) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QA Pipeline (src/synthesis/)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Query Processing & Normalization                            │
│  2. Multi-Agent Retrieval (EnhancedRouter)                      │
│  3. Context Assembly                                            │
│  4. LLM Answer Generation (Claude)                              │
│  5. Citation Validation                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Specialized Agents                          │
├─────────────────────────────────────────────────────────────────┤
│  📋 GO Agent  │  ⚖️ Legal Agent  │  🎓 Scheme Agent  │  📊 Data │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Vector Database (Qdrant)                    │
├─────────────────────────────────────────────────────────────────┤
│  • 10,000+ document chunks                                      │
│  • Semantic embeddings (sentence-transformers)                  │
│  • Metadata filtering (type, year, section)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Qdrant vector database (local or cloud)
- Anthropic API key (for Claude)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-policy-assistant.git
cd ai-policy-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env with your credentials
ANTHROPIC_API_KEY=your_claude_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

5. **Initialize the database** (if starting fresh)
```bash
# Process documents and generate embeddings
python scripts/run_enhanced_ingestion_pipeline.py

# Generate embeddings and upload to Qdrant
python scripts/generate_embeddings.py
```

### Running the System

#### Option 1: Streamlit UI (Recommended for exploration)

```bash
streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

#### Option 2: REST API (Recommended for integrations)

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation available at http://localhost:8000/docs

#### Option 3: Python SDK (Recommended for scripts)

```python
from src.agents.enhanced_router import EnhancedRouter
from src.synthesis import QAPipeline
import os

# Initialize
router = EnhancedRouter(
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY")
)

pipeline = QAPipeline(
    router=router,
    claude_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Ask a question
response = pipeline.answer_query(
    "What are the eligibility criteria for the Amma Vodi scheme?"
)

print(response.answer)
print(f"Confidence: {response.confidence_score}")
```

---

## 💬 Usage Examples

### Example 1: Simple Query

**Query:** "What is the Right to Education Act?"

**Response:**
```
The **Right to Education (RTE) Act, 2009** is a landmark legislation that:

• Provides free and compulsory education to all children aged 6-14 years [Source 1]
• Mandates 25% reservation for economically weaker sections in private schools [Source 2]
• Establishes norms and standards for schools [Source 1]
• Prohibits physical punishment and mental harassment [Source 3]

The Act came into effect on April 1, 2010, making education a fundamental right 
under Article 21A of the Constitution [Source 1].

Sources:
1. AP RTE Rules 2010, Section 2
2. G.O.Ms.No. 456/2011, Clause 3.1
3. RTE Act 2009, Section 17
```

### Example 2: Detailed Analysis

**Query:** "Explain the teacher recruitment process in detail"

**Mode:** `detailed`

**Response:**
```
# Teacher Recruitment Process in Andhra Pradesh

## Legal Framework
The recruitment of teachers in government schools is governed by:
• **AP Education Act 1982** [Source 1]
• **G.O.Ms.No. 123/2019** - Teacher Recruitment Guidelines [Source 2]
• **DSC Rules 2022** [Source 3]

## Eligibility Criteria
### Educational Qualifications
- Bachelor's degree with 50% marks [Source 2]
- B.Ed or equivalent teacher training certification [Source 2]
- TET/CTET qualification mandatory [Source 3]

### Age Limits
- Minimum: 18 years
- Maximum: 42 years (with relaxations for SC/ST/BC) [Source 4]

## Selection Process
1. **Written Examination (DSC)** - 150 marks [Source 5]
2. **Document Verification** [Source 5]
3. **Medical Fitness Certificate** [Source 6]
4. **Final Merit List Publication** [Source 5]

[... detailed analysis continues ...]
```

### Example 3: Comparative Analysis

**Query:** "Compare G.O.Ms.No. 123 and G.O.Ms.No. 456"

**Mode:** `comparative`

**Response:**
```
# Comparison: G.O.Ms.No. 123 vs G.O.Ms.No. 456

| Aspect | G.O.Ms.No. 123 (2019) | G.O.Ms.No. 456 (2020) |
|--------|----------------------|----------------------|
| Purpose | Amma Vodi Launch | Amma Vodi Extension |
| Budget | ₹3,000 Cr [Source 1] | ₹4,500 Cr [Source 2] |
| Beneficiaries | 43 lakh families [Source 1] | 52 lakh families [Source 2] |
| Amount | ₹15,000/year [Source 1] | ₹15,000/year [Source 2] |

## Key Differences
- G.O. 456 extended eligibility to include vocational courses [Source 2]
- G.O. 456 relaxed income criteria from ₹1.5L to ₹2L [Source 2]
- G.O. 456 introduced online application process [Source 3]

[... detailed comparison continues ...]
```

### Example 4: Quick Answer

**Query:** "What is the age limit for primary school admission?"

**Mode:** `concise`

**Response:**
```
Children aged **6-7 years** are eligible for Class 1 admission, with a 
relaxation of ±1 year based on the academic calendar [Source 1].
```

---

## 🎛️ Answer Modes

The system supports multiple answer modes to suit different use cases:

| Mode | Description | Use Case | Response Length |
|------|-------------|----------|-----------------|
| **normal_qa** | Balanced answers with citations | General queries | 200-500 words |
| **detailed** | Comprehensive analysis with legal framework | Complex policy questions | 500-1000 words |
| **concise** | Quick, direct answers | Simple factual queries | 2-3 sentences |
| **comparative** | Side-by-side comparisons | Comparing documents/provisions | 300-600 words |

### Choosing the Right Mode

- Use **normal_qa** (default) for most questions
- Use **detailed** when you need comprehensive understanding with legal citations
- Use **concise** when you just need a quick fact
- Use **comparative** when analyzing differences between documents or provisions

---

## 🎨 Streamlit UI Features

The interactive UI provides:

### Query Interface
- 📝 Natural language text input
- 🎛️ Mode selection dropdown
- ⚙️ Advanced options (top_k, filters)
- 💡 Example queries for inspiration

### Results Display
- 📄 Formatted answer with markdown rendering
- 📊 Confidence score with visual indicator
- ⏱️ Processing time and performance metrics
- 📚 Interactive citation cards with expandable details

### Citation Visualization
- 📈 Relevance score bar chart
- 🔗 Citation network diagram
- 📑 Source document details
- ✅ Citation validation status

### Usage Tracking
- 📊 Session statistics dashboard
- 💰 Cost estimation in real-time
- 📈 Query history with mode breakdown
- 💾 Export usage data (JSON/CSV)

### Example Features
- 🔍 Search history with quick reload
- 💾 Export answers (JSON/Markdown)
- 🔄 Reinitialize system without restart
- 📋 Copy citation details

---

## 🔌 REST API

### Authentication

All API requests require an API key:

```bash
curl -H "X-API-Key: your_api_key" \
     -X POST http://localhost:8000/query \
     -d '{"query": "What is RTE Act?"}'
```

### Query Endpoint

**POST** `/query`

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    headers={"X-API-Key": "your_api_key"},
    json={
        "query": "What are teacher qualification requirements?",
        "mode": "detailed",
        "top_k": 15
    }
)

data = response.json()
print(data["answer"])
print(f"Confidence: {data['confidence_score']}")
print(f"Citations: {data['citations']['unique_sources_cited']}")
```

### Batch Processing

```python
response = requests.post(
    "http://localhost:8000/query/batch",
    headers={"X-API-Key": "your_api_key"},
    json={
        "queries": [
            {"query": "What is RTE Act?", "mode": "concise"},
            {"query": "Teacher recruitment process", "mode": "normal_qa"}
        ]
    }
)
```

See [api/README.md](api/README.md) for complete API documentation.

---

## 📊 Knowledge Base

The system indexes the following document types:

### Government Orders (GOs)
- **Count**: 3,500+ documents
- **Coverage**: 1990-2024
- **Topics**: Policy implementation, administrative decisions, schemes

### Legal Documents
- **Acts**: 45+ education-related acts
- **Rules**: 200+ rule sets
- **Sections**: Cross-referenced and indexed

### Schemes
- **Active Schemes**: 50+ major schemes
- **Details**: Eligibility, benefits, implementation guidelines
- **Examples**: Amma Vodi, Vidya Deevena, Jagananna Amma Vodi

### Case Law (Judicial)
- **Cases**: 1,000+ education-related judgments
- **Courts**: Supreme Court, High Courts
- **Topics**: Admissions, reservations, teacher disputes

### Data Reports
- **Enrollment statistics**
- **Infrastructure metrics**
- **Performance indicators**

---

## 🔧 Configuration

### Query Processing

```python
# Custom configuration
pipeline = QAPipeline(
    router=router,
    claude_api_key=api_key,
    model="claude-sonnet-4-20250514",      # Claude model
    enable_usage_tracking=True,             # Track costs
    usage_log_file="logs/custom_usage.jsonl"
)

# Query with options
response = pipeline.answer_query(
    query="Your question here",
    mode="detailed",
    top_k=15  # Number of document chunks
)
```

### Context Assembly

```python
from src.synthesis import ContextAssembler

assembler = ContextAssembler(
    max_chunks=10,      # Maximum chunks to include
    max_tokens=8000     # Approximate token limit
)
```

### Retrieval Filters

```python
# Filter by document type, year, or other metadata
router.route_query(
    query="teacher recruitment",
    filters={
        "doc_type": "government_order",
        "year_min": 2020,
        "year_max": 2024
    }
)
```

---

## 📈 Performance & Costs

### Response Times

| Configuration | Avg Time | Range |
|---------------|----------|-------|
| Concise, top_k=5 | 2.5s | 2-3s |
| Normal, top_k=10 | 4.0s | 3-5s |
| Detailed, top_k=15 | 6.5s | 5-8s |

### Token Usage

- **Input tokens**: ~8,000 per query (context + prompt)
- **Output tokens**: ~500 per query (answer)
- **Total**: ~8,500 tokens per query

### Cost Estimation (Claude Sonnet 4)

- **Input**: $3 per million tokens
- **Output**: $15 per million tokens
- **Per query**: ~$0.03
- **1,000 queries**: ~$30

### Optimization Tips

1. Use `concise` mode for lower costs
2. Reduce `top_k` for faster responses
3. Enable caching for repeated queries
4. Use batch API for multiple queries

---

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_synthesis.py -v
pytest tests/test_agents.py -v
pytest tests/test_query_processing.py -v
```

### Integration Tests

```bash
# Test complete pipeline
pytest tests/integration/test_complete_system.py -v

# Test with real queries
python scripts/test_complete_system.py
```

### Evaluation

```bash
# Run evaluation on test set
python scripts/evaluate_system.py

# Results include:
# - Answer quality scores
# - Citation accuracy
# - Response times
# - Cost per query
```

---

## 📁 Project Structure

```
ai-policy-assistant/
├── api/                          # REST API
│   ├── main.py                  # FastAPI application
│   ├── routes/                  # API endpoints
│   ├── middleware/              # Auth, logging
│   └── README.md               # API documentation
├── src/
│   ├── agents/                  # Specialized retrieval agents
│   │   ├── enhanced_router.py  # Multi-agent orchestrator
│   │   └── ...
│   ├── synthesis/               # Answer generation pipeline
│   │   ├── qa_pipeline.py      # Main QA pipeline
│   │   ├── README.md           # Module documentation
│   │   └── ...
│   ├── query_processing/        # Query normalization
│   ├── retrieval/              # Vector search
│   ├── ingestion/              # Document processing
│   └── evaluation/             # Quality metrics
├── ui/
│   └── streamlit_app.py        # Streamlit interface
├── data/
│   ├── raw/                    # Original documents
│   ├── processed/              # Processed documents
│   ├── embeddings/             # Vector embeddings
│   └── dictionaries/           # Acronyms, synonyms, etc.
├── scripts/                     # Utility scripts
├── tests/                       # Test suite
├── docs/                        # Documentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔐 Security & Privacy

### Data Security
- All document processing happens locally
- No document content sent to external services (except for embeddings)
- API keys stored in environment variables only

### API Security
- API key authentication required
- Rate limiting enabled
- Request logging with privacy controls
- HTTPS recommended for production

### Query Privacy
- Queries logged locally only (opt-in)
- Usage logs truncate sensitive information
- Configurable log retention policies

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Claude API key required"
```bash
# Solution: Set environment variable
export ANTHROPIC_API_KEY=your_key_here
```

**Issue**: "Connection to Qdrant failed"
```bash
# Solution: Verify Qdrant is running
curl $QDRANT_URL/health

# Or start local Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**Issue**: Low confidence scores
- Try rephrasing your query
- Use more specific terminology
- Check if topic is covered in knowledge base
- Increase `top_k` for more sources

**Issue**: Slow response times
- Use `concise` mode for faster answers
- Reduce `top_k` to retrieve fewer chunks
- Check network latency to Qdrant/Claude
- Enable response caching

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more issues and solutions.

---

## 🚀 Roadmap

### Completed ✅
- [x] Multi-agent retrieval system
- [x] Claude integration with retry logic
- [x] Citation validation
- [x] Usage tracking and cost estimation
- [x] Streamlit UI with visualizations
- [x] REST API with authentication
- [x] Comprehensive documentation

### In Progress 🔄
- [ ] Advanced caching strategies
- [ ] WebSocket streaming for long queries
- [ ] Fine-tuned embeddings for domain
- [ ] Expanded knowledge graph integration

### Planned 📋
- [ ] Multi-language support (Telugu, Hindi)
- [ ] Voice query interface
- [ ] Mobile app
- [ ] Automated document ingestion pipeline
- [ ] User feedback collection
- [ ] A/B testing framework

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linters
flake8 src/ tests/
black src/ tests/
mypy src/

# Run tests
pytest tests/ -v --cov=src
```

---

## 📚 Documentation

- **[Setup Guide](docs/setup_guide.md)** - Detailed installation instructions
- **[API Documentation](api/README.md)** - REST API reference
- **[Synthesis Module](src/synthesis/README.md)** - QA pipeline documentation
- **[Evaluation Methodology](docs/evaluation_methodology.md)** - Quality metrics
- **[Data Sources](docs/data_sources.md)** - Document coverage details
- **[Testing Guide](docs/TESTING_GUIDE.md)** - How to test the system

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Anthropic** - Claude API for answer generation
- **Qdrant** - Vector database for semantic search
- **Sentence Transformers** - Embedding models
- **FastAPI & Streamlit** - Web frameworks

---

## 📧 Contact

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/ai-policy-assistant/issues)
- **Email**: your.email@example.com
- **Documentation**: [Full documentation](docs/)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star on GitHub!

---

<div align="center">
  <strong>Built with ❤️ for better education policy access in Andhra Pradesh</strong>
</div>
