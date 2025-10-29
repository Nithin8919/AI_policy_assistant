# QUICKSTART: Phase 2 - Embedding Generation & Vector Storage

This guide walks you through Phase 2 of the AP Policy Co-Pilot project: converting your processed document chunks into searchable vector embeddings.

## 🎯 What Phase 2 Does

Phase 2 takes the processed chunks from Phase 1 and:

1. **Generates embeddings** using sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
2. **Creates 5 specialized Qdrant collections** (one per document type)
3. **Stores embeddings with full metadata** for semantic search
4. **Provides search functionality** ready for Phase 3 query processing

## 📋 Prerequisites

### ✅ Phase 1 Complete
- Document ingestion pipeline has run successfully
- Chunks are available in `data/processed/chunks/all_chunks.jsonl`
- Approximately 4,200+ chunks from 233 documents

### 🛠 Required Dependencies

Install Python packages:
```bash
pip install sentence-transformers qdrant-client numpy tqdm
```

### 🚀 Qdrant Vector Database

**Option A: Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Local Installation**
```bash
# On macOS
brew install qdrant

# On Ubuntu
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz
./qdrant
```

**Option C: Qdrant Cloud**
- Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
- Get your cluster URL and API key

## ⚙️ Configuration

### 1. Environment Variables

Copy and configure your environment:
```bash
cp .env.example .env
```

Edit `.env` with your Qdrant settings:
```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Leave empty for local Qdrant

# Embedding Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32
EMBEDDING_CHECKPOINT_INTERVAL=100

# Vector Store Settings
VECTOR_STORE_BATCH_SIZE=100
VECTOR_STORE_COLLECTION_PREFIX=policy_assistant
```

### 2. Verify Configuration

Test your setup:
```bash
python scripts/generate_embeddings.py --test-connection
```

Expected output:
```
✓ Vector store connection successful
Collections: 0
```

## 🚀 Running Phase 2

### Step 1: Generate Embeddings

**Full Generation (All Document Types)**
```bash
python scripts/generate_embeddings.py
```

**Quick Test (5 Documents)**
```bash
python scripts/generate_embeddings.py --chunks-file data/processed/chunks/sample_chunks.jsonl
```

**Custom Configuration**
```bash
python scripts/generate_embeddings.py \
    --batch-size 64 \
    --checkpoint-interval 50 \
    --recreate-collections
```

### Step 2: Monitor Progress

The script provides real-time progress:
```
Loading chunks from data/processed/chunks/all_chunks.jsonl
Loaded 4,203 chunks
Chunks grouped by document type:
  acts: 1,250 chunks
  government_orders: 1,800 chunks
  judicial_documents: 650 chunks
  data_reports: 350 chunks
  external_sources: 153 chunks

Generating embeddings: 100%|██████████| 4203/4203 [03:45<00:00, 18.6it/s]
```

### Step 3: Verify Results

```bash
python scripts/generate_embeddings.py --validate
```

Expected output:
```
✓ policy_assistant_legal_documents: 1250 embeddings found
✓ policy_assistant_government_orders: 1800 embeddings found
✓ policy_assistant_judicial_documents: 650 embeddings found
✓ policy_assistant_data_reports: 350 embeddings found
✓ policy_assistant_external_sources: 153 embeddings found
```

## 🔍 Testing the System

### Basic Functionality Test

```bash
python scripts/test_embeddings.py
```

This runs:
- ✅ Embedder functionality test
- ✅ Vector store connectivity test  
- ✅ Integration test (end-to-end)
- ✅ Performance test (100 embeddings)

### Search Functionality Test

```bash
python scripts/test_search.py
```

This tests semantic search with education policy queries:
- "Right to Education Act implementation"
- "teacher recruitment and DSC notification"
- "Nadu Nedu school infrastructure program"
- "Jagananna Amma Vodi scholarship scheme"
- And more...

### Advanced Search Tests

```bash
# Test filtered search (by metadata)
python scripts/test_search.py --test-type filtered

# Test document type search
python scripts/test_search.py --test-type document_type

# Test with custom parameters
python scripts/test_search.py --top-k 10 --output results/my_search_test.json
```

## 📊 What You Get

### 5 Qdrant Collections

| Collection | Document Types | Purpose |
|------------|----------------|---------|
| `policy_assistant_legal_documents` | Acts, Rules | Legal framework search |
| `policy_assistant_government_orders` | GOs, Circulars | Policy implementation |
| `policy_assistant_judicial_documents` | Court cases | Legal precedents |
| `policy_assistant_data_reports` | Statistics, Budgets | Data-driven insights |
| `policy_assistant_external_sources` | Research, Standards | External references |

### Rich Metadata

Each embedding includes:
```json
{
  "chunk_id": "doc_123_chunk_5",
  "doc_id": "AP_Education_Act_2023",
  "content": "Section 15: Right to Quality Education...",
  "doc_type": "acts",
  "section_id": "section_15", 
  "year": 2023,
  "priority": "critical",
  "token_count": 156,
  "content_hash": "sha256_hash"
}
```

### Search Capabilities

**Semantic Search**
```python
from src.embeddings.vector_store import VectorStore, VectorStoreConfig
from src.embeddings.embedder import Embedder

# Initialize
embedder = Embedder()
vector_store = VectorStore(VectorStoreConfig())

# Search
query_result = embedder.embed_single("teacher qualification requirements")
results = vector_store.search(query_result.embedding, limit=10)
```

**Filtered Search**
```python
# Search specific document types
results = vector_store.search_by_document_type(
    query_vector=query_embedding,
    doc_types=["acts", "government_orders"],
    limit=10
)

# Search with metadata filters
results = vector_store.search(
    query_vector=query_embedding,
    filters={"year": {"range": {"gte": 2020}}, "priority": "critical"},
    limit=10
)
```

## 📈 Performance Expectations

### Embedding Generation
- **Speed**: ~20-30 embeddings/second (batch size 32)
- **Time**: ~3-5 minutes for 4,200 chunks
- **Memory**: ~2-4 GB RAM during generation
- **Storage**: ~50-100 MB for all embeddings

### Search Performance
- **Latency**: <100ms for typical queries
- **Throughput**: >100 searches/second
- **Accuracy**: >85% relevant results for policy queries

## 🛠 Troubleshooting

### Common Issues

**1. Qdrant Connection Failed**
```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# Start Qdrant if needed
docker run -p 6333:6333 qdrant/qdrant
```

**2. No Chunks Found**
```bash
# Check if Phase 1 completed
ls -la data/processed/chunks/

# Run ingestion if needed
python src/ingestion/run_ingestion.py
```

**3. CUDA/GPU Issues**
```bash
# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""
python scripts/generate_embeddings.py
```

**4. Memory Issues**
```bash
# Reduce batch size
python scripts/generate_embeddings.py --batch-size 16
```

### Performance Tuning

**For Faster Generation:**
```bash
# Larger batch size (if you have more RAM)
python scripts/generate_embeddings.py --batch-size 64

# Fewer checkpoints
python scripts/generate_embeddings.py --checkpoint-interval 200
```

**For Better Search:**
```bash
# Adjust score threshold in .env
SEARCH_SCORE_THRESHOLD=0.7  # Higher = more strict
```

## 📁 Generated Files

After Phase 2 completion:
```
data/
├── embeddings/
│   ├── generation_stats.json          # Generation statistics
│   └── checkpoint_metadata.json       # Checkpoint data
├── evaluation/
│   └── search_test_results.json       # Search test results
└── processed/
    └── chunks/
        └── all_chunks.jsonl            # Original chunks (from Phase 1)
```

## ✅ Verification Checklist

- [ ] Qdrant is running and accessible
- [ ] All 5 collections created successfully
- [ ] 4,200+ embeddings stored across collections
- [ ] Search tests return relevant results
- [ ] Generation statistics look reasonable
- [ ] No errors in logs

## 🎯 What's Next: Phase 3

With Phase 2 complete, you now have:
- ✅ **Searchable vector database** with 4,200+ policy chunks
- ✅ **5 specialized collections** for different document types
- ✅ **Semantic search capability** with metadata filtering
- ✅ **Full-text search** via content indexing

**Ready for Phase 3:** Query Processing Pipeline
- 🔄 Query normalization and enhancement
- 🏷 Entity extraction (districts, schemes, metrics)
- 🧠 Intent classification
- 🔍 Multi-agent retrieval system
- 📝 Answer generation with citations

Continue to Phase 3 when ready!

---

## 🆘 Need Help?

- **Check logs**: Look in `logs/` directory for detailed error messages
- **Test components**: Use individual test scripts to isolate issues
- **Monitor resources**: Check CPU/memory usage during generation
- **Verify data**: Ensure chunks from Phase 1 are properly formatted

**Common Commands:**
```bash
# Quick health check
python scripts/generate_embeddings.py --test-connection

# Full system test
python scripts/test_embeddings.py

# Search functionality test
python scripts/test_search.py

# Regenerate if needed
python scripts/generate_embeddings.py --recreate-collections
```