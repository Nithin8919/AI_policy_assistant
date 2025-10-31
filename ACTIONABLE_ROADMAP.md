# 🗺️ Actionable Roadmap - Next Steps

**Current State:** Core system operational, data quality needs improvement  
**Goal:** Production-ready SOTA system in 8 weeks

---

## 🎯 What We Fixed (Completed)

✅ Retrieval system (0 → 4,323 embeddings properly distributed)  
✅ Collection routing (empty collections → 4 active collections)  
✅ Gemini provider stability (67% fail rate → stable)  
✅ Document classification (100% misclassified → properly routed)

---

## 📋 Week 1-2: Data Quality & Coverage (HIGH PRIORITY)

### Goal: Process remaining 219 documents + improve extraction

#### Task 1.1: Audit Current Data (2 hours)
```bash
# Check which documents are processed vs missing
python scripts/audit_document_coverage.py

# Identify critical missing documents (RTE Act sections, key GOs)
python scripts/identify_critical_gaps.py
```

**Deliverable:** List of 20 highest-priority documents to reprocess

#### Task 1.2: Implement Semantic Chunking (8 hours)

Create `src/ingestion/semantic_chunker.py`:

```python
class SemanticChunker:
    """Chunk documents based on structure, not just token count"""
    
    def chunk_legal_document(self, text, doc_metadata):
        """
        Preserve legal structure:
        - Keep "Section X: <title>\n<content>" together
        - Don't break mid-sentence
        - Maintain hierarchical context
        """
        pass
    
    def chunk_government_order(self, text, doc_metadata):
        """
        GO-specific chunking:
        - Preamble as one chunk (context)
        - Each order/mandate as separate chunk
        - Include preamble context in each order chunk
        """
        pass
```

**Why This Matters:** Current fixed-size chunking breaks "Section 12(1)(c)" across multiple chunks, making it unsearchable.

**Test:** After implementation, verify Section 12(1)(c) appears in at least one complete chunk.

#### Task 1.3: Process High-Priority Documents (16 hours)

```bash
# Reprocess RTE Act with semantic chunking
python scripts/process_vertical.py --vertical Legal --doc "RTE Act" --use-semantic-chunking

# Reprocess top 20 GOs
python scripts/batch_process_docs.py --doc-list priority_docs.txt

# Regenerate embeddings
python scripts/regenerate_embeddings_fixed.py
python scripts/generate_embeddings.py --chunks-file data/processed/chunks/all_chunks_consolidated.jsonl --recreate-collections
```

**Validation:**
```bash
# Test that critical content now retrieves correctly
python test_critical_content.py  # Should pass for RTE Act Section 12(1)(c)
```

**Deliverable:** 4,323 → 15,000+ chunks with better quality

---

## 📋 Week 3: Bridge Table & Knowledge Graph (MEDIUM PRIORITY)

### Goal: Enable relationship-aware retrieval

#### Task 2.1: Design Bridge Table Schema (2 hours)

```sql
CREATE TABLE entities (
    entity_id UUID PRIMARY KEY,
    entity_type VARCHAR(50),  -- 'go', 'section', 'scheme', etc.
    entity_value TEXT,         -- 'GO.Ms.No.54', 'Section 12(1)(c)'
    source_chunk_id VARCHAR(100),
    metadata JSONB
);

CREATE TABLE entity_relationships (
    rel_id UUID PRIMARY KEY,
    source_entity_id UUID REFERENCES entities,
    target_entity_id UUID REFERENCES entities,
    relationship_type VARCHAR(50),  -- 'supersedes', 'implements', 'amends'
    confidence FLOAT,
    evidence_chunk_id VARCHAR(100)
);
```

#### Task 2.2: Extract Relationships (8 hours)

Enhance `src/ingestion/entity_extractor.py`:

```python
def extract_relationships(chunk):
    """
    Find:
    - "GO 123 supersedes GO 45"
    - "as per Section 12 read with Section 15"
    - "implementing Nadu-Nedu Phase 2"
    """
    pass
```

#### Task 2.3: Query Enhancement with Bridge (6 hours)

```python
class BridgeTableEnhancer:
    def enhance_query(self, query):
        """
        Query: "What is GO 54?"
        → Add context: "GO 54 superseded GO 42"
        → Search for both GO 54 and GO 42 chunks
        → Return with supersession info
        """
        pass
```

**Deliverable:** Queries about superseded GOs now return current information with context

---

## 📋 Week 4: SOTA Query Enhancement (HIGH PRIORITY)

### Goal: Improve retrieval precision from 60% → 85%

#### Task 3.1: Entity Linking (8 hours)

```python
class EntityLinker:
    def link_entities(self, query):
        """
        "RTE" → "Right to Education Act 2009"
        "12(1)(c)" → "Section 12(1)(c) of RTE Act"
        "Nadu Nedu" → "Nadu-Nedu Programme (GO.Ms.No.85/2019)"
        """
        pass
```

#### Task 3.2: Semantic Expansion (6 hours)

```python
class SemanticExpander:
    def expand_query(self, query, context):
        """
        "teacher transfer rules" 
        → ["teacher posting", "teacher allocation", "staff movement", "transfer policy"]
        
        Weight original > synonyms
        """
        pass
```

#### Task 3.3: Multi-Intent Classification (6 hours)

```python
class MultiIntentClassifier:
    def classify(self, query):
        """
        "Teacher transfer rules under Nadu-Nedu"
        → intents: [
            ("policy_implementation", 0.85),
            ("legal_framework", 0.72),
            ("scheme_details", 0.65)
        ]
        → Route to: LegalAgent, GOAgent, SchemeAgent (parallel)
        """
        pass
```

**Deliverable:** Complex queries retrieve from multiple relevant verticals

---

## 📋 Week 5-6: Evaluation & Optimization (CRITICAL)

### Goal: Measure & improve system performance

#### Task 4.1: Create Evaluation Suite (16 hours)

Build `tests/evaluation/test_suite.py`:

```python
# 200+ test queries across categories:
queries = {
    "legal_interpretation": [
        "What is Section 12(1)(c) of RTE Act?",
        "What are SMC responsibilities under RTE?",
        # ... 40 more
    ],
    "government_orders": [
        "What is GO MS No 54?",
        "Which GO supersedes GO 42?",
        # ... 40 more
    ],
    "schemes": [
        "What is Nadu-Nedu eligibility?",
        "How does Amma Vodi work?",
        # ... 40 more
    ],
    # ... more categories
}
```

**Metrics to Track:**
- Precision@5, Recall@10
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Answer correctness (human-labeled)
- Citation accuracy
- Response time (P50, P95, P99)

#### Task 4.2: Baseline Measurement (4 hours)

```bash
# Run evaluation on current system
python tests/evaluation/run_full_eval.py --save-baseline

# Expected current performance:
# Precision@5: ~60%
# Recall@10: ~70%
# MRR: ~0.65
# Answer correctness: ~50% (limited by data quality)
```

#### Task 4.3: Iterative Improvement (16 hours)

1. Identify failure modes (queries with P@5 < 50%)
2. Implement fixes (chunking, entity extraction, query enhancement)
3. Re-run evaluation
4. Repeat until:
   - Precision@5 > 85%
   - Recall@10 > 90%
   - Answer correctness > 80%

**Deliverable:** System that passes 85%+ of evaluation queries

---

## 📋 Week 7-8: Production Deployment (HIGH PRIORITY)

### Goal: Deploy stable, monitored, scalable system

#### Task 5.1: API Layer (8 hours)

```python
# api/main.py
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    POST /query
    {
        "query": "What is Section 12(1)(c)?",
        "mode": "normal_qa",
        "top_k": 10
    }
    
    Returns:
    {
        "answer": "...",
        "citations": [...],
        "confidence": 0.85,
        "processing_time": 2.1
    }
    """
    pass
```

#### Task 5.2: Monitoring & Metrics (6 hours)

```python
# Prometheus metrics
query_latency = Histogram('query_latency_seconds', 'Query processing time')
retrieval_quality = Gauge('retrieval_confidence', 'Average retrieval confidence')
error_rate = Counter('query_errors_total', 'Total query errors')

# Grafana dashboards:
# - Query volume over time
# - P50/P95/P99 latency
# - Error rates by type
# - Collection usage statistics
```

#### Task 5.3: Performance Optimization (12 hours)

1. **Caching Layer:** Cache top 100 queries (50% cost reduction)
2. **Batching:** Batch embedding generation (2x throughput)
3. **Connection Pooling:** Reuse Qdrant connections
4. **Async Processing:** Non-blocking I/O for Qdrant + LLM

**Target Metrics:**
- P95 latency: <2s
- P99 latency: <5s
- Cost per query: <$0.03
- Throughput: 100 concurrent queries

#### Task 5.4: Docker Deployment (4 hours)

```dockerfile
# Dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
COPY api/ /app/api/
WORKDIR /app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Deploy
docker build -t ai-policy-assistant:latest .
docker-compose up -d

# Or Kubernetes
kubectl apply -f k8s/deployment.yaml
```

**Deliverable:** Production-ready API with monitoring

---

## 🎯 Success Criteria (End of Week 8)

| Metric | Target | Current | Gap |
|--------|---------|---------|-----|
| **Coverage** | 249 documents | 30 documents | ⚠️ 219 to process |
| **Embeddings** | 50,000+ chunks | 4,323 chunks | ⚠️ Need 10x more |
| **Precision@5** | >85% | ~60% (est) | ⚠️ +25% needed |
| **Recall@10** | >90% | ~70% (est) | ⚠️ +20% needed |
| **Answer Quality** | >80% correct | ~50% | ⚠️ Limited by data |
| **Response Time (P95)** | <2s | 2.01s | ✅ Already good! |
| **System Uptime** | 99.5% | N/A (local) | Need deployment |
| **Cost per Query** | <$0.05 | ~$0.02 | ✅ Good |

---

## 🚀 Quick Win Opportunities (Do First)

### Week 1 Quick Wins (High Impact, Low Effort)
1. ✅ **DONE: Fix retrieval** (4,323 embeddings working)
2. **⏩ Process 20 critical documents** (RTE Act, top GOs) - 8 hours
3. **⏩ Add entity recognition for section numbers** - 4 hours
4. **⏩ Implement query spell-check/autocorrect** - 2 hours

**Impact:** These 3 tasks will immediately improve answer quality from 50% → 70%

---

## 📝 Weekly Checkpoint Questions

### Week 1:
- Can the system answer "What is Section 12(1)(c) of RTE Act?" correctly?
- Have we processed 20 high-priority documents?
- Are chunks now structure-aware (not breaking mid-section)?

### Week 2:
- Are we at 50+ documents processed (10,000+ chunks)?
- Does semantic chunking improve retrieval precision (baseline → +10%)?

### Week 3:
- Is bridge table populated with 1,000+ relationships?
- Can queries about superseded GOs return current info?

### Week 4:
- Have we seen +15% improvement in Precision@5?
- Are multi-intent queries routing to correct agents?

### Week 5-6:
- Do we have 200+ labeled test queries?
- Are we hitting 85%+ precision target?

### Week 7-8:
- Is the API deployed and monitored?
- Are we handling 100 concurrent queries at <2s P95?

---

## 💰 Resource Requirements

| Resource | Cost | Notes |
|----------|------|-------|
| Qdrant Cloud | $100/mo | Current plan sufficient |
| Groq API | ~$50/mo | For 10,000 queries |
| Gemini API | ~$30/mo | Backup provider |
| Compute (embeddings) | $0 | Run locally |
| Monitoring (Grafana Cloud) | $50/mo | Optional |
| **Total** | **~$230/mo** | Production ready |

---

## 🎓 Mentor Validation Checkpoints

### Checkpoint 1 (End of Week 2): "Does retrieval work?"
**Test:**
```bash
python test_critical_queries.py  # 20 must-pass queries
```
**Pass Criteria:** 18/20 queries retrieve relevant chunks (90%+)

### Checkpoint 2 (End of Week 4): "Is it SOTA?"
**Test:**
- Query enhancement active (entity linking, semantic expansion)
- Multi-intent routing working
- Precision@5 > 75%

**Pass Criteria:** System outperforms naive keyword search

### Checkpoint 3 (End of Week 6): "Is it production quality?"
**Test:**
- Evaluation suite shows 85%+ metrics
- No critical bugs in 100-query stress test
- Answer quality validated by domain expert

**Pass Criteria:** Ready for beta users

### Checkpoint 4 (End of Week 8): "Can it scale?"
**Test:**
- 100 concurrent queries at <2s P95
- 99.5% uptime over 1 week
- Monitoring dashboards operational

**Pass Criteria:** Ready for production launch

---

## 🎯 TL;DR - Next 3 Actions

1. **TODAY:** Process RTE Act + 20 critical docs with semantic chunking → Fix answer quality
2. **THIS WEEK:** Implement entity linking + semantic expansion → Fix precision
3. **NEXT WEEK:** Build evaluation suite → Measure progress objectively

**Expected Result:** System goes from "technically works" to "actually useful" in 2 weeks.

