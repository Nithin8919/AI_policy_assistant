# 🚀 Quick Start - Real Retrieval Integration

## ⏱️ 5-Minute Setup

### Step 1: Verify Environment (1 min)

```bash
# Check if .env exists and has required variables
cat .env | grep -E "QDRANT_URL|QDRANT_API_KEY|GROQ_API_KEY|GOOGLE_API_KEY"
```

**Required variables**:
```bash
QDRANT_URL=https://your-qdrant-instance.io:6333
QDRANT_API_KEY=your_qdrant_key
GROQ_API_KEY=your_groq_key           # OR
GOOGLE_API_KEY=your_google_key       # (at least one)
```

### Step 2: Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

### Step 3: Run Verification (3 min)

```bash
# Option A: Use the script
./verify_integration.sh

# Option B: Run directly
python test_real_retrieval.py
```

### Step 4: Check Results

✅ **SUCCESS** looks like:
```
================================================================================
                              TEST SUMMARY
================================================================================

GROQ:
  Passed: 5/5
  Failed: 0/5
  ✅ Success Rate: 100% ✅

================================================================================
                              VERDICT
================================================================================

✅ 🎉 ALL TESTS PASSED! System is using REAL retrieval from Qdrant!
```

❌ **FAILURE** looks like:
```
❌ Processing time: 0.37s (TOO FAST - might be mock data)
❌ No chunks retrieved! Using mock data?
❌ Source appears to be MOCK DATA!
```

---

## 🧪 Test Individual Query

```python
# Quick test script
from src.agents.enhanced_router import EnhancedRouter
from src.query_processing.qa_pipeline_multi_llm import QAPipeline
import os

# Initialize
router = EnhancedRouter(
    qdrant_url=os.getenv('QDRANT_URL'),
    qdrant_api_key=os.getenv('QDRANT_API_KEY')
)

pipeline = QAPipeline(router=router, llm_provider='groq')

# Query
response = pipeline.answer_query("What is Section 12(1)(c) of RTE Act?")

# Verify
print(f"Time: {response.processing_time:.2f}s")  # Should be 2-5s
print(f"Chunks: {response.retrieval_stats['chunks_retrieved']}")  # Should be > 0
print(f"Confidence: {response.confidence_score:.0%}")  # Should be > 70%
print(f"\nAnswer: {response.answer[:200]}...")

# Check citations
for cite_num, source in response.citations['citation_details'].items():
    print(f"\n[{cite_num}] {source['document']}")
    assert 'Mock' not in source['document'], "Still using mock data!"
```

---

## 🔍 Red Flags to Watch

| Symptom | Means | Fix |
|---------|-------|-----|
| Response < 1.5s | Not calling Qdrant | Check router initialization |
| 0 chunks | No retrieval | Verify Qdrant connection |
| "Mock" in citations | Using fake data | Remove mock imports |
| Confidence < 30% | Poor retrieval | Check Qdrant has data |
| No agent names | Router not working | Verify EnhancedRouter import |

---

## 🎯 Integration Checklist

Before proceeding, verify:

- [ ] `.env` file exists with all required variables
- [ ] `api/routes/query.py` imports `EnhancedRouter` (not Mock)
- [ ] Qdrant instance is accessible
- [ ] Qdrant collections have data
- [ ] At least one LLM API key is set
- [ ] All dependencies installed

After running tests, verify:

- [ ] All tests pass (80%+ success rate)
- [ ] Response times are 2-5 seconds
- [ ] Citations show real document names
- [ ] Confidence scores > 70%
- [ ] Agent names appear in retrieval_stats

---

## 🚀 Start the API

Once tests pass:

```bash
# Start server
uvicorn api.main:app --reload --port 8000

# In another terminal, test:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the RTE Act?",
    "mode": "normal_qa"
  }'
```

Expected response includes:
- `"chunks_retrieved": 10` or more
- Real document names in `sources`
- `"confidence": 0.87` or higher
- Processing time in logs: 2-5 seconds

---

## 📚 Documentation

- **Complete Guide**: `REAL_RETRIEVAL_INTEGRATION.md`
- **Summary**: `INTEGRATION_SUMMARY.md`
- **Test Script**: `test_real_retrieval.py`
- **API Docs**: http://localhost:8000/docs (after starting server)

---

## 💡 Pro Tips

1. **Use Groq for Speed**: Default is now `groq` (faster than Gemini)
2. **Check Logs**: Look for "EnhancedRouter" and agent names
3. **Monitor Times**: Real retrieval = 2-5s, Mock = <1s
4. **Validate Citations**: Should have GO numbers, section numbers, years
5. **Test Qdrant First**: Ensure collections exist before running full tests

---

## 🆘 Quick Troubleshooting

### "Missing Qdrant credentials"
```bash
echo "QDRANT_URL=your_url" >> .env
echo "QDRANT_API_KEY=your_key" >> .env
```

### "Connection failed"
```bash
curl $QDRANT_URL/health
# Should return: {"title":"qdrant","version":"..."}
```

### "No chunks retrieved"
```bash
# Check if data exists:
python -c "
from qdrant_client import QdrantClient
import os
client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
collections = client.get_collections()
for c in collections.collections:
    count = client.count(c.name)
    print(f'{c.name}: {count.count} points')
"
```

### Still getting mock data
```bash
# Verify code changes:
grep "EnhancedRouter" api/routes/query.py
grep "mock" api/routes/query.py  # Should be empty

# Force refresh:
rm -rf api/__pycache__ api/routes/__pycache__
python test_real_retrieval.py
```

---

<div align="center">
  <h3>🎉 You're Ready!</h3>
  <p>Run <code>./verify_integration.sh</code> to begin</p>
</div>

