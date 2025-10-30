# Work Delegation Plan: You + Claude Code

**Date:** October 30, 2024  
**Objective:** Parallel development to accelerate completion

---

## 🎯 Division of Labor

### 👨‍💻 **Me (This Session) - System Architecture & Core Logic**
Focus: Complex implementations, system design, integration

### 🤖 **Claude Code (Your Runner) - Data Work & Testing**
Focus: Iterative development, data preparation, testing, refinement

---

## 📋 Current Status

| Stage | Status | Owner | Next Steps |
|-------|--------|-------|------------|
| **1. Query Processing** | ✅ COMPLETE | Me | Testing (Claude Code) |
| **2. Agent Router** | 🔄 IN PROGRESS | Me | Implementation |
| **3. Data Ingestion** | 🔄 READY | Claude Code | Run on verticals |
| **4. Embeddings** | ⏸️ PENDING | Claude Code | After ingestion |
| **5. Knowledge Graph** | ⏸️ PENDING | Me | Design + Claude Code data |
| **6. Retrieval** | ⏸️ PENDING | Me | After embeddings |
| **7. Synthesis** | ⏸️ PENDING | Me | LLM integration |
| **8. Evaluation** | ⏸️ PENDING | Claude Code | Test suite |

---

## 🤖 Tasks for Claude Code (Your Runner)

### Priority 1: Data Foundations & Testing ⚡ START NOW

#### Task 1.1: Test Query Processing Pipeline
**Estimated Time:** 30 minutes

```bash
cd /Users/nitin/.cursor/worktrees/AI_policy_Assistant/MKM79

# Create test script
cat > test_query_processing.py << 'EOF'
from src.query_processing.pipeline import QueryProcessingPipeline
import json

# Test queries from different categories
test_queries = [
    "What is the PTR in Guntur district for 2023-24?",
    "Show Nadu-Nedu enrollment statistics",
    "How does Amma Vodi scheme work?",
    "What does Section 12(1)(c) of RTE Act say?",
    "List all districts in AP",
    "Compare dropout rates between Vijayawada and Visakhapatnam",
    "Show me GO MS No 54 details",
    "Why was Nadu-Nedu extended in 2022?",
]

pipeline = QueryProcessingPipeline()

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)
    
    result = pipeline.process(query)
    
    print(f"Normalized: {result.normalized_query}")
    print(f"Intent: {result.primary_intent} ({result.intent_confidence:.2f})")
    print(f"Entities: {result.entity_summary}")
    print(f"Verticals: {result.suggested_verticals}")
    print(f"Complexity: {result.query_complexity}")
    print(f"Expansions: {len(result.query_expansions)}")
    print(f"Time: {result.processing_time_ms:.2f}ms")

EOF

python test_query_processing.py
```

**Deliverables:**
- Test output report
- Any bugs/issues found
- Performance metrics

---

#### Task 1.2: Expand Dictionaries
**Estimated Time:** 1 hour

**Action:** Enhance dictionary files with more comprehensive data

```bash
# Update acronyms.json
cat > data/dictionaries/acronyms.json << 'EOF'
{
  "RTE": ["Right to Education"],
  "NEP": ["National Education Policy"],
  "UDISE": ["Unified District Information System for Education"],
  "UDISE+": ["Unified District Information System for Education Plus"],
  "PTR": ["Pupil-Teacher Ratio", "Teacher-Pupil Ratio"],
  "GER": ["Gross Enrollment Ratio"],
  "NER": ["Net Enrollment Ratio"],
  "SSA": ["Sarva Shiksha Abhiyan"],
  "RMSA": ["Rashtriya Madhyamik Shiksha Abhiyan"],
  "NCERT": ["National Council of Educational Research and Training"],
  "SCERT": ["State Council of Educational Research and Training"],
  "NCTE": ["National Council for Teacher Education"],
  "NAS": ["National Achievement Survey"],
  "ASER": ["Annual Status of Education Report"],
  "MDM": ["Mid-Day Meal"],
  "GO": ["Government Order"],
  "DSC": ["District Selection Committee"],
  "TRT": ["Teacher Recruitment Test"],
  "NCPCR": ["National Commission for Protection of Child Rights"]
}
EOF

# Update ap_gazetteer.json with all 13 districts
cat > data/dictionaries/ap_gazetteer.json << 'EOF'
{
  "districts": [
    "Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Rajahmundry",
    "Anantapur", "Chittoor", "Kurnool", "Nellore", "Kadapa",
    "Prakasam", "Srikakulam", "Vizianagaram"
  ],
  "schemes": [
    "Nadu-Nedu",
    "Jagananna Amma Vodi",
    "Jagananna Gorumudda",
    "Vidya Deevena",
    "Vasathi Deevena",
    "Mid-Day Meal Scheme",
    "Sarva Shiksha Abhiyan",
    "Rashtriya Madhyamik Shiksha Abhiyan"
  ],
  "metrics": [
    "enrollment",
    "dropout rate",
    "PTR",
    "GER",
    "NER",
    "infrastructure",
    "budget allocation",
    "teacher strength",
    "student strength"
  ]
}
EOF

# Create synonyms dictionary
cat > data/dictionaries/synonyms.json << 'EOF'
{
  "student": ["pupil", "learner", "child"],
  "teacher": ["educator", "instructor", "faculty"],
  "school": ["institution", "educational institution"],
  "enrollment": ["enrolment", "admission", "registration"],
  "dropout": ["attrition", "wastage"],
  "scheme": ["programme", "program", "yojana"],
  "district": ["mandal", "region"]
}
EOF
```

**Deliverables:**
- Updated dictionary files
- Documentation of additions

---

#### Task 1.3: Run Ingestion on Legal Vertical
**Estimated Time:** 2 hours

**Action:** Process Legal vertical documents

```bash
cd /Users/nitin/.cursor/worktrees/AI_policy_Assistant/MKM79

# Run ingestion on Legal vertical
python scripts/process_vertical.py --vertical Legal --output-dir data/processed_verticals

# Check outputs
ls -la data/processed_verticals/Legal/

# Review processing summary
cat data/processed_verticals/Legal/_processing_summary.json
```

**Expected Outputs:**
- 12 processed JSON files (one per Legal document)
- Processing summary with success/failure counts
- Quality metrics

**Deliverables:**
- Processing logs
- Summary statistics
- Any errors encountered
- Sample of extracted sections/entities

---

### Priority 2: Continue Ingestion ⏳ AFTER P1

#### Task 2.1: Process Government_Orders Vertical
```bash
python scripts/process_vertical.py --vertical Government_Orders
```

#### Task 2.2: Process Schemes Vertical
```bash
python scripts/process_vertical.py --vertical Schemes
```

#### Task 2.3: Batch Process Remaining Verticals
```bash
# Process all remaining verticals
python scripts/process_vertical.py --all
```

**Deliverables:**
- All 249 documents processed
- Consolidated processing report
- Identified issues/gaps

---

### Priority 3: Generate Embeddings ⏳ AFTER P2

#### Task 3.1: Generate Embeddings for Legal
```bash
python scripts/generate_embeddings.py \
  --input data/processed_verticals/Legal \
  --collection legal_v1 \
  --model sentence-transformers/all-MiniLM-L6-v2
```

#### Task 3.2: Generate for All Verticals
```bash
# Batch generate embeddings
for vertical in Legal Government_Orders Schemes Judicial Data_Reports; do
  python scripts/generate_embeddings.py \
    --input data/processed_verticals/$vertical \
    --collection ${vertical}_v1
done
```

**Deliverables:**
- Embeddings in Qdrant (or local store)
- Embedding statistics
- Sample similarity tests

---

### Priority 4: Evaluation Suite ⏳ AFTER P3

#### Task 4.1: Create Test Query Set
**Action:** Build comprehensive test queries

```python
# Create data/evaluation/extended_test_queries.json
{
  "legal_queries": [
    "What is Section 12(1)(c) of RTE Act?",
    "Explain AP Education Act 1982 Section 3",
    "What are the amendments to RTE Rules 2010?"
  ],
  "data_queries": [
    "What is the PTR in Guntur for 2023-24?",
    "Show enrollment trends from 2020-2024",
    "Compare dropout rates across districts"
  ],
  "scheme_queries": [
    "How does Nadu-Nedu work?",
    "Who is eligible for Amma Vodi?",
    "What is the budget for Gorumudda scheme?"
  ],
  "judicial_queries": [
    "Show 2024 AP judgments on RTE",
    "What did the court say about Section 12(1)(c)?",
    "List education-related cases"
  ],
  "go_queries": [
    "Show GO MS No 54 details",
    "Which GOs supersede GO MS No 42?",
    "Find Nadu-Nedu related GOs"
  ]
}
```

#### Task 4.2: Run Evaluation Tests
```bash
python scripts/evaluate_system.py \
  --test-queries data/evaluation/extended_test_queries.json \
  --output data/evaluation/results.json
```

**Deliverables:**
- Test query set (50+ queries)
- Evaluation results
- Performance metrics

---

### Priority 5: GO Supersession Mapping ⏳ ONGOING

#### Task 5.1: Extract GO Relationships
**Action:** From Government_Orders PDFs, create supersession map

```bash
# Manual extraction then create CSV
cat > data/knowledge_graph/go_supersession.csv << 'EOF'
go_number,supersedes,date,subject,status
GO_MS_54,GO_MS_42,2019-06-13,Jagananna Amma Vodi,active
GO_MS_129,,2022-07-15,Nadu-Nedu Extension,active
...
EOF
```

**Deliverables:**
- `go_supersession.csv` with relationships
- Documentation of extraction method

---

## 👨‍💻 Tasks for Me (This Session)

### Priority 1: Agent Router & Vertical Strategies ⚡ NOW

#### Stage 2.1: Agent Router Implementation
**File:** `src/agents/router.py`

**Features:**
- Route based on intent + entities
- Multi-agent orchestration
- Confidence-based fallback
- Parallel agent execution

#### Stage 2.2: Vertical-Specific Retrieval
**Files:** `src/agents/{legal_agent, go_agent, judicial_agent, data_agent}.py`

**Features:**
- Legal: Section-based retrieval with cross-references
- GO: Supersession chain traversal
- Judicial: Case law precedent mapping
- Data: Time-series and metric queries

---

### Priority 2: Knowledge Graph Foundation ⏳ AFTER STAGE 2

#### Stage 5.1: KG Schema Design
**File:** `src/knowledge_graph/schema.py`

**Entities:**
- Act, Section, Rule, Amendment
- GO, Supersession
- Case, Judgment, Precedent
- Scheme, Budget, Beneficiary

**Relations:**
- CONTAINS (Act → Section)
- SUPERSEDES (GO → GO)
- CITES (Case → Act/Section)
- IMPLEMENTS (GO → Scheme)

#### Stage 5.2: Bridge Table Builder
**File:** `src/knowledge_graph/bridge_builder.py`

**Features:**
- Automated relationship extraction
- Manual overrides/corrections
- Query API for graph traversal

---

### Priority 3: Retrieval System ⏳ AFTER STAGE 4 (Embeddings)

#### Stage 6.1: Hybrid Retrieval
**File:** `src/retrieval/hybrid_retriever.py`

**Features:**
- Vector search (semantic)
- Keyword search (BM25)
- Knowledge graph traversal
- Score fusion (RRF)

#### Stage 6.2: Reranker
**File:** `src/retrieval/reranker.py`

**Features:**
- Cross-encoder reranking
- Confidence scoring
- Result deduplication

---

### Priority 4: Synthesis & LLM Integration ⏳ AFTER STAGE 6

#### Stage 7.1: Answer Generator
**File:** `src/synthesis/answer_generator.py`

**Features:**
- Claude API integration
- Structured prompts per intent
- Citation extraction
- Multi-vertical synthesis

#### Stage 7.2: Verification
**File:** `src/synthesis/verifier.py`

**Features:**
- Fact checking
- Citation validation
- Confidence scoring
- Hallucination detection

---

## 📊 Progress Tracking

### Week 1 (Current)
- [x] Stage 1: Query Processing ✅
- [ ] Claude Code: Test + dictionaries + Legal ingestion
- [ ] Stage 2: Agent Router (in progress)

### Week 2
- [ ] Claude Code: All verticals ingestion
- [ ] Claude Code: Embeddings generation
- [ ] Stage 5: Knowledge Graph
- [ ] Stage 6: Retrieval

### Week 3
- [ ] Stage 7: Synthesis
- [ ] Stage 8: Evaluation
- [ ] Claude Code: Comprehensive testing
- [ ] Integration & optimization

---

## 🔄 Sync Points

### Daily Sync
**What:** Quick status update
**When:** End of each work session
**Format:**
```
- Completed: [tasks]
- Blocked on: [issues]
- Next: [upcoming tasks]
```

### Weekly Review
**What:** Comprehensive review + planning
**When:** End of week
**Format:**
- Demo of working features
- Review metrics/quality
- Adjust plan if needed

---

## 📦 Deliverables Checklist

### From Claude Code
- [ ] Query processing test report
- [ ] Expanded dictionaries
- [ ] Legal vertical processed (12 files)
- [ ] All verticals processed (249 files)
- [ ] Embeddings generated (all verticals)
- [ ] GO supersession map
- [ ] Test query set (50+)
- [ ] Evaluation results

### From Me
- [x] Query processing pipeline ✅
- [ ] Agent router
- [ ] Vertical-specific retrieval
- [ ] Knowledge graph schema
- [ ] Bridge table builder
- [ ] Hybrid retrieval system
- [ ] Reranker
- [ ] Answer generator
- [ ] Verification system

---

## 🆘 Communication Protocol

### For Claude Code Issues
**If stuck:**
1. Document the issue
2. Share error logs/outputs
3. Provide context (what you tried)
4. Suggest possible solutions

**Report Format:**
```
Issue: [brief description]
Context: [what were you doing]
Error: [error message/logs]
Tried: [attempted solutions]
Need: [what you need to proceed]
```

### For Integration Points
**When you need my output:**
1. Specify exactly what you need
2. Provide format example
3. Explain use case
4. Set priority/deadline

---

## 🎯 Success Criteria

### End of Week 1
- ✅ Stage 1 complete and tested
- ✅ Legal vertical processed
- ✅ Dictionaries expanded
- ⏳ Agent router 50% complete

### End of Week 2
- All verticals ingested
- Embeddings generated
- Knowledge graph operational
- Basic retrieval working

### End of Week 3
- Complete end-to-end system
- Evaluation passing (>80% accuracy)
- Documentation complete
- Ready for demo

---

## 💡 Tips for Success

### For Claude Code
- **Iterative approach:** Test small, iterate fast
- **Document everything:** Keep logs of what you do
- **Quality over speed:** Better to do it right once
- **Ask questions:** If unclear, ask before proceeding

### For Me
- **Modular design:** Keep components loosely coupled
- **Test as you go:** Don't wait until the end
- **Document decisions:** Explain "why" not just "what"
- **Stay focused:** One stage at a time

---

**Let's build this together! 🚀**

