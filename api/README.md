# API Documentation

FastAPI REST API for the AI Policy Assistant system.

## Overview

The API provides programmatic access to the AI Policy Assistant's question-answering capabilities, allowing integration with external systems, web applications, and automated workflows.

## Features

- 🔍 Natural language query processing
- 📚 Citation-based answers from policy documents
- 🎯 Multiple answer modes (normal, detailed, concise, comparative)
- 📊 Comprehensive metadata and metrics
- 🔐 API key authentication
- 📝 Automatic request logging
- ⚡ Rate limiting and caching
- 🚀 High-performance async operations

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY=your_claude_key
export QDRANT_URL=your_qdrant_url
export QDRANT_API_KEY=your_qdrant_key
export API_KEY=your_api_key  # For client authentication
```

### Running the Server

```bash
# Development mode (with auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

```bash
# Build image
docker build -t ai-policy-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
  -e QDRANT_URL=${QDRANT_URL} \
  -e QDRANT_API_KEY=${QDRANT_API_KEY} \
  ai-policy-api
```

## API Endpoints

### Health Check

**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-30T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "qdrant": "connected",
    "claude": "available"
  }
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Query Endpoint

**POST** `/query`

Submit a natural language query and receive a citation-based answer.

**Request Body:**
```json
{
  "query": "What are the eligibility criteria for the Amma Vodi scheme?",
  "mode": "normal_qa",
  "top_k": 10,
  "options": {
    "include_citations": true,
    "include_metadata": true
  }
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | Natural language question |
| mode | string | No | "normal_qa" | Answer mode: `normal_qa`, `detailed`, `concise`, `comparative` |
| top_k | integer | No | 10 | Number of document chunks to retrieve (3-20) |
| options.include_citations | boolean | No | true | Include full citation details |
| options.include_metadata | boolean | No | true | Include retrieval/LLM metadata |

**Response:**
```json
{
  "query": "What are the eligibility criteria for the Amma Vodi scheme?",
  "answer": "The **Amma Vodi** scheme eligibility criteria are:\n\n- **Target Beneficiaries**: Mothers/guardians of children studying in classes 1-12\n- **Annual Income**: Family income must not exceed ₹2 lakh per annum [Source 1]\n- **School Type**: Children must be enrolled in government schools [Source 2]\n- **Attendance**: Minimum 75% attendance required [Source 1]\n\n## Sources\n1. G.O.Ms.No. 123 - Amma Vodi Scheme Guidelines\n2. G.O.Ms.No. 456 - Implementation Instructions",
  "confidence_score": 0.87,
  "mode": "normal_qa",
  "processing_time": 4.23,
  "citations": {
    "total_citations": 3,
    "unique_sources_cited": 2,
    "all_citations_valid": true,
    "citation_density": 0.02,
    "sources_used_percentage": 20.0,
    "citation_details": {
      "1": {
        "index": 1,
        "document": "G.O.Ms.No. 123 - Amma Vodi Scheme Guidelines",
        "section": "Section 3.1",
        "year": "2019",
        "doc_type": "government_order",
        "score": 0.89
      },
      "2": {
        "index": 2,
        "document": "G.O.Ms.No. 456 - Implementation Instructions",
        "section": "Section 2.4",
        "year": "2020",
        "doc_type": "government_order",
        "score": 0.85
      }
    }
  },
  "retrieval_stats": {
    "chunks_retrieved": 10,
    "agents_queried": 2,
    "agents_used": ["SchemeAgent", "GovernmentOrderAgent"],
    "query_complexity": "SIMPLE",
    "retrieval_time": 1.45
  },
  "llm_stats": {
    "model": "claude-sonnet-4-20250514",
    "input_tokens": 7823,
    "output_tokens": 456,
    "total_tokens": 8279,
    "llm_time": 2.78
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "What are teacher qualification requirements?",
    "mode": "detailed",
    "top_k": 15
  }'
```

**Python Client:**
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
```

---

### Document Search

**POST** `/search/documents`

Search for relevant documents without generating an answer.

**Request Body:**
```json
{
  "query": "teacher recruitment",
  "top_k": 5,
  "filters": {
    "doc_type": "government_order",
    "year_min": 2020,
    "year_max": 2024
  }
}
```

**Response:**
```json
{
  "query": "teacher recruitment",
  "total_results": 5,
  "results": [
    {
      "document": "G.O.Ms.No. 123",
      "title": "Teacher Recruitment Guidelines 2023",
      "doc_type": "government_order",
      "year": 2023,
      "section": "Section 4",
      "score": 0.92,
      "chunk_id": "go_123_chunk_45",
      "excerpt": "The recruitment process for government school teachers..."
    }
  ]
}
```

---

### Batch Query

**POST** `/query/batch`

Process multiple queries in a single request.

**Request Body:**
```json
{
  "queries": [
    {"query": "What is RTE Act?", "mode": "concise"},
    {"query": "Teacher qualification requirements", "mode": "normal_qa"}
  ],
  "top_k": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "query": "What is RTE Act?",
      "answer": "...",
      "confidence_score": 0.85,
      ...
    },
    {
      "query": "Teacher qualification requirements",
      "answer": "...",
      "confidence_score": 0.78,
      ...
    }
  ],
  "total_queries": 2,
  "successful": 2,
  "failed": 0,
  "total_processing_time": 8.45
}
```

---

### Usage Statistics

**GET** `/usage/stats`

Get API usage statistics for monitoring and billing.

**Query Parameters:**
- `period`: Time period (`day`, `week`, `month`, `all`)
- `group_by`: Group by field (`mode`, `user`, `none`)

**Response:**
```json
{
  "period": "day",
  "total_queries": 1247,
  "total_tokens": 10456234,
  "estimated_cost_usd": 45.67,
  "by_mode": {
    "normal_qa": {
      "queries": 856,
      "tokens": 7234567,
      "cost_usd": 31.24
    },
    "detailed": {
      "queries": 234,
      "tokens": 2345678,
      "cost_usd": 10.12
    },
    "concise": {
      "queries": 157,
      "tokens": 875989,
      "cost_usd": 4.31
    }
  },
  "average_response_time": 4.23,
  "success_rate": 0.987
}
```

---

## Authentication

All endpoints (except `/health`) require API key authentication.

### Header-based Authentication

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/query
```

### Query Parameter Authentication

```bash
curl "http://localhost:8000/query?api_key=your_api_key"
```

### Obtaining an API Key

Contact your system administrator or configure in `api/middleware/auth.py`.

---

## Error Handling

### Error Response Format

```json
{
  "error": "Invalid query parameter",
  "message": "top_k must be between 3 and 20",
  "code": "INVALID_PARAMETER",
  "timestamp": "2025-10-30T12:00:00Z"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid parameters or malformed request |
| 401 | Unauthorized | Missing or invalid API key |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Dependency unavailable (Qdrant/Claude) |

### Common Errors

**Invalid API Key:**
```json
{
  "error": "Unauthorized",
  "message": "Invalid API key",
  "code": "INVALID_API_KEY"
}
```

**Rate Limit Exceeded:**
```json
{
  "error": "Rate limit exceeded",
  "message": "Maximum 100 requests per minute exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 45
}
```

**Service Unavailable:**
```json
{
  "error": "Service unavailable",
  "message": "Vector database connection failed",
  "code": "SERVICE_UNAVAILABLE"
}
```

---

## Rate Limiting

Default rate limits:

| Tier | Requests/min | Requests/day | Tokens/day |
|------|--------------|--------------|------------|
| Free | 10 | 100 | 1M |
| Basic | 60 | 1000 | 10M |
| Pro | 300 | 10000 | 100M |
| Enterprise | Custom | Custom | Custom |

Rate limit headers in response:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1698765432
```

---

## Request Logging

All requests are automatically logged to `logs/api_requests.jsonl`.

**Log Format:**
```json
{
  "timestamp": "2025-10-30T12:00:00Z",
  "request_id": "req_abc123",
  "endpoint": "/query",
  "method": "POST",
  "query": "What is RTE Act?",
  "mode": "normal_qa",
  "user_id": "user_123",
  "response_time": 4.23,
  "status_code": 200,
  "tokens_used": 8279,
  "cost_usd": 0.0324
}
```

---

## Caching

Results are cached to improve performance and reduce costs.

**Cache Configuration:**
- Cache duration: 1 hour (configurable)
- Cache key: `hash(query + mode + top_k)`
- Cache hit rate: Typically 20-30%

**Cache Headers:**
```
X-Cache: HIT
X-Cache-Age: 345
```

To bypass cache:
```json
{
  "query": "...",
  "options": {
    "bypass_cache": true
  }
}
```

---

## WebSocket Support

Real-time streaming for long-running queries.

**Endpoint:** `ws://localhost:8000/ws/query`

**Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/query?api_key=your_key');

ws.onopen = () => {
  ws.send(JSON.stringify({
    query: "Detailed analysis of RTE Act",
    mode: "detailed"
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'progress') {
    console.log(`Progress: ${data.stage}`);
  } else if (data.type === 'complete') {
    console.log('Answer:', data.answer);
  }
};
```

---

## API Client Libraries

### Python

```python
from api_client import PolicyAssistant

client = PolicyAssistant(
    api_key="your_api_key",
    base_url="http://localhost:8000"
)

# Simple query
response = client.query("What is RTE Act?")
print(response.answer)

# Advanced query
response = client.query(
    query="Teacher requirements",
    mode="detailed",
    top_k=15,
    include_citations=True
)

# Batch query
responses = client.batch_query([
    "What is RTE Act?",
    "Teacher qualification requirements"
])
```

### JavaScript/TypeScript

```typescript
import { PolicyAssistant } from '@ai-policy/client';

const client = new PolicyAssistant({
  apiKey: 'your_api_key',
  baseURL: 'http://localhost:8000'
});

// Simple query
const response = await client.query('What is RTE Act?');
console.log(response.answer);

// Advanced query
const response = await client.query({
  query: 'Teacher requirements',
  mode: 'detailed',
  topK: 15
});
```

### cURL

```bash
#!/bin/bash
API_KEY="your_api_key"
BASE_URL="http://localhost:8000"

query_api() {
  local query="$1"
  local mode="${2:-normal_qa}"
  
  curl -X POST "${BASE_URL}/query" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\", \"mode\": \"$mode\"}" \
    | jq '.answer'
}

# Usage
query_api "What is RTE Act?" "concise"
```

---

## Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=your_claude_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key

# Optional
API_KEY=your_api_key              # For authentication
PORT=8000                         # Server port
LOG_LEVEL=INFO                    # Logging level
ENABLE_CACHE=true                 # Enable response caching
CACHE_TTL=3600                    # Cache TTL in seconds
RATE_LIMIT_PER_MIN=60            # Rate limit per minute
MAX_WORKERS=4                     # Uvicorn workers
```

### Configuration File

`api/config.yaml`:
```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4

api:
  rate_limit_per_min: 60
  max_query_length: 1000
  default_top_k: 10
  allowed_modes:
    - normal_qa
    - detailed
    - concise
    - comparative

cache:
  enabled: true
  ttl: 3600
  max_size: 1000

logging:
  level: INFO
  format: json
  output: logs/api.log
```

---

## Performance Optimization

### Tips for Faster Responses

1. **Use concise mode** for quick answers:
   ```json
   {"query": "...", "mode": "concise"}
   ```

2. **Reduce top_k** for fewer sources:
   ```json
   {"query": "...", "top_k": 5}
   ```

3. **Enable caching** for repeated queries

4. **Use batch endpoint** for multiple queries

5. **WebSocket streaming** for real-time feedback

### Typical Response Times

| Configuration | Response Time |
|---------------|---------------|
| Concise, top_k=5 | 2-3s |
| Normal, top_k=10 | 3-5s |
| Detailed, top_k=15 | 5-8s |
| Comparative, top_k=20 | 7-10s |

---

## Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

### Metrics Endpoint

**GET** `/metrics`

Prometheus-compatible metrics:
```
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/query",status="200"} 1247

# HELP api_response_time_seconds Response time in seconds
# TYPE api_response_time_seconds histogram
api_response_time_seconds_bucket{le="1.0"} 123
api_response_time_seconds_bucket{le="5.0"} 1089
```

---

## Testing

### Unit Tests

```bash
pytest tests/test_api.py -v
```

### Integration Tests

```bash
pytest tests/integration/test_api_integration.py -v
```

### Load Testing

```bash
# Using locust
locust -f tests/load_test.py --host http://localhost:8000
```

---

## Deployment

### Production Checklist

- [ ] Set strong API keys
- [ ] Configure rate limiting
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring and alerting
- [ ] Configure log rotation
- [ ] Enable caching
- [ ] Set up backup Qdrant instances
- [ ] Configure CORS policies
- [ ] Set resource limits
- [ ] Enable request validation

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    image: ai-policy-api:latest
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_API_KEY=${QDRANT_API_KEY}
    depends_on:
      - qdrant
  
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
```

---

## Troubleshooting

### Issue: 401 Unauthorized

**Solution**: Check API key configuration:
```bash
curl -H "X-API-Key: your_correct_key" http://localhost:8000/health
```

### Issue: 503 Service Unavailable

**Cause**: Qdrant or Claude unavailable

**Solution**: Check service connectivity:
```bash
curl $QDRANT_URL/health
# Verify ANTHROPIC_API_KEY is valid
```

### Issue: Slow responses

**Solutions**:
1. Enable caching
2. Reduce top_k
3. Use concise mode
4. Add more Uvicorn workers

### Issue: High memory usage

**Solutions**:
1. Reduce cache size
2. Lower max_workers
3. Limit concurrent requests
4. Set query timeout

---

## Support

- **Documentation**: [Full docs](../docs/)
- **Issues**: GitHub Issues
- **Email**: support@example.com

---

## Changelog

### v1.0.0 (2025-10-30)
- Initial release
- Query endpoint with multiple modes
- Citation validation
- Usage tracking
- Rate limiting
- Caching support

---

## License

Part of the AI Policy Assistant project.

