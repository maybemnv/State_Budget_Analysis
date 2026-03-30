# API Guide

Complete reference for the DataLens AI REST API and WebSocket endpoints.

## Base URL

- Development: `http://localhost:8000`
- Production: `https://your-domain.com`

## Authentication

Currently, the API does not require authentication. Rate limiting is applied per IP address.

## Rate Limits

| Endpoint Type | Limit |
|---------------|-------|
| Upload | 10 requests/minute |
| Chat (HTTP) | 60 requests/minute |
| Chat (WebSocket) | 1 connection per session |
| All other | 120 requests/minute |

## Content Types

- File uploads: `multipart/form-data`
- JSON requests: `application/json`
- WebSocket messages: JSON text frames

## Common Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request - invalid parameters |
| 404 | Session not found |
| 413 | File too large |
| 422 | Invalid file type or parse error |
| 429 | Rate limit exceeded |
| 500 | Server error |

## Endpoints

### Upload

#### POST /upload

Upload a dataset file (CSV, Excel, or Parquet).

**Request:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data.csv"
```

**Response:**
```json
{
  "session_id": "sess_abc123xyz",
  "filename": "data.csv",
  "rows": 15000,
  "columns": 12,
  "column_names": ["date", "category", "amount", "department", ...]
}
```

**Error Responses:**
```json
// 413 Payload Too Large
{ "detail": "File exceeds 100MB limit" }

// 422 Unprocessable Entity
{ "detail": "File type '.txt' not supported. Allowed: {'.csv', '.xlsx', '.xls', '.parquet'}" }
```

### Sessions

#### GET /sessions

List all active session IDs.

**Response:**
```json
{
  "count": 5,
  "sessions": ["sess_abc123", "sess_def456", "sess_ghi789"]
}
```

#### GET /sessions/{session_id}

Get dataset metadata for a session.

**Response:**
```json
{
  "session_id": "sess_abc123xyz",
  "filename": "data.csv",
  "shape": [15000, 12],
  "columns": ["date", "category", "amount", "department", ...],
  "dtypes": {
    "date": "datetime64[ns]",
    "category": "object",
    "amount": "float64",
    "department": "object"
  },
  "numeric_columns": ["amount"],
  "categorical_columns": ["category", "department"],
  "missing_values": 23
}
```

#### DELETE /sessions/{session_id}

Delete a session and all associated data (messages, charts, tool runs).

**Response:**
```json
{ "status": "deleted", "session_id": "sess_abc123xyz" }
```

### Chat (HTTP)

#### POST /chat/{session_id}

Send a message to the AI agent via HTTP.

**Request:**
```bash
curl -X POST http://localhost:8000/chat/sess_abc123xyz \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the top 5 spending categories?"}'
```

**Response:**
```json
{
  "answer": "The top 5 spending categories are: IT ($2.5M), HR ($1.8M), Facilities ($1.5M), Marketing ($1.2M), and Operations ($980K).",
  "chart_spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "mark": "bar",
    "encoding": {
      "x": {"field": "category", "type": "nominal"},
      "y": {"field": "amount", "type": "quantitative"}
    }
  },
  "has_error": false,
  "steps": [
    {
      "tool": "describe_dataset",
      "args": {},
      "result": { "columns": [...], "dtypes": {...} }
    },
    {
      "tool": "group_by_stats",
      "args": { "group_column": "category", "agg_column": "amount", "agg_func": "sum" },
      "result": { "IT": 2500000, "HR": 1800000, ... }
    },
    {
      "tool": "generate_chart_spec",
      "args": { "chart_type": "bar", "x_column": "category", "y_column": "amount" },
      "result": { /* vega spec */ }
    }
  ]
}
```

### Chat History

#### GET /chat/{session_id}/messages

Retrieve conversation history with tool runs.

**Response:**
```json
{
  "count": 12,
  "messages": [
    {
      "id": 1,
      "session_id": "sess_abc123xyz",
      "role": "user",
      "content": "What are the top 5 spending categories?",
      "tool_name": null,
      "tool_input": null,
      "tool_result": null,
      "created_at": "2026-03-30T14:30:00"
    },
    {
      "id": 2,
      "session_id": "sess_abc123xyz",
      "role": "assistant",
      "content": "The top 5 spending categories are...",
      "tool_name": "group_by_stats",
      "tool_input": { "group_column": "category", "agg_column": "amount" },
      "tool_result": { "IT": 2500000, ... },
      "created_at": "2026-03-30T14:30:02"
    }
  ]
}
```

#### GET /chat/{session_id}/charts

Get all charts generated in the session.

**Response:**
```json
{
  "count": 3,
  "charts": [
    {
      "id": 1,
      "session_id": "sess_abc123xyz",
      "chart_type": "auto",
      "vega_spec": { /* vega-lite spec */ },
      "query": "What are the top 5 spending categories?",
      "created_at": "2026-03-30T14:30:05"
    }
  ]
}
```

### Chat (WebSocket)

#### WS /ws/{session_id}

Real-time streaming chat endpoint. Connects via WebSocket and streams agent events.

**JavaScript Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/sess_abc123xyz');

ws.onopen = () => {
  ws.send(JSON.stringify({ message: 'Show me a correlation matrix' }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'thought':
      console.log('Agent thinking:', data.content);
      break;
    case 'tool_call':
      console.log('Tool called:', data.tool, data.input);
      break;
    case 'tool_result':
      console.log('Tool result:', data.result);
      break;
    case 'chart':
      renderChart(data.spec);
      break;
    case 'answer':
      console.log('Final answer:', data.content);
      break;
    case 'error':
      console.error('Error:', data.message);
      break;
    case 'done':
      console.log('Stream complete');
      break;
  }
};

ws.onclose = () => {
  console.log('Connection closed');
};
```

**Event Types:**

| Type | Description | Payload |
|------|-------------|---------|
| `thought` | Agent reasoning step | `{ type, content }` |
| `tool_call` | Tool execution started | `{ type, tool, input }` |
| `tool_result` | Tool execution completed | `{ type, tool, result }` |
| `chart` | Chart specification generated | `{ type, spec }` |
| `answer` | Final agent response | `{ type, content }` |
| `error` | Error occurred | `{ type, message }` |
| `done` | Stream finished | `{ type }` |

**Python Example (async):**
```python
import asyncio
import json
import websockets

async def chat():
    uri = "ws://localhost:8000/ws/sess_abc123xyz"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"message": "What trends do you see?"}))
        
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data.get('content', '')}")
            
            if data['type'] == 'done':
                break

asyncio.run(chat())
```

### Health

#### GET /health

Check API health status.

**Response:**
```json
{ "status": "ok", "version": "2.0.0" }
```

#### GET /

API root with endpoint overview.

**Response:**
```json
{
  "name": "DataLens AI API",
  "version": "2.0.0",
  "docs": "/docs",
  "endpoints": {
    "upload": "POST /upload",
    "session_info": "GET /sessions/{session_id}",
    "delete_session": "DELETE /sessions/{session_id}",
    "chat_http": "POST /chat/{session_id}",
    "chat_ws": "WS /ws/{session_id}",
    "health": "GET /health"
  }
}
```

## Error Handling

All errors follow this structure:

```json
{
  "detail": "Human-readable error message",
  "code": "optional_error_code"
}
```

Common error scenarios:

| Scenario | Status | Detail |
|----------|--------|--------|
| Session expired | 404 | "Session not found" |
| Invalid file type | 422 | "File type '.xyz' not supported" |
| File too large | 413 | "File exceeds 100MB limit" |
| Malformed JSON | 400 | "Invalid JSON in request body" |
| Agent error | 500 | "Agent execution failed: ..." |

## Rate Limit Headers

When rate limits are exceeded, responses include:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1711929600
Retry-After: 45
```

## SDK Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload file
with open("data.csv", "rb") as f:
    response = requests.post(f"{BASE_URL}/upload", files={"file": f})
    session_id = response.json()["session_id"]

# Chat
response = requests.post(
    f"{BASE_URL}/chat/{session_id}",
    json={"message": "Show summary statistics"}
)
print(response.json()["answer"])
```

### JavaScript/TypeScript

```typescript
async function uploadFile(file: File): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/upload', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  return data.session_id;
}

async function chat(sessionId: string, message: string) {
  const response = await fetch(`http://localhost:8000/chat/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  
  return response.json();
}
```

## OpenAPI/Swagger

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
