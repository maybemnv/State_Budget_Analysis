# Architecture

System design and data flow documentation for DataLens AI.

## High-Level Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │     │    Backend      │     │   External      │
│   (Next.js)     │◄───►│   (FastAPI)     │◄───►│   Services      │
│                 │     │                 │     │                 │
│ - React UI      │     │ - ReAct Agent   │     │ - Google Gemini │
│ - Vega-Lite     │     │ - Tool Registry │     │ - PostgreSQL    │
│ - WebSocket     │     │ - File Parser   │     │ - Redis         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  User Browser   │     │   Database      │
│                 │     │                 │
│ - Upload files  │     │ - Messages      │
│ - View charts   │     │ - Sessions      │
│ - Chat UI       │     │ - Charts        │
└─────────────────┘     │ - Tool Runs     │
                        └─────────────────┘
```

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DataLens AI                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Frontend   │  │   Backend    │  │  Database    │          │
│  │              │  │              │  │              │          │
│  │ next.config  │  │ main.py      │  │ PostgreSQL   │          │
│  │ src/app/     │  │ routes/      │  │ Redis        │          │
│  │ src/components│  │ agent/       │  │              │          │
│  │ public/      │  │ tools/       │  │              │          │
│  │              │  │ analyzers/   │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                │                │                     │
│         └────────────────┴────────────────┘                     │
│                          │                                      │
│                    ┌─────┴─────┐                                │
│                    │  Docker   │                                │
│                    └───────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. File Upload Flow

```
User ──► Frontend ──► POST /upload ──► Backend
                                        │
                                        ▼
                                   Parse File
                                   (CSV/XLSX/PQ)
                                        │
                                        ▼
                                   Clean Data
                                   (strip, dropna)
                                        │
                                        ▼
                                   Store in DB
                                   (Session record)
                                        │
                                        ▼
                                   Return session_id
                                        │
User ◄──────────────────────────────────┘
```

### 2. Chat Flow (WebSocket)

```
User ──► Frontend ──► WS /ws/{session_id}
                          │
                          ▼
                    Load Session
                    (from DB + Redis cache)
                          │
                          ▼
                    Get Conversation History
                    (last 10 messages)
                          │
                          ▼
                    Run ReAct Agent
                    ├─► LLM (Gemini)
                    ├─► Tool Selection
                    ├─► Tool Execution
                    └─► Stream Events
                          │
                          ▼
                    Save Results
                    ├─► Message (user + assistant)
                    ├─► ToolRun (metrics)
                    └─► Chart (if generated)
                          │
User ◄────────────────────┘
     (streaming events)
```

### 3. Agent Decision Flow

```
User Query
    │
    ▼
┌─────────────┐
│   ReAct     │
│   Agent     │
└─────────────┘
    │
    ├───► Thought: What does user want?
    │
    ├───► Action: Select Tool
    │     ├─► describe_dataset
    │     ├─► descriptive_stats
    │     ├─► group_by_stats
    │     ├─► generate_chart_spec
    │     └─► ... (15 tools total)
    │
    ├───► Observation: Tool Result
    │
    └───► Answer: Final Response
          (with optional chart_spec)
```

## Backend Architecture

### Layer Structure

```
┌─────────────────────────────────────────┐
│           API Layer (FastAPI)           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ /upload │ │ /chat   │ │/sessions│ │
│  │   POST  │ │ WS/HTTP │ │   GET   │ │
│  └─────────┘ └─────────┘ └─────────┘ │
├─────────────────────────────────────────┤
│         Agent Layer (LangChain)         │
│  ┌─────────────────────────────────┐    │
│  │     ReAct Agent (Gemini LLM)    │    │
│  │  ┌─────────┐    ┌───────────┐  │    │
│  │  │  Prompt │───►│ Tool Call │  │    │
│  │  │ Template│    │   Logic   │  │    │
│  │  └─────────┘    └───────────┘  │    │
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│           Tool Layer                    │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Dataset  │ │Statistical│ │   ML   │ │
│  │  Tools   │ │  Tools   │ │ Tools  │ │
│  └──────────┘ └──────────┘ └────────┘ │
│  ┌──────────┐ ┌──────────┐           │
│  │  Time    │ │  Chart   │           │
│  │  Series  │ │  Tools   │           │
│  └──────────┘ └──────────┘           │
├─────────────────────────────────────────┤
│         Analyzer Layer                  │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │statistical│ │    ml    │ │time_series│
│  │   .py    │ │   .py    │ │  /      │ │
│  └──────────┘ └──────────┘ └────────┘ │
├─────────────────────────────────────────┤
│           Data Layer                    │
│  ┌─────────────┐  ┌─────────────┐     │
│  │ PostgreSQL  │  │    Redis    │     │
│  │ (Primary)   │  │   (Cache)   │     │
│  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────┘
```

### Database Schema

```sql
-- Sessions: Uploaded datasets
CREATE TABLE sessions (
    id VARCHAR PRIMARY KEY,
    filename VARCHAR NOT NULL,
    schema JSONB NOT NULL,  -- columns, dtypes, shape
    data JSONB,             -- serialized dataframe (optional)
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

-- Messages: Chat history
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR REFERENCES sessions(id),
    role VARCHAR(20) CHECK (role IN ('user', 'assistant')),
    content TEXT,
    tool_name VARCHAR,
    tool_input JSONB,
    tool_result JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tool Runs: Execution metrics
CREATE TABLE tool_runs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR REFERENCES sessions(id),
    tool_name VARCHAR NOT NULL,
    input_json JSONB,
    result_json JSONB,
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Charts: Generated visualizations
CREATE TABLE charts (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR REFERENCES sessions(id),
    chart_type VARCHAR,
    vega_spec JSONB NOT NULL,
    query TEXT,  -- The question that generated this chart
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Session State Management

```
User Upload
    │
    ▼
┌──────────────┐
│  Create DB   │
│   Session    │
└──────────────┘
    │
    ▼
┌──────────────┐     ┌──────────────┐
│   Cache in   │◄────┤   Redis      │
│    Redis     │     │   (TTL)      │
└──────────────┘     └──────────────┘
    │
    │ Chat Request
    ▼
┌──────────────┐
│  Check Redis │──── Miss? ───► Load from DB
│   Cache      │
└──────────────┘
    │
    ▼
┌──────────────┐
│   Refresh    │
│    TTL       │
└──────────────┘
```

## Frontend Architecture

### Component Hierarchy

```
App (Next.js)
├── Layout
│   ├── Header
│   └── Sidebar
│
├── Pages
│   ├── / (Home)
│   │   └── UploadZone
│   │       └── FileDrop
│   │
│   ├── /chat/[sessionId]
│   │   ├── ChatContainer
│   │   │   ├── MessageList
│   │   │   │   ├── UserMessage
│   │   │   │   └── AssistantMessage
│   │   │   │       ├── TextContent
│   │   │   │       ├── ChartRenderer
│   │   │   │       │   └── VegaLiteChart
│   │   │   │       └── ToolSteps
│   │   │   └── ChatInput
│   │   └── DatasetPreview
│   │
│   └── /sessions
│       └── SessionList
│
└── Shared Components
    ├── Button
    ├── Card
    ├── Loading
    └── ErrorBoundary
```

### State Management

```
┌─────────────────────────────────────────┐
│           React Context                 │
│         (Global State)                  │
├─────────────────────────────────────────┤
│                                         │
│  SessionContext                         │
│  ├── currentSessionId                   │
│  ├── datasetInfo                        │
│  └── messages[]                         │
│                                         │
│  WebSocketContext                       │
│  ├── connectionStatus                   │
│  ├── isStreaming                        │
│  └── pendingMessage                     │
│                                         │
│  ChartContext                           │
│  └── generatedCharts[]                  │
│                                         │
└─────────────────────────────────────────┘
```

## Tool Registry

15 analytical tools organized by category:

```
tools/
├── dataset_tools.py
│   ├── describe_dataset      # Schema + summary
│   └── generate_chart_spec   # Vega-Lite generation
│
├── statistical_tools.py
│   ├── descriptive_stats     # Mean, std, skew, kurtosis
│   ├── group_by_stats       # Aggregation by category
│   ├── correlation_matrix    # Pearson correlations
│   ├── value_counts          # Frequency tables
│   └── outliers_summary      # IQR/Z-score detection
│
├── ml_tools.py
│   ├── run_pca              # Dimensionality reduction
│   ├── run_kmeans           # Clustering
│   ├── detect_anomalies     # Isolation Forest
│   ├── run_regression       # Random Forest regressor
│   └── run_classification   # Random Forest classifier
│
└── time_series_tools.py
    ├── check_stationarity   # ADF/KPSS tests
    ├── run_forecast        # ARIMA/Prophet
    └── decompose_time_series  # Trend/seasonal/residual
```

## Communication Patterns

### WebSocket Event Flow

```
┌──────────┐                    ┌──────────┐
│  Client  │                    │  Server  │
└────┬─────┘                    └────┬─────┘
     │                                │
     │  1. Connect WS /ws/{id}       │
     │ ──────────────────────────────►│
     │                                │
     │  2. Accept Connection          │
     │ ◄──────────────────────────────│
     │                                │
     │  3. Send Message               │
     │ { "message": "..." }           │
     │ ──────────────────────────────►│
     │                                │
     │  4. Stream Events              │
     │ ◄──── { "type": "thought" }    │
     │ ◄──── { "type": "tool_call" }  │
     │ ◄──── { "type": "tool_result"} │
     │ ◄──── { "type": "chart" }      │
     │ ◄──── { "type": "answer" }     │
     │ ◄──── { "type": "done" }       │
     │                                │
     │  5. Close / Disconnect         │
     │ ◄─────────────────────────────►│
```

## Scaling Considerations

### Horizontal Scaling

```
                    ┌─────────────┐
                    │  Load Balancer
                    │   (Nginx/ALB) │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
      ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
      │Backend 1│     │Backend 2│     │Backend 3│
      │  (API)  │     │  (API)  │     │  (API)  │
      └────┬────┘     └────┬────┘     └────┬────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                    ┌──────┴──────┐
                    │ PostgreSQL  │
                    │   Primary   │
                    └─────────────┘
                           │
                    ┌──────┴──────┐
                    │    Redis    │
                    │   Cluster   │
                    └─────────────┘
```

### Session Affinity

WebSocket connections require sticky sessions:
- Nginx: `ip_hash` or cookie-based
- ALB: Enable stickiness
- Cloud Run: Built-in session affinity

## Security Architecture

```
Internet
    │
    ▼
┌─────────────┐
│   WAF       │  ──► Rate limiting, DDoS protection
│  (Cloudflare│
│  /AWS WAF)  │
└─────────────┘
    │
    ▼
┌─────────────┐
│   HTTPS     │  ──► TLS 1.3 termination
│  (Nginx)    │
└─────────────┘
    │
    ▼
┌─────────────┐
│   CORS      │  ──► Origin validation
│  Headers    │
└─────────────┘
    │
    ▼
┌─────────────┐
│  Backend    │  ──► Input validation, SQL injection protection
│  (FastAPI)  │
└─────────────┘
    │
    ▼
┌─────────────┐
│  Database   │  ──► Encrypted at rest
│ (PostgreSQL)│
└─────────────┘
```
