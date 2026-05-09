# Architecture

System design and data flow documentation for DataLens.

## High-Level Architecture

```mermaid
graph TD
    UI[Frontend Next.js<br/>- React UI<br/>- Vega-Lite<br/>- WebSocket]
    API[Backend FastAPI<br/>- ReAct Agent<br/>- Tool Registry<br/>- File Parser]
    EXT[External Services<br/>- Google Gemini<br/>- PostgreSQL<br/>- Redis]
    User[User Browser<br/>- Upload files<br/>- View charts<br/>- Chat UI]
    DB[Database<br/>- Messages<br/>- Sessions<br/>- Charts<br/>- Tool Runs]
    
    UI <--> API
    API <--> EXT
    User --> UI
    API --> DB
```

## Component Diagram

```mermaid
graph TD
    subgraph DataLens AI
        UI[Frontend<br/>next.config<br/>src/app/<br/>src/components<br/>public/]
        API[Backend<br/>main.py<br/>routes/<br/>agent/<br/>tools/<br/>analyzers/]
        DB[Database<br/>PostgreSQL<br/>Redis]
        Docker[Docker]
        
        UI --- Docker
        API --- Docker
        DB --- Docker
    end
```

## Data Flow

### 1. File Upload Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    User->>Frontend: Upload File
    Frontend->>Backend: POST /upload
    Note over Backend: Parse File (CSV/XLSX/PQ)
    Note over Backend: Clean Data (strip, dropna)
    Note over Backend: Store in DB (Session record)
    Backend-->>User: Return session_id
```

### 2. Chat Flow (WebSocket)

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant WS as WS /ws/{session_id}
    User->>Frontend: Send message
    Frontend->>WS: Send over WebSocket
    Note over WS: Load Session (from DB + Redis cache)
    Note over WS: Get Conversation History (last 10 messages)
    Note over WS: Run ReAct Agent<br/>- LLM (Gemini)<br/>- Tool Selection<br/>- Tool Execution<br/>- Stream Events
    Note over WS: Save Results<br/>- Message (user + assistant)<br/>- ToolRun (metrics)<br/>- Chart (if generated)
    WS-->>User: Streaming events
```

### 3. Agent Decision Flow

```mermaid
graph TD
    Query[User Query] --> Agent[ReAct Agent]
    Agent --> Thought[Thought: What does user want?]
    Agent --> Action[Action: Select Tool]
    Action --> T1[describe_dataset]
    Action --> T2[descriptive_stats]
    Action --> T3[group_by_stats]
    Action --> T4[generate_chart_spec]
    Action --> T5[... 15 tools total]
    Agent --> Obs[Observation: Tool Result]
    Agent --> Ans[Answer: Final Response<br/>with optional chart_spec]
```

## Backend Architecture

### Layer Structure

```mermaid
graph TD
    subgraph API Layer FastAPI
        A1[POST /upload]
        A2[WS/HTTP /chat]
        A3[GET /sessions]
    end
    subgraph Agent Layer LangChain
        AG[ReAct Agent Gemini LLM<br/>Prompt Template --> Tool Call Logic]
    end
    subgraph Tool Layer
        T1[Dataset Tools]
        T2[Statistical Tools]
        T3[ML Tools]
        T4[Time Series]
        T5[Chart Tools]
    end
    subgraph Analyzer Layer
        AN1[statistical.py]
        AN2[ml.py]
        AN3[time_series/]
    end
    subgraph Data Layer
        D1[PostgreSQL Primary]
        D2[Redis Cache]
    end
    
    A1 --> AG
    A2 --> AG
    A3 --> AG
    AG --> T1
    AG --> T2
    AG --> T3
    AG --> T4
    AG --> T5
    T2 --> AN1
    T3 --> AN2
    T4 --> AN3
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

```mermaid
graph TD
    U[User Upload] --> Create[Create DB Session]
    Create --> Cache[Cache in Redis]
    RedisTTL[Redis TTL] -.-> Cache
    Cache --> CR[Chat Request]
    CR --> Check[Check Redis Cache]
    Check -->|Miss| DB[Load from DB]
    Check --> Refresh[Refresh TTL]
    DB --> Refresh
```

## Frontend Architecture

### Component Hierarchy

```mermaid
graph TD
    App[App Next.js] --> Layout
    App --> Pages
    App --> Shared[Shared Components]
    Layout --> Header
    Layout --> Sidebar
    Pages --> Home[/]
    Pages --> Chat[/chat/sessionId]
    Pages --> Sessions[/sessions]
    Home --> UploadZone --> FileDrop
    Chat --> ChatContainer
    Chat --> DatasetPreview
    ChatContainer --> MessageList
    ChatContainer --> ChatInput
    MessageList --> UserMessage
    MessageList --> AssistantMessage
    AssistantMessage --> TextContent
    AssistantMessage --> ChartRenderer --> VegaLiteChart
    AssistantMessage --> ToolSteps
    Sessions --> SessionList
    Shared --> Button
    Shared --> Card
    Shared --> Loading
    Shared --> ErrorBoundary
```

### State Management

```mermaid
classDiagram
    class ReactContextGlobalState {
        <<Global State>>
    }
    class SessionContext {
        +currentSessionId
        +datasetInfo
        +messages[]
    }
    class WebSocketContext {
        +connectionStatus
        +isStreaming
        +pendingMessage
    }
    class ChartContext {
        +generatedCharts[]
    }
    ReactContextGlobalState <|-- SessionContext
    ReactContextGlobalState <|-- WebSocketContext
    ReactContextGlobalState <|-- ChartContext
```

## Tool Registry

15 analytical tools organized by category:

```mermaid
graph LR
    tools --> dataset_tools.py
    tools --> statistical_tools.py
    tools --> ml_tools.py
    tools --> time_series_tools.py
    
    dataset_tools.py --> describe_dataset
    dataset_tools.py --> generate_chart_spec
    
    statistical_tools.py --> descriptive_stats
    statistical_tools.py --> group_by_stats
    statistical_tools.py --> correlation_matrix
    statistical_tools.py --> value_counts
    statistical_tools.py --> outliers_summary
    
    ml_tools.py --> run_pca
    ml_tools.py --> run_kmeans
    ml_tools.py --> detect_anomalies
    ml_tools.py --> run_regression
    ml_tools.py --> run_classification
    
    time_series_tools.py --> check_stationarity
    time_series_tools.py --> run_forecast
    time_series_tools.py --> decompose_time_series
```

## Communication Patterns

### WebSocket Event Flow

```mermaid
sequenceDiagram
    participant Client
    participant Server
    Client->>Server: 1. Connect WS /ws/{id}
    Server-->>Client: 2. Accept Connection
    Client->>Server: 3. Send Message {"message": "..."}
    Server-->>Client: 4. Stream Events {"type": "thought"}
    Server-->>Client: {"type": "tool_call"}
    Server-->>Client: {"type": "tool_result"}
    Server-->>Client: {"type": "chart"}
    Server-->>Client: {"type": "answer"}
    Server-->>Client: {"type": "done"}
    Client<->>Server: 5. Close / Disconnect
```

## Scaling Considerations

### Horizontal Scaling

```mermaid
graph TD
    LB[Load Balancer<br/>Nginx/ALB] --> B1[Backend 1 API]
    LB --> B2[Backend 2 API]
    LB --> B3[Backend 3 API]
    B1 --> DB[PostgreSQL Primary]
    B2 --> DB
    B3 --> DB
    DB --> Redis[Redis Cluster]
```

### Session Affinity

WebSocket connections require sticky sessions:
- Nginx: `ip_hash` or cookie-based
- ALB: Enable stickiness
- Cloud Run: Built-in session affinity

## Security Architecture

```mermaid
graph TD
    Internet --> WAF[WAF<br/>Cloudflare/AWS WAF]
    WAF -.->|Rate limiting, DDoS protection| WAF
    WAF --> HTTPS[HTTPS Nginx]
    HTTPS -.->|TLS 1.3 termination| HTTPS
    HTTPS --> CORS[CORS Headers]
    CORS -.->|Origin validation| CORS
    CORS --> Backend[Backend FastAPI]
    Backend -.->|Input validation, SQL injection protection| Backend
    Backend --> DB[Database PostgreSQL]
    DB -.->|Encrypted at rest| DB
```
