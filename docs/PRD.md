# DataLens AI — Product Requirements Document
### Autonomous Data Analyst Platform · v2.0 · February 2026

| Version | Status | Date | Author |
|---------|--------|------|--------|
| 2.0.0 | Draft | Feb 2026 | You |

---

## Project Summary

DataLens AI is a complete rebuild of the existing CSV Data Analyzer — moving from a Streamlit prototype to a production-grade, agentic data analysis platform. The new system combines a **Next.js 15 frontend** with shadcn/ui components and React Three Fiber 3D visualizations, a **FastAPI Python backend** preserving all existing analysis logic, and a **LangChain-powered Autonomous Data Analyst agent** that interprets natural language commands and orchestrates multi-step analytical workflows in real time.

---

## Table of Contents

1. [Product Overview & Vision](#1-product-overview--vision)
2. [Technology Stack](#2-technology-stack)
3. [System Architecture](#3-system-architecture)
4. [Agentic Workflow Design](#4-agentic-workflow-design)
5. [UI Design & Layout](#5-ui-design--layout)
6. [Algorithm Expansion](#6-algorithm-expansion)
7. [Migration Plan from v1](#7-migration-plan-from-v1)
8. [Open Decisions & Future Scope](#8-open-decisions--future-scope)

---

## 1. Product Overview & Vision

### 1.1 Problem Statement

The existing CSV Data Analyzer (v1) is a functional prototype built on Streamlit. While it delivers real analytical value, it suffers from three core limitations that prevent it from being a serious, production-ready tool:

- The Streamlit UI is rigid and visually dated, making it difficult to compose complex multi-panel layouts or interactive 3D visualizations
- Analysis is purely passive — users must know which tab to visit and manually trigger every operation, creating cognitive overhead
- The Gemini AI integration is shallow — a single-shot query with no reasoning chain, no tool use, and no ability to run follow-up analyses autonomously

### 1.2 Vision

> **DataLens AI transforms passive data exploration into an active conversation.** Users describe what they want to understand, and an autonomous agent decides which analyses to run, executes them in sequence, interprets results, and presents a coherent story — all in real time.

### 1.3 Success Metrics

| Metric | Target |
|--------|--------|
| Time to first insight | < 30 seconds from upload |
| Agent task completion rate | > 85% on benchmark query set |
| Supported file size | Up to 100MB CSV |
| P95 agent response latency | < 8 seconds |
| Visualization render time | < 2 seconds |

---

## 2. Technology Stack

### 2.1 Why This Stack

Every technology choice is justified by the specific requirements of an agentic data platform:

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Frontend UI | Next.js 15 (App Router) | Server components, streaming SSR, native WebSocket support, and best-in-class DX for React apps |
| Component Library | shadcn/ui + Tailwind CSS | Unstyled, fully composable components — no fighting with a design system, full control over the dark theme |
| 3D Visualization | React Three Fiber + Drei | Declarative Three.js in React — enables PCA/UMAP 3D scatter, cluster orbs, and animated data landscapes |
| 2D Charts | Recharts + Visx | Recharts for standard charts; Visx (D3-based) for custom correlation and heatmap renders |
| Agent Framework | LangChain (Python) | Tool-calling agents, structured output, streaming callbacks, and a large ecosystem of integrations |
| LLM | GPT-4o / Gemini 1.5 Pro | Both supported via LangChain abstraction; users bring their own API key. Local Ollama (DeepSeek-R1) as private option |
| Backend API | FastAPI (Python) | Async, fast, WebSocket-native. All existing analysis logic (pandas, sklearn, statsmodels) migrates directly |
| Real-time Comms | WebSockets | Streams agent reasoning steps and tool call results to the UI incrementally as they execute |
| Data Processing | Pandas + Polars | Pandas for compatibility; Polars added for large file (50MB+) processing performance |

### 2.2 LangChain vs CrewAI — Decision Record

**Decision: LangChain ✅**

DataLens needs **one smart agent with many tools**, not multiple collaborating agents with different roles. LangChain's ReAct tool-calling pattern is exactly this — a single reasoning loop that selects and chains tools. CrewAI (which is built on LangChain) adds multi-agent orchestration overhead that provides no benefit here.

CrewAI becomes the right choice if you later want specialized agents — a Data Cleaning Agent, a Visualization Agent, and a Report Writing Agent working in parallel. That is a **v3 conversation**.

---

## 3. System Architecture

### 3.1 High-Level Architecture

The system is divided into three bounded contexts that communicate via a defined API surface:

```
┌─────────────────────┐      WebSocket + REST      ┌──────────────────────────┐      Function Calls      ┌─────────────────────┐
│   FRONTEND LAYER    │ ◄──────────────────────── ► │      AGENT LAYER         │ ◄──────────────────────► │   ANALYSIS LAYER    │
│                     │                             │                          │                          │                     │
│  Next.js 15         │                             │  LangChain ReAct Agent   │                          │  FastAPI Endpoints  │
│  shadcn/ui          │                             │  Tool Registry (12)      │                          │  Existing Analyzers │
│  React Three Fiber  │                             │  Streaming Callbacks     │                          │  New ML Algorithms  │
│  WebSocket Client   │                             │  Dataset Context Mgr     │                          │                     │
└─────────────────────┘                             └──────────────────────────┘                          └─────────────────────┘
```

**Communication flow:**
1. Frontend uploads file via REST → receives `session_id`
2. Opens WebSocket connection
3. User sends natural language query
4. Agent streams reasoning + tool calls over WebSocket
5. Analysis Layer executes each tool
6. Results stream back to UI in real time

### 3.2 Backend Directory Structure

```
backend/
  main.py                    # FastAPI app, WebSocket endpoint
  agent/
    analyst_agent.py         # LangChain ReAct agent definition
    tools/                   # One file per tool
      run_pca.py
      run_clustering.py
      detect_anomalies.py
      run_regression.py
      run_forecast.py
      run_shap.py
      describe_data.py
      generate_chart_spec.py
      ...
    callbacks.py             # Streaming callback → WebSocket
    context.py               # Dataset context per session
  analyzers/                 # Migrated from existing src/analyzers/
    statistical_analyzer.py
    ml_analyzer.py
    time_series/
  api/
    upload.py                # File upload + validation
    sessions.py              # Session management
  utils/
```

### 3.3 Frontend Directory Structure

```
frontend/
  app/
    layout.tsx               # Root layout, theme provider
    page.tsx                 # Landing / upload screen
    workspace/[sessionId]/
      page.tsx               # Main analysis workspace
  components/
    agent/
      AgentChat.tsx          # Chat input + streaming message thread
      ThoughtStep.tsx        # Individual agent reasoning step
      ToolCallCard.tsx       # Tool call with args + result summary
    viz/
      Chart2D.tsx            # Recharts wrapper
      Scene3D.tsx            # React Three Fiber canvas
      PCAScatter3D.tsx       # 3D PCA component
      ClusterOrbs.tsx        # 3D cluster visualization
      HeatmapGrid.tsx        # Correlation heatmap
    data/
      DataTable.tsx          # Virtualized table (TanStack)
      ColumnSummary.tsx      # Per-column stats card
    layout/
      Sidebar.tsx            # File info + session controls
      CanvasPanel.tsx        # Persistent viz canvas
  lib/
    websocket.ts             # WebSocket hook
    types.ts                 # Shared TypeScript types
```

---

## 4. Agentic Workflow Design

### 4.1 Agent Architecture

The Autonomous Data Analyst uses a **LangChain ReAct (Reasoning + Acting)** agent with a streaming callback handler. The agent receives a natural language query and a structured dataset context, then iteratively reasons about which tools to call, executes them, observes results, and continues until it can generate a complete answer.

**The ReAct Loop:**
```
Thought:     What does the user actually need?
Action:      Which tool answers this?
Observation: What did the tool return?
Thought:     Is this enough, or do I need more?
[repeat]
Final Answer: Structured response with all findings and chart specs
```

### 4.2 Agent Tool Registry

The agent has access to 12 tools, each a well-defined Python function with a Pydantic input schema:

| Tool Name | When Agent Uses It | Returns |
|-----------|-------------------|---------|
| `describe_dataset` | First tool called on any query — gets schema, dtypes, null counts, sample rows | Structured dataset summary JSON |
| `get_column_stats` | When user asks about a specific column or metric | Mean, median, std, skew, kurtosis, value counts |
| `detect_anomalies` | "find outliers", "anything weird", "anomalies" | Row indices + scores from Isolation Forest + Z-score |
| `run_correlation` | "what's related", "find relationships", "correlation" | Correlation matrix + top N significant pairs |
| `run_pca` | "reduce dimensions", "visualize clusters", dimensionality queries | Component loadings + explained variance + 2D/3D coords |
| `run_umap` | Clustering, embedding, or when PCA variance < 70% | 2D/3D UMAP embeddings for visualization |
| `run_kmeans` | "cluster", "segment", "group users/customers" | Cluster labels, centroids, silhouette score |
| `run_hdbscan` | When K is unknown or data is noisy | Cluster labels with noise detection |
| `run_regression` | "predict", "forecast", "what affects X" | Coefficients, R², RMSE, feature importance |
| `run_shap` | After regression — "explain why", "feature importance" | SHAP values per feature + summary chart spec |
| `run_forecast` | Time series data, "next quarter", "trend" | ARIMA/Prophet predictions + confidence intervals |
| `generate_chart_spec` | Any time a visualization would help explain findings | Vega-Lite spec JSON consumed by frontend |

### 4.3 Example Agent Traces

#### Query: `"Find anomalies in this dataset"`

```
1. describe_dataset       → learns shape, column types, null counts
2. get_column_stats       → identifies skewed distributions on all numeric cols
3. detect_anomalies       → Isolation Forest returns 23 anomalous rows
4. generate_chart_spec    → scatter plot highlighting anomalous rows in red
5. Final Answer           → narrative explaining anomaly pattern, affected rows, likely causes
```

#### Query: `"Cluster users and explain the patterns"`

```
1. describe_dataset       → identifies this looks like user/customer data
2. run_correlation        → finds which features cluster well together
3. run_pca                → reduces to 10 components explaining 85% variance
4. run_kmeans             → tries K=3,4,5 — K=4 has best silhouette score
5. run_umap               → generates 3D embeddings for visualization
6. generate_chart_spec×2  → 3D cluster viz + cluster profile bar chart
7. Final Answer           → cluster profiles in plain English, business interpretation
```

#### Query: `"Predict next quarter revenue"`

```
1. describe_dataset       → detects date column, confirms time series
2. get_column_stats       → checks revenue for trend, seasonality
3. run_forecast           → ARIMA + Prophet both run, best model selected by AIC
4. run_regression         → identifies leading indicator features
5. run_shap               → explains which features drive the forecast
6. generate_chart_spec    → forecast line chart with confidence intervals
7. Final Answer           → point forecast + range + key assumptions + risk factors
```

### 4.4 Streaming Architecture

The agent streams its reasoning process to the UI via WebSocket in real time. Each message has a typed structure:

```typescript
// WebSocket message types streamed to frontend
{ type: 'thought',     content: 'I need to check for time series patterns...' }
{ type: 'tool_call',   tool: 'run_forecast', args: { column: 'revenue' } }
{ type: 'tool_result', tool: 'run_forecast', summary: 'ARIMA(2,1,2) selected, AIC: 234.1' }
{ type: 'chart_spec',  spec: { ...vega-lite JSON... } }
{ type: 'answer',      content: 'Based on the analysis...' }
{ type: 'error',       message: 'Column not found: ...' }
```

---

## 5. UI Design & Layout

### 5.1 Design Philosophy

- **Dark-first** design with a slate/indigo palette — data tools look credible when they look serious
- **Three-panel layout** that never collapses context — users see their data, chat with the agent, and review visualizations simultaneously
- **3D only where it adds value** — PCA, UMAP, cluster embeddings. Never 3D bar charts.
- **Agent reasoning is visible** — users see every thought step and tool call, building trust and understanding

### 5.2 Main Workspace Layout

```
┌──────────────────┬─────────────────────────────────────┬────────────────────────┐
│   LEFT SIDEBAR   │         CENTER — AGENT CHAT          │   RIGHT — VIZ CANVAS   │
│    280px fixed   │         Flexible width               │     360px fixed         │
│                  │                                      │                         │
│  File info card  │  Message thread (scrollable)         │  Chart gallery (pinned) │
│  Column browser  │  ├─ User message bubbles             │  3D scene (RTF canvas)  │
│  Data type badges│  ├─ Agent thought steps              │  Fullscreen toggle      │
│  Quick stats     │  ├─ Tool call cards (expandable)     │  Export PNG / SVG       │
│  Session history │  ├─ Inline chart renders             │  Chart history tabs     │
│  LLM selector    │  └─ Final answer text                │                         │
│                  │                                      │                         │
│                  │  ─────────────────────────────────── │                         │
│                  │  [ Command input bar — sticky bottom ]│                         │
│                  │  [ Suggested query chips             ]│                         │
└──────────────────┴─────────────────────────────────────┴────────────────────────┘
```

### 5.3 When to Use 3D vs 2D

| Visualization Type | Render | Reason |
|-------------------|--------|--------|
| PCA with 3 components | **3D (RTF)** | Depth represents the third principal component — genuinely more information |
| UMAP embeddings | **3D (RTF)** | Topology becomes clearer with an extra dimension; animated rotation reveals structure |
| K-means cluster orbs | **3D (RTF)** | Cluster separation, size, and density all readable at a glance |
| Correlation heatmap | 2D (Visx) | Matrix data; 3D adds zero value and makes it harder to read |
| Time series forecast | 2D (Recharts) | Time is a 1D axis; 3D would be misleading |
| Distribution / histogram | 2D (Recharts) | Shape of distribution is clearest in 2D |
| Scatter with two variables | 2D (Recharts) | Two variables = 2D; adding Z without a third variable is chart junk |
| SHAP feature importance | 2D horizontal bar | Ranking is a 1D concept; bar chart is canonical |

### 5.4 Component Colour Tokens

```css
--background:     #0F172A   /* slate-900 */
--surface:        #1E293B   /* slate-800 */
--border:         #334155   /* slate-700 */
--accent:         #6366F1   /* indigo-500 */
--accent-soft:    #EEF2FF   /* indigo-50 */
--violet:         #8B5CF6   /* violet-500 */
--green:          #10B981   /* emerald-500 */
--amber:          #F59E0B   /* amber-500 */
--text-primary:   #F1F5F9   /* slate-100 */
--text-muted:     #94A3B8   /* slate-400 */
```

---

## 6. Algorithm Expansion

### 6.1 Migration from v1

All existing algorithms from the Streamlit app migrate directly to the FastAPI backend as tool-callable functions. Nothing is lost — the existing `statistical_analyzer`, `ml_analyzer`, and `time_series` modules are **refactored, not rewritten**.

**Migrated from v1:** Descriptive stats, PCA, K-means, ARIMA, correlation analysis, group-by analysis, histograms, box plots, scatter plots.

### 6.2 New Algorithms Added in v2

| Algorithm | Category | Library | Why Added |
|-----------|----------|---------|-----------|
| Isolation Forest | Anomaly Detection | `scikit-learn` | Handles high-dimensional anomalies better than z-score; works on non-normal data |
| HDBSCAN | Clustering | `hdbscan` | No fixed K required; identifies noise points; better for real-world messy data |
| UMAP | Dim. Reduction | `umap-learn` | Preserves local structure better than PCA; far superior for visualization |
| SHAP Values | Explainability | `shap` | Explains model predictions in plain English; the single most impressive feature to show |
| Prophet | Forecasting | `prophet` | Handles seasonality, holidays, missing data automatically — much easier than raw ARIMA |
| Random Forest | Regression / Classification | `scikit-learn` | Better than linear regression for non-linear relationships; native feature importance |
| Permutation Importance | Explainability | `scikit-learn` | Model-agnostic feature importance that doesn't overvalue high-cardinality features |

---

## 7. Migration Plan from v1

### Phase 1 — Backend Foundation *(Week 1–2)*
- Set up FastAPI project structure with `uv` dependency management
- Migrate all existing analyzer modules from `src/` into `backend/analyzers/`
- Wrap each analyzer function as a LangChain tool with Pydantic schemas
- Implement WebSocket endpoint with streaming callback handler
- Write integration tests for all migrated tools

### Phase 2 — Agent Core *(Week 2–3)*
- Implement ReAct agent with the 12-tool registry
- Build dataset context manager (schema, sample, column metadata per session)
- Test agent traces against benchmark query set (30 representative queries)
- Implement error handling and graceful fallbacks when tools fail

### Phase 3 — Frontend Shell *(Week 3–4)*
- Next.js 15 project setup with shadcn/ui, Tailwind dark mode config
- Three-panel layout implementation
- WebSocket hook and message type system
- File upload with drag-and-drop, validation, progress indicator
- Agent chat thread with thought step + tool call card components

### Phase 4 — Visualizations *(Week 4–5)*
- 2D chart components: histogram, scatter, line, heatmap, bar (Recharts + Visx)
- React Three Fiber canvas setup with camera controls (Drei `OrbitControls`)
- PCA 3D scatter component
- UMAP 3D embedding component with animated entry
- K-means cluster orbs with color-coded segments

### Phase 5 — Polish & Deploy *(Week 5–6)*
- Suggested query chips based on detected data type
- Export functionality (PNG charts, PDF report, CSV results)
- Docker Compose setup: frontend + backend + nginx
- Performance testing on large files (50MB+)
- README + demo GIF

---

## 8. Open Decisions & Future Scope

### 8.1 Decisions to Make Before Building

| Decision | Options & Recommendation |
|----------|--------------------------|
| Python execution for large files | FastAPI server *(recommended)* vs Pyodide in-browser. Pyodide is impressive for demos but limited on 50MB+ files. |
| LLM default | GPT-4o gives best tool-calling reliability. Gemini 1.5 Pro is cheaper. Recommend GPT-4o default, Gemini as option. |
| Local LLM support | Ollama + DeepSeek-R1 via LangChain's Ollama integration. Good for privacy-sensitive data. Add in Phase 5. |
| Authentication | None for v2 (local tool), or simple API key per session. Don't over-engineer. |
| Chart persistence | Store chart specs (JSON) not rendered images — regenerate on load for consistency. |

### 8.2 v3 Scope — Out of Bounds for Now

These are real ideas worth building, but explicitly deferred to avoid scope creep:

- **Multi-agent CrewAI architecture** — separate Cleaning Agent, Analysis Agent, Visualization Agent, Report Writer working in parallel
- **MCP server for IDE integration** — Cursor / VS Code extension so analysts can query data from inside their editor
- **Database connectors** — PostgreSQL, BigQuery, Snowflake instead of file upload only
- **Collaborative sessions** — multiple users analyzing the same dataset simultaneously
- **Scheduled analysis reports** — automated runs with email/Slack delivery

---

*DataLens AI — PRD v2.0 · February 2026 · For internal development use*
