# DataLens AI v2.0 - Implementation Tasks

Based on [PRD v2.0](docs/PRD.md).

## Phase 1: Backend Foundation (Week 1–2)
**Goal:** robust FastAPI server with migrated logic and WebSocket support.

- [x] **Project Setup**
  - [x] Initialize FastAPI project structure (`backend/`)
  - [x] Configure `uv` for dependency management
  - [x] Set up `pyproject.toml` with dependencies (FastAPI, LangChain, Pandas, Scikit-learn, etc.)

- [x] **Migration**
  - [x] Migrate `src/analyzers/statistical_analyzer.py` to `backend/analyzers/`
  - [x] Migrate `src/analyzers/ml_analyzer.py` to `backend/analyzers/`
  - [x] Migrate `src/time_series/` modules to `backend/analyzers/time_series/`
  - [x] Refactor migrated code to be pure functions (remove any Streamlit dependencies)

- [x] **Tool Wrapping**
  - [x] Create Pydantic input schemas for all analyzer functions
  - [x] Wrap functions as LangChain tools (approx. 12 tools)
  - [x] Implement `describe_dataset` tool
  - [x] Implement `generate_chart_spec` tool

- [x] **API Core**
  - [x] Implement WebSocket endpoint (`/ws/{client_id}`)
  - [x] Create streaming callback handler for LangChain to WebSocket
  - [x] Implement file upload endpoint (`/upload`) with validation
  - [x] Create session management for file/dataframe context

- [x] **Testing**
  - [x] Write integration tests for all migrated tools
  - [x] Verify tool outputs match v1 outputs

## Phase 2: Agent Core (Week 2–3)
**Goal:** A smart ReAct agent that can plan and execute analysis.

- [x] **Agent Implementation**
  - [x] Initialize LangChain ReAct agent
  - [x] Register the 12-tool registry with the agent
  - [x] configure LLM (GPT-4o / Gemini 1.5 Pro) integration

- [x] **Context Management**
  - [x] Build Dataset Context Manager (schema, dtypes, sample rows)
  - [x] Implement context injection into system prompt

- [x] **Agent Logic**
  - [x] Implement structured output parser for "Final Answer"
  - [x] Implement error handling (retries on tool failure)
  - [x] Create benchmark query set (30 queries) for testing
  - [x] Optimize system prompt for tool selection accuracy

## Phase 3: Frontend Shell (Week 3–4)
**Goal:** Next.js application structure and real-time communication.

- [ ] **Frontend Setup**
  - [ ] Initialize Next.js 15 (App Router) project
  - [ ] Install and configure Tailwind CSS
  - [ ] Install `shadcn/ui` and configure dark theme components
  - [ ] Set up project directory structure (`components/`, `lib/`, `hooks/`)

- [ ] **Core Layout**
  - [ ] Implement Three-Panel Layout (Sidebar, Chat, Viz)
  - [ ] Build Sidebar component (File info, session controls)

- [ ] **Communication**
  - [ ] Implement `useWebSocket` hook for real-time streaming
  - [ ] Define TypeScript interfaces for WebSocket message types (`thought`, `tool_call`, `chart_spec`, etc.)

- [ ] **File Upload**
  - [ ] Create Drag-and-Drop upload component
  - [ ] Connect upload to backend API
  - [ ] Add upload progress indicator

- [ ] **Chat Interface**
  - [ ] Build `AgentChat` component
  - [ ] Create `ThoughtStep` component for streaming reasoning
  - [ ] Create `ToolCallCard` component for showing tool execution

## Phase 4: Visualizations (Week 4–5)
**Goal:** High-quality 2D and 3D data visualization.

- [ ] **2D Visualizations (Recharts/Visx)**
  - [ ] Implement `Chart2D` wrapper component
  - [ ] Build Histogram component
  - [ ] Build Scatter plot component
  - [ ] Build Line chart component
  - [ ] Build Heatmap component (Visx)
  - [ ] Build Bar chart component

- [ ] **3D Visualizations (React Three Fiber)**
  - [ ] Set up `Scene3D` canvas with `OrbitControls`
  - [ ] Implement `PCAScatter3D` component
  - [ ] Implement `ClusterOrbs` component (K-means viz)
  - [ ] Implement `UMAP3D` component with animation
  - [ ] Ensure responsive resizing of canvas

- [ ] **Integration**
  - [ ] Update `generate_chart_spec` tool to produce compatible JSON
  - [ ] Implement chart rendering logic in frontend based on agent output

## Phase 5: Polish & Deploy (Week 5–6)
**Goal:** Production-ready release.

- [ ] **UX Polish**
  - [ ] Implement "Suggested Queries" chips based on data type
  - [ ] Add loading states and transitions
  - [ ] Polish dark mode color palette

- [ ] **Features**
  - [ ] Implement Export functionality (PNG, PDF, CSV)
  - [ ] Add "Clear Session" / Reset functionality

- [ ] **Performance & Deployment**
  - [ ] Test with large files (50MB+) and optimize
  - [ ] Create `Dockerfile` for Backend
  - [ ] Create `Dockerfile` for Frontend
  - [ ] Create `docker-compose.yml` for full stack orchestration
  - [x] Write `README.md` with setup instructions and demo GIF
