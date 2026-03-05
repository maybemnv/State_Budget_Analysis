# DataLens AI v2.0 - Implementation Tasks

Based on [PRD v2.0](docs/PRD.md).

## Phase 1: Backend Foundation (Week 1–2)
**Goal:** robust FastAPI server with migrated logic and WebSocket support.

- [ ] **Project Setup**
  - [ ] Initialize FastAPI project structure (`backend/`)
  - [ ] Configure `uv` for dependency management
  - [ ] Set up `pyproject.toml` with dependencies (FastAPI, LangChain, Pandas, Scikit-learn, etc.)

- [ ] **Migration**
  - [ ] Migrate `src/analyzers/statistical_analyzer.py` to `backend/analyzers/`
  - [ ] Migrate `src/analyzers/ml_analyzer.py` to `backend/analyzers/`
  - [ ] Migrate `src/time_series/` modules to `backend/analyzers/time_series/`
  - [ ] Refactor migrated code to be pure functions (remove any Streamlit dependencies)

- [ ] **Tool Wrapping**
  - [ ] Create Pydantic input schemas for all analyzer functions
  - [ ] Wrap functions as LangChain tools (approx. 12 tools)
  - [ ] Implement `describe_dataset` tool
  - [ ] Implement `generate_chart_spec` tool

- [ ] **API Core**
  - [ ] Implement WebSocket endpoint (`/ws/{client_id}`)
  - [ ] Create streaming callback handler for LangChain to WebSocket
  - [ ] Implement file upload endpoint (`/upload`) with validation
  - [ ] Create session management for file/dataframe context

- [ ] **Testing**
  - [ ] Write integration tests for all migrated tools
  - [ ] Verify tool outputs match v1 outputs

## Phase 2: Agent Core (Week 2–3)
**Goal:** A smart ReAct agent that can plan and execute analysis.

- [ ] **Agent Implementation**
  - [ ] Initialize LangChain ReAct agent
  - [ ] Register the 12-tool registry with the agent
  - [ ] configure LLM (GPT-4o / Gemini 1.5 Pro) integration

- [ ] **Context Management**
  - [ ] Build Dataset Context Manager (schema, dtypes, sample rows)
  - [ ] Implement context injection into system prompt

- [ ] **Agent Logic**
  - [ ] Implement structured output parser for "Final Answer"
  - [ ] Implement error handling (retries on tool failure)
  - [ ] Create benchmark query set (30 queries) for testing
  - [ ] Optimize system prompt for tool selection accuracy

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
  - [ ] Write `README.md` with setup instructions and demo GIF
