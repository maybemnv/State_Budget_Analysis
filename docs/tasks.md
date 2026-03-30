# DataLens AI v2.0 - Implementation Tasks

Based on [PRD v2.0](docs/PRD.md) — **Terminal Aesthetic Edition**

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
  - [x] Wrap functions as LangChain tools (15 tools total)
  - [x] Implement `describe_dataset` tool
  - [x] Implement `generate_chart_spec` tool

- [x] **API Core**
  - [x] Implement WebSocket endpoint (`/ws/{session_id}`)
  - [x] Create streaming callback handler for LangChain to WebSocket
  - [x] Implement file upload endpoint (`/upload`) with validation
  - [x] Create session management for file/dataframe context

- [x] **Testing**
  - [x] Write integration tests for all migrated tools
  - [x] Verify tool outputs match v1 outputs
  - [x] 70 tests passing across API, statistical, ML, time series, benchmarks

## Phase 2: Agent Core (Week 2–3)
**Goal:** A smart ReAct agent that can plan and execute analysis.

- [x] **Agent Implementation**
  - [x] Initialize LangChain ReAct agent
  - [x] Register the 15-tool registry with the agent
  - [x] Configure LLM (Gemini 1.5 Pro / GPT-4o) integration

- [x] **Context Management**
  - [x] Build Dataset Context Manager (schema, dtypes, sample rows)
  - [x] Implement context injection into system prompt

- [x] **Agent Logic**
  - [x] Implement structured output parser for "Final Answer"
  - [x] Implement error handling (retries on tool failure)
  - [x] Create benchmark query set (30 queries) for testing
  - [x] Optimize system prompt for tool selection accuracy

## Phase 3: Frontend Shell (Week 3–4)
**Goal:** Next.js 15 application with terminal aesthetic and real-time communication.

- [X] **Frontend Setup**
  - [X] Initialize Next.js 15 (App Router) project
  - [X] Install and configure Tailwind CSS with custom theme
  - [X] Install `shadcn/ui` — use as base, not final design
  - [X] Set up typography: Geist Mono (body), Satoshi (headings)
  - [X] Configure color tokens: warm black (#0A0A0F), burnt orange (#FF6B35), teal (#00DCB4)

- [X] **Core Layout**
  - [X] Implement Three-Panel Layout (Sidebar 280px, Chat flexible, Viz 360px)
  - [X] Build Sidebar component (File info, session controls, column browser)
  - [X] Implement agent timeline scrubber at bottom

- [X] **Agent Components**
  - [X] Build `AgentAvatar` — animated orb that pulses/shifts color
  - [X] Create `ThoughtStep` with typewriter animation (~60 WPM)
  - [X] Create `ToolCallCard` that builds itself (args populate one by one)
  - [X] Implement `AgentChat` with message thread

- [X] **Communication**
  - [X] Implement `useWebSocket` hook with message queue
  - [X] Define TypeScript interfaces for WebSocket message types
  - [X] Add typing indicator when agent is thinking

- [X] **File Upload**
  - [X] Create Drag-and-Drop upload component
  - [X] Implement "unfold" animation — rows animate in as parsed
  - [X] Connect upload to backend API
  - [X] Add upload progress indicator

- [X] **Command Palette**
  - [X] Implement `Cmd+K` command palette
  - [X] Context-aware suggested queries based on data type
  - [X] Keyboard-first navigation

## Phase 4: Visualizations (Week 4–5)
**Goal:** High-quality 2D and 3D data visualization with post-processing effects.

- [ ] **2D Visualizations (Recharts/Visx)**
  - [ ] Implement `Chart2D` wrapper with draw-in animations
  - [ ] Build Histogram — bars grow from zero with stagger
  - [ ] Build Scatter plot — points fade in
  - [ ] Build Line chart — line traces left-to-right
  - [ ] Build Force-Directed Graph for correlations (Visx)
  - [ ] Build Bar chart — horizontal for SHAP importance

- [ ] **3D Visualizations (React Three Fiber + Drei + PostFX)**
  - [ ] Set up `Scene3D` canvas with `OrbitControls`
  - [ ] Configure PostFX: depth-of-field, bloom glow
  - [ ] Implement `PCAScatter3D` — glow on hover, depth blur on distant points
  - [ ] Implement `ClusterOrbs` — refractive semi-transparent spheres
  - [ ] Implement `UMAPEmbedding` — animated morph between n_neighbors values
  - [ ] Add particle trails for cluster confidence
  - [ ] Ensure responsive resizing of canvas

- [ ] **Integration**
  - [ ] Update `generate_chart_spec` tool to produce compatible JSON
  - [ ] Implement chart rendering logic based on agent output
  - [ ] Viz canvas expands on "big insight" detection

## Phase 5: Polish & Deploy (Week 5–6)
**Goal:** Production-ready release with soul.

- [ ] **UX Polish**
  - [ ] Implement "Suggested Queries" via Cmd+K (not chips)
  - [ ] Add loading states with agent avatar pulses
  - [ ] Polish warm dark mode color palette
  - [ ] Agent timeline scrubber — click to jump to reasoning step

- [ ] **Features**
  - [ ] **Auto-Insight Mode** — agent scans data, surfaces 3-5 non-obvious findings
  - [ ] Implement Export functionality (PNG, PDF, CSV)
  - [ ] Add "Clear Session" / Reset functionality
  - [ ] Sound design (optional, muted by default): keyboard clicks, pops, chimes

- [ ] **Performance & Deployment**
  - [ ] Test with large files (50MB+) and optimize
  - [X] Create `Dockerfile` for Backend
  - [X] Create `Dockerfile` for Frontend
  - [X] Create `docker-compose.yml` for full stack orchestration
  - [x] Write `README.md` with setup instructions and test commands

---

## Design Checklist — "Does This Feel Alive?"

Before marking any UI task complete, ask:

- [ ] Does it have **weight**? (animations with easing, not linear)
- [ ] Does it **respond**? (hover states, focus rings, clicks)
- [ ] Does it **build itself**? (nothing appears instantly — everything animates in)
- [ ] Does it fit the **terminal aesthetic**? (monospace data, warm blacks, burnt orange accents)
- [ ] Would a user **screenshot this**? (if not, why not?)