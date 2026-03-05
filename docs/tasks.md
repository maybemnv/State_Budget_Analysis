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

- [ ] **Frontend Setup**
  - [ ] Initialize Next.js 15 (App Router) project
  - [ ] Install and configure Tailwind CSS with custom theme
  - [ ] Install `shadcn/ui` — use as base, not final design
  - [ ] Set up typography: Geist Mono (body), Satoshi (headings)
  - [ ] Configure color tokens: warm black (#0A0A0F), burnt orange (#FF6B35), teal (#00DCB4)

- [ ] **Core Layout**
  - [ ] Implement Three-Panel Layout (Sidebar 280px, Chat flexible, Viz 360px)
  - [ ] Build Sidebar component (File info, session controls, column browser)
  - [ ] Implement agent timeline scrubber at bottom

- [ ] **Agent Components**
  - [ ] Build `AgentAvatar` — animated orb that pulses/shifts color
  - [ ] Create `ThoughtStep` with typewriter animation (~60 WPM)
  - [ ] Create `ToolCallCard` that builds itself (args populate one by one)
  - [ ] Implement `AgentChat` with message thread

- [ ] **Communication**
  - [ ] Implement `useWebSocket` hook with message queue
  - [ ] Define TypeScript interfaces for WebSocket message types
  - [ ] Add typing indicator when agent is thinking

- [ ] **File Upload**
  - [ ] Create Drag-and-Drop upload component
  - [ ] Implement "unfold" animation — rows animate in as parsed
  - [ ] Connect upload to backend API
  - [ ] Add upload progress indicator

- [ ] **Command Palette**
  - [ ] Implement `Cmd+K` command palette
  - [ ] Context-aware suggested queries based on data type
  - [ ] Keyboard-first navigation

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
  - [ ] Create `Dockerfile` for Backend
  - [ ] Create `Dockerfile` for Frontend
  - [ ] Create `docker-compose.yml` for full stack orchestration
  - [x] Write `README.md` with setup instructions and test commands

---

## Design Checklist — "Does This Feel Alive?"

Before marking any UI task complete, ask:

- [ ] Does it have **weight**? (animations with easing, not linear)
- [ ] Does it **respond**? (hover states, focus rings, clicks)
- [ ] Does it **build itself**? (nothing appears instantly — everything animates in)
- [ ] Does it fit the **terminal aesthetic**? (monospace data, warm blacks, burnt orange accents)
- [ ] Would a user **screenshot this**? (if not, why not?)

---

## Test Suite Status

| Suite | Tests | Status |
|-------|-------|--------|
| `test_api.py` | 6 | ✅ Passing |
| `test_statistical.py` | 11 | ✅ Passing |
| `test_ml.py` | 8 | ✅ Passing |
| `test_time_series.py` | 10 | ✅ Passing |
| `test_benchmarks.py` | 35 | ✅ Passing |
| **Total** | **70** | **✅ All Passing** |

---

## Open Tasks — Priority Order

**P0 (Do First):**
1. Custom theme (warm black, burnt orange, teal) — replaces default slate/indigo
2. Typewriter effect on agent thoughts
3. Agent avatar (animated orb)

**P1 (Core Experience):**
4. Tool call cards that build themselves
5. 3D visualizations with depth-of-field + glow
6. Cmd+K command palette

**P2 (Differentiators):**
7. Force-directed correlation graph
8. Auto-Insight Mode
9. Agent timeline scrubber

**P3 (Polish):**
10. Sound design
11. Chart draw-in animations
12. Viz canvas expand on insight
