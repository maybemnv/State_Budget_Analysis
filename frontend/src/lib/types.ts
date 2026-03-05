// WebSocket message types streamed from backend
export type WSMessageType =
  | ThoughtMessage
  | ToolCallMessage
  | ToolResultMessage
  | ChartSpecMessage
  | AnswerMessage
  | ErrorMessage
  | DoneMessage

export interface ThoughtMessage {
  type: "thought"
  content: string
}

export interface ToolCallMessage {
  type: "tool_call"
  tool: string
  args: Record<string, unknown>
}

export interface ToolResultMessage {
  type: "tool_result"
  tool: string
  result: unknown
}

export interface ChartSpecMessage {
  type: "chart"
  spec: VegaLiteSpec
}

export interface AnswerMessage {
  type: "answer"
  content: string
}

export interface ErrorMessage {
  type: "error"
  message: string
}

export interface DoneMessage {
  type: "done"
}

// Vega-Lite specification type
export interface VegaLiteSpec {
  $schema?: string
  title?: string | { text: string }
  mark: string | { type: string }
  encoding: Record<string, unknown>
  data?: { values: Record<string, unknown>[] }
  [key: string]: unknown
}

// Agent state for avatar
export type AgentState = "idle" | "thinking" | "executing" | "done" | "error"

// Session metadata
export interface SessionInfo {
  session_id: string
  filename: string
  shape: [number, number]
  columns: string[]
  dtypes: Record<string, string>
}

// Upload response
export interface UploadResponse {
  session_id: string
  filename: string
  rows: number
  columns: number
  column_names: string[]
}

// Chat response
export interface ChatResponse {
  answer: string
  chart_spec?: VegaLiteSpec
  has_error: boolean
  steps: Array<{
    tool: string
    args: Record<string, unknown>
    result: unknown
  }>
}

// Timeline step for agent reasoning
export interface TimelineStep {
  type: "describe" | "stats" | "anomaly" | "chart" | "answer" | string
  timestamp: string
  status: "pending" | "active" | "completed" | "error"
}
