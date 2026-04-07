"use client"

import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { AgentAvatar } from "./AgentAvatar"
import { ThoughtStep } from "./ThoughtStep"
import { ToolCallCard } from "./ToolCallCard"
import { BackendStatusIndicator } from "@/components/ui/BackendStatusIndicator"
import { useWebSocket } from "@/hooks/useWebSocket"
import type { AgentState, VegaLiteSpec } from "@/lib/types"
import { ArrowRight, Command, Sparkles } from "lucide-react"

interface AgentChatProps {
  sessionId: string
  onChartSpec?: (spec: VegaLiteSpec) => void
  onAgentStateChange?: (state: AgentState) => void
  onTimelineStep?: (step: { label: string; timestamp: string }) => void
}

interface ChatMessage {
  id: string
  type: "user" | "thought" | "tool_call" | "tool_result" | "answer" | "error"
  content: string | unknown
  tool?: string
  args?: Record<string, unknown>
  timestamp: Date
}

const SUGGESTIONS = [
  "What columns does this dataset have?",
  "Find anomalies in the data",
  "Show correlations between variables",
  "Cluster the data into groups",
]

const now = () => new Date().toLocaleTimeString("en-US", { hour12: false })
const uid = (prefix: string) => `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`

export function AgentChat({ sessionId, onChartSpec, onAgentStateChange, onTimelineStep }: AgentChatProps) {
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [agentState, setAgentState] = useState<AgentState>("idle")
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const setAgent = (state: AgentState) => {
    setAgentState(state)
    onAgentStateChange?.(state)
  }

  const { sendMessage, isConnected } = useWebSocket({
    sessionId,
    onThought: (content) => {
      setAgent("thinking")
      onTimelineStep?.({ label: "thinking", timestamp: now() })
      setMessages((p) => [...p, { id: uid("thought"), type: "thought", content, timestamp: new Date() }])
    },
    onToolCall: (tool, args) => {
      setAgent("executing")
      onTimelineStep?.({ label: tool, timestamp: now() })
      setMessages((p) => [...p, { id: uid("tool"), type: "tool_call", tool, args, content: "", timestamp: new Date() }])
    },
    onToolResult: (_, result) => {
      setMessages((p) => {
        const lastToolIdx = [...p].reverse().findIndex((m) => m.type === "tool_call")
        if (lastToolIdx === -1) return p
        const idx = p.length - 1 - lastToolIdx
        const updated = [...p]
        updated[idx] = { ...updated[idx], type: "tool_result", content: result }
        return updated
      })
    },
    onAnswer: (content) => {
      setAgent("done")
      onTimelineStep?.({ label: "answer", timestamp: now() })
      setMessages((p) => [...p, { id: uid("answer"), type: "answer", content, timestamp: new Date() }])
      setTimeout(() => setAgent("idle"), 2000)
    },
    onChart: (spec) => {
      onTimelineStep?.({ label: "chart", timestamp: now() })
      onChartSpec?.(spec as VegaLiteSpec)
    },
    onError: (message) => {
      setAgent("error")
      setMessages((p) => [...p, { id: uid("error"), type: "error", content: message, timestamp: new Date() }])
      setTimeout(() => setAgent("idle"), 2000)
    },
    onDone: () => setAgent("idle"),
  })

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || !isConnected) return
    setMessages((p) => [...p, { id: uid("user"), type: "user", content: input, timestamp: new Date() }])
    sendMessage({ message: input })
    setInput("")
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-5 py-3">
        <div className="flex items-center gap-3">
          <AgentAvatar state={agentState} />
          <div>
            <p className="text-sm font-semibold text-text-primary">DataLens Agent</p>
            <p className="text-[11px] text-text-muted">
              {agentState === "thinking" && "Reasoning..."}
              {agentState === "executing" && "Running analysis..."}
              {agentState === "done" && "Ready"}
              {agentState === "error" && "Error"}
              {agentState === "idle" && (isConnected ? "Connected" : "Connecting...")}
            </p>
          </div>
        </div>
        <BackendStatusIndicator />
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 px-5 py-5">
        <div className="space-y-4">
          {/* Connection error */}
          {!isConnected && messages.length === 0 && (
            <div className="rounded border border-error/20 bg-error/5 p-4 text-center">
              <p className="text-sm text-error">Connection failed</p>
              <p className="mt-1 text-xs text-text-muted">Ensure backend is running at http://localhost:8000</p>
            </div>
          )}

          {/* Empty state */}
          {messages.length === 0 && isConnected && (
            <div className="flex flex-col items-center justify-center py-14 text-center">
              <Sparkles className="mb-4 h-10 w-10 text-primary/30" />
              <h3 className="mb-1 text-base font-semibold text-text-primary">Ask anything about your data</h3>
              <p className="mb-6 text-sm text-text-muted">The agent reasons step-by-step and shows its work.</p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); inputRef.current?.focus() }}
                    className="rounded-full border border-border px-3.5 py-1.5 text-xs text-text-secondary transition-colors hover:border-border-hover hover:text-text-primary"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message list */}
          {messages.map((msg, i) => {
            if (msg.type === "user") {
              return (
                <div key={msg.id} className="flex justify-end">
                  <div className="max-w-[78%] rounded-lg bg-primary px-4 py-2.5 text-sm text-primary-foreground">
                    {String(msg.content)}
                  </div>
                </div>
              )
            }
            if (msg.type === "thought") {
              return <ThoughtStep key={msg.id} content={String(msg.content)} index={i} />
            }
            if (msg.type === "tool_call" || msg.type === "tool_result") {
              return (
                <ToolCallCard
                  key={msg.id}
                  tool={msg.tool ?? "unknown"}
                  args={msg.args}
                  result={msg.type === "tool_result" ? msg.content : undefined}
                  status={msg.type === "tool_call" ? "executing" : "completed"}
                  index={i}
                />
              )
            }
            if (msg.type === "answer") {
              return (
                <div key={msg.id} className="overflow-hidden rounded-lg border-l-2 border-success bg-elevated animate-fade-in-up">
                  <div className="flex items-center gap-2 border-b border-border px-4 py-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-success" />
                    <span className="text-[10px] font-semibold uppercase tracking-widest text-success">Final Answer</span>
                  </div>
                  <div className="px-4 py-3 text-sm leading-relaxed text-text-primary">
                    {String(msg.content)}
                  </div>
                </div>
              )
            }
            if (msg.type === "error") {
              return (
                <div key={msg.id} className="rounded border border-error/20 bg-error/5 px-4 py-3 text-sm text-error animate-fade-in-up">
                  {String(msg.content)}
                </div>
              )
            }
            return null
          })}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-border px-5 pb-5 pt-4">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isConnected ? "Type your question..." : "Connecting..."}
            disabled={!isConnected}
            className="flex-1 border-0 border-b-2 border-border bg-transparent py-2 text-sm text-text-primary placeholder:text-text-disabled outline-none transition-colors focus:border-primary disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || !isConnected}
            className="flex h-9 w-9 items-center justify-center rounded bg-primary text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-40"
          >
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-2 flex items-center gap-1 text-[11px] text-text-disabled">
          <Command className="h-3 w-3" />
          <span>K for suggestions</span>
        </div>
      </form>
    </div>
  )
}
