"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { AgentAvatar } from "./AgentAvatar"
import { ThinkingBlock } from "./ThinkingBlock"
import { ToolCallCard } from "./ToolCallCard"
import { BackendStatusIndicator } from "@/components/ui/BackendStatusIndicator"
import { useWebSocket } from "@/hooks/useWebSocket"
import { useWorkspaceStore } from "@/lib/store"
import type { AgentState, VegaLiteSpec } from "@/lib/types"
import { ArrowRight, Command, Sparkles } from "lucide-react"

interface AgentChatProps {
  sessionId: string
  onChartSpec?: (spec: VegaLiteSpec) => void
  onTimelineStep?: (step: { label: string; timestamp: string }) => void
}

interface ToolCall {
  id: string
  tool: string
  args?: Record<string, unknown>
  result?: unknown
  done: boolean
}

interface Turn {
  id: string
  role: "user" | "agent"
  content?: string
  thoughts?: string[]
  toolCalls?: ToolCall[]
  answer?: string
  thinking: boolean
}

const SUGGESTIONS = [
  "What columns does this dataset have?",
  "Find anomalies in the data",
  "Show correlations between variables",
  "Cluster the data into groups",
]

const now = () => new Date().toLocaleTimeString("en-US", { hour12: false })
const uid = () => `${Date.now()}-${Math.random().toString(36).slice(2)}`

// Maximum turns to keep in memory — prevents memory leaks in long conversations
const MAX_TURNS = 50

export function AgentChat({ sessionId, onChartSpec, onTimelineStep }: AgentChatProps) {
  const [input, setInput] = useState("")
  const [turns, setTurns] = useState<Turn[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const isSendingRef = useRef(false)

  // Zustand store — shared state
  const agentState = useWorkspaceStore((s) => s.agentState)
  const setAgentState = useWorkspaceStore((s) => s.setAgentState)

  // Helpers to mutate the latest agent turn — using functional updates to avoid stale closures
  const updateLatestAgentTurn = useCallback((updater: (turn: Turn) => Turn) => {
    setTurns((prev) => {
      const idx = [...prev].reverse().findIndex((t) => t.role === "agent")
      if (idx === -1) return prev
      const realIdx = prev.length - 1 - idx
      const updated = [...prev]
      updated[realIdx] = updater(updated[realIdx])
      return updated
    })
  }, [])

  const ensureAgentTurn = useCallback((cb: (turns: Turn[]) => Turn[]) => {
    setTurns((prev) => {
      const last = prev[prev.length - 1]
      if (last?.role === "agent" && last.thinking) return cb(prev)
      const newTurn: Turn = { id: uid(), role: "agent", thoughts: [], toolCalls: [], thinking: true }
      return cb([...prev, newTurn])
    })
  }, [])

  const { sendMessage, isConnected, isReconnecting, reconnectAttempts, error: wsError } = useWebSocket({
    sessionId,
    maxReconnectAttempts: 10,
    onThought: (content) => {
      setAgentState("thinking")
      onTimelineStep?.({ label: "thinking", timestamp: now() })
      ensureAgentTurn((turns) => {
        const updated = [...turns]
        const last = updated[updated.length - 1]
        updated[updated.length - 1] = { ...last, thoughts: [...(last.thoughts ?? []), content] }
        return updated
      })
    },
    onToolCall: (tool, args) => {
      setAgentState("executing")
      onTimelineStep?.({ label: tool, timestamp: now() })
      const callId = uid()
      ensureAgentTurn((turns) => {
        const updated = [...turns]
        const last = updated[updated.length - 1]
        updated[updated.length - 1] = {
          ...last,
          toolCalls: [...(last.toolCalls ?? []), { id: callId, tool, args, done: false }],
        }
        return updated
      })
    },
    onToolResult: (_, result) => {
      updateLatestAgentTurn((turn) => {
        const toolCalls = [...(turn.toolCalls ?? [])]
        const lastPending = [...toolCalls].reverse().findIndex((c) => !c.done)
        if (lastPending !== -1) {
          const realIdx = toolCalls.length - 1 - lastPending
          toolCalls[realIdx] = { ...toolCalls[realIdx], result, done: true }
        }
        return { ...turn, toolCalls }
      })
    },
    onAnswer: (content) => {
      setAgentState("done")
      onTimelineStep?.({ label: "answer", timestamp: now() })
      updateLatestAgentTurn((turn) => ({ ...turn, answer: content, thinking: false }))
      setTimeout(() => setAgentState("idle"), 2000)
    },
    onChart: (spec) => {
      onTimelineStep?.({ label: "chart", timestamp: now() })
      onChartSpec?.(spec as VegaLiteSpec)
    },
    onError: (message) => {
      setAgentState("error")
      updateLatestAgentTurn((turn) => ({ ...turn, answer: `Error: ${message}`, thinking: false }))
      setTimeout(() => setAgentState("idle"), 2000)
    },
    onDone: () => setAgentState("idle"),
    onDisconnect: () => {
      // Don't set error state immediately — the hook tracks reconnection
    },
  })

  // Auto-scroll to bottom on new turns
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [turns])

  // Keyboard shortcut: Cmd+K for command palette (placeholder)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    // Prevent double-sending
    if (!input.trim() || !isConnected || isSendingRef.current) return

    isSendingRef.current = true
    setTurns((prev) => {
      // Trim old turns to prevent memory leaks
      const trimmed = prev.length >= MAX_TURNS ? prev.slice(-MAX_TURNS + 1) : prev
      return [...trimmed, { id: uid(), role: "user", content: input.trim(), thinking: false }]
    })

    sendMessage({ message: input.trim() })
    setInput("")

    // Release lock after a short cooldown
    setTimeout(() => {
      isSendingRef.current = false
    }, 300)
  }, [input, isConnected, sendMessage])

  const handleSuggestionClick = useCallback((question: string) => {
    setInput(question)
    inputRef.current?.focus()
  }, [])

  // Connection status message
  const connectionStatus = isReconnecting
    ? `Reconnecting… (${reconnectAttempts}/10)`
    : wsError
      ? wsError
      : !isConnected
        ? "Connecting…"
        : null

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5 sm:px-5" role="banner" aria-label="Agent chat header">
        <div className="flex items-center gap-3">
          <AgentAvatar state={agentState} />
          <div>
            <p className="text-sm font-semibold text-text-primary">DataLens Agent</p>
            <p className="text-[11px] text-text-muted" aria-live="polite" aria-atomic="true">
              {agentState === "thinking" && "Reasoning…"}
              {agentState === "executing" && "Running analysis…"}
              {agentState === "done" && "Ready"}
              {agentState === "error" && "Error occurred"}
              {agentState === "idle" && (isConnected ? "Connected" : "Connecting…")}
            </p>
          </div>
        </div>
        <BackendStatusIndicator />
      </div>

      {/* Connection error banner */}
      {!isConnected && !wsError && turns.length === 0 && (
        <div className="mx-4 mt-3 rounded border border-warning/20 bg-warning/5 px-4 py-3 text-center" role="alert">
          <p className="text-sm text-warning">Connecting to agent backend…</p>
          <p className="mt-1 text-xs text-text-muted">Ensure backend is running at http://localhost:8000</p>
        </div>
      )}

      {/* Persistent connection error */}
      {wsError && turns.length === 0 && (
        <div className="mx-4 mt-3 rounded border border-error/20 bg-error/5 px-4 py-3 text-center" role="alert">
          <p className="text-sm text-error">Connection failed</p>
          <p className="mt-1 text-xs text-text-muted">{wsError}</p>
        </div>
      )}

      {/* Turns */}
      <ScrollArea className="flex-1 px-4 py-5 sm:px-5" aria-label="Chat messages" role="log" aria-live="polite">
        <div className="space-y-6">
          {/* Empty state */}
          {turns.length === 0 && isConnected && (
            <div className="flex flex-col items-center justify-center py-14 text-center">
              <Sparkles className="mb-4 h-10 w-10 text-primary/30" aria-hidden="true" />
              <h3 className="mb-1 text-base font-semibold text-text-primary">Ask anything about your data</h3>
              <p className="mb-6 text-sm text-text-muted">The agent reasons step-by-step and shows its work.</p>
              <div className="flex flex-wrap justify-center gap-2" role="list" aria-label="Suggested questions">
                {SUGGESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSuggestionClick(q)}
                    className="rounded-full border border-border px-3.5 py-1.5 text-xs text-text-secondary transition-colors hover:border-border-hover hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                    role="listitem"
                    aria-label={`Ask: ${q}`}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Turn rendering */}
          {turns.map((turn) => {
            if (turn.role === "user") {
              return (
                <div key={turn.id} className="flex justify-end">
                  <div className="max-w-[78%] rounded-lg bg-primary px-4 py-2.5 text-sm text-primary-foreground">
                    {turn.content}
                  </div>
                </div>
              )
            }

            // Agent turn
            return (
              <div key={turn.id} className="space-y-3">
                {/* Thinking block — collapsed when done */}
                {(turn.thoughts?.length ?? 0) > 0 && (
                  <ThinkingBlock
                    thoughts={turn.thoughts!}
                    isLive={turn.thinking}
                  />
                )}

                {/* Tool calls */}
                {turn.toolCalls?.map((tc, i) => (
                  <ToolCallCard
                    key={tc.id}
                    tool={tc.tool}
                    args={tc.args}
                    result={tc.result}
                    status={tc.done ? "completed" : "executing"}
                    index={i}
                  />
                ))}

                {/* Final answer */}
                {turn.answer && (
                  <div className={cn(
                    "overflow-hidden rounded-lg border-l-2 bg-elevated animate-fade-in-up",
                    turn.answer.startsWith("Error:")
                      ? "border-error bg-error/5"
                      : "border-success"
                  )}>
                    {!turn.answer.startsWith("Error:") && (
                      <div className="flex items-center gap-2 border-b border-border px-4 py-2">
                        <div className="h-1.5 w-1.5 rounded-full bg-success" aria-hidden="true" />
                        <span className="text-[10px] font-semibold uppercase tracking-widest text-success">Answer</span>
                      </div>
                    )}
                    <div className={cn(
                      "px-4 py-3 text-sm leading-relaxed",
                      turn.answer.startsWith("Error:") ? "text-error" : "text-text-primary"
                    )}>
                      {turn.answer}
                    </div>
                  </div>
                )}
              </div>
            )
          })}

          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-border px-4 pb-4 pt-3 sm:px-5 sm:pb-5 sm:pt-4" role="form" aria-label="Send a message">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isConnected ? "Type your question…" : "Connecting…"}
            disabled={!isConnected}
            className="flex-1 border-0 border-b-2 border-border bg-transparent py-2 text-sm text-text-primary placeholder:text-text-disabled outline-none transition-colors focus:border-primary disabled:opacity-50"
            aria-label="Message input"
            aria-describedby="chat-hint"
            autoComplete="off"
          />
          <button
            type="submit"
            disabled={!input.trim() || !isConnected}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded bg-primary text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            aria-label="Send message"
          >
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
        <div id="chat-hint" className="mt-2 flex items-center gap-1 text-[11px] text-text-disabled">
          <Command className="h-3 w-3" aria-hidden="true" />
          <span>K to focus input</span>
        </div>
      </form>
    </div>
  )
}
