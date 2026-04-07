"use client"

import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { AgentAvatar } from "./AgentAvatar"
import { ThinkingBlock } from "./ThinkingBlock"
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
  content?: string         // user message text
  thoughts?: string[]     // agent reasoning steps
  toolCalls?: ToolCall[]  // agent tool calls
  answer?: string         // final answer
  thinking: boolean       // is agent still generating this turn
}

const SUGGESTIONS = [
  "What columns does this dataset have?",
  "Find anomalies in the data",
  "Show correlations between variables",
  "Cluster the data into groups",
]

const now = () => new Date().toLocaleTimeString("en-US", { hour12: false })
const uid = () => `${Date.now()}-${Math.random().toString(36).slice(2)}`

export function AgentChat({ sessionId, onChartSpec, onAgentStateChange, onTimelineStep }: AgentChatProps) {
  const [input, setInput] = useState("")
  const [turns, setTurns] = useState<Turn[]>([])
  const [agentState, setAgentState] = useState<AgentState>("idle")
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const setAgent = (state: AgentState) => {
    setAgentState(state)
    onAgentStateChange?.(state)
  }

  // Helpers to mutate the latest agent turn
  const updateLatestAgentTurn = (updater: (turn: Turn) => Turn) => {
    setTurns((prev) => {
      const idx = [...prev].reverse().findIndex((t) => t.role === "agent")
      if (idx === -1) return prev
      const realIdx = prev.length - 1 - idx
      const updated = [...prev]
      updated[realIdx] = updater(updated[realIdx])
      return updated
    })
  }

  const ensureAgentTurn = (cb: (turns: Turn[]) => Turn[]) => {
    setTurns((prev) => {
      const last = prev[prev.length - 1]
      if (last?.role === "agent" && last.thinking) return cb(prev)
      const newTurn: Turn = { id: uid(), role: "agent", thoughts: [], toolCalls: [], thinking: true }
      return cb([...prev, newTurn])
    })
  }

  const { sendMessage, isConnected } = useWebSocket({
    sessionId,
    onThought: (content) => {
      setAgent("thinking")
      onTimelineStep?.({ label: "thinking", timestamp: now() })
      ensureAgentTurn((turns) => {
        const updated = [...turns]
        const last = updated[updated.length - 1]
        updated[updated.length - 1] = { ...last, thoughts: [...(last.thoughts ?? []), content] }
        return updated
      })
    },
    onToolCall: (tool, args) => {
      setAgent("executing")
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
      setAgent("done")
      onTimelineStep?.({ label: "answer", timestamp: now() })
      updateLatestAgentTurn((turn) => ({ ...turn, answer: content, thinking: false }))
      setTimeout(() => setAgent("idle"), 2000)
    },
    onChart: (spec) => {
      onTimelineStep?.({ label: "chart", timestamp: now() })
      onChartSpec?.(spec as VegaLiteSpec)
    },
    onError: (message) => {
      setAgent("error")
      updateLatestAgentTurn((turn) => ({ ...turn, answer: `Error: ${message}`, thinking: false }))
      setTimeout(() => setAgent("idle"), 2000)
    },
    onDone: () => setAgent("idle"),
  })

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [turns])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || !isConnected) return
    setTurns((p) => [...p, { id: uid(), role: "user", content: input, thinking: false }])
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

      {/* Turns */}
      <ScrollArea className="flex-1 px-5 py-5">
        <div className="space-y-6">
          {/* Connection error */}
          {!isConnected && turns.length === 0 && (
            <div className="rounded border border-error/20 bg-error/5 p-4 text-center">
              <p className="text-sm text-error">Connection failed</p>
              <p className="mt-1 text-xs text-text-muted">Ensure backend is running at http://localhost:8000</p>
            </div>
          )}

          {/* Empty state */}
          {turns.length === 0 && isConnected && (
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
                        <div className="h-1.5 w-1.5 rounded-full bg-success" />
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
