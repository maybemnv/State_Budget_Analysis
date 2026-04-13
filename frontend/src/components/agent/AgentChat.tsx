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
import type { VegaLiteSpec } from "@/lib/types"
import { ArrowUp, Sparkles } from "lucide-react"

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
const MAX_TURNS = 50

export function AgentChat({ sessionId, onChartSpec, onTimelineStep }: AgentChatProps) {
  const [input, setInput] = useState("")
  const [turns, setTurns] = useState<Turn[]>([])
  const [showScrollButton, setShowScrollButton] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const isSendingRef = useRef(false)
  const shouldAutoScrollRef = useRef(true)

  const agentState = useWorkspaceStore((s) => s.agentState)
  const setAgentState = useWorkspaceStore((s) => s.setAgentState)

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
  })

  // Auto-scroll to bottom on new turns (only if user hasn't scrolled up)
  useEffect(() => {
    if (shouldAutoScrollRef.current && viewportRef.current) {
      const viewport = viewportRef.current
      const scrollContainer = viewport.querySelector("[data-radix-scroll-area-viewport]")
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }, [turns])

  // Detect manual scroll (user scrolling up = disable auto-scroll)
  useEffect(() => {
    const viewport = viewportRef.current?.querySelector("[data-radix-scroll-area-viewport]")
    if (!viewport) return

    const handleScroll = () => {
      const el = viewport as HTMLElement
      const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
      shouldAutoScrollRef.current = isNearBottom
      setShowScrollButton(!isNearBottom)
    }

    viewport.addEventListener("scroll", handleScroll, { passive: true })
    return () => viewport.removeEventListener("scroll", handleScroll)
  }, [])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || !isConnected || isSendingRef.current) return

    isSendingRef.current = true
    shouldAutoScrollRef.current = true
    setShowScrollButton(false)
    setTurns((prev) => {
      const trimmedTurns = prev.length >= MAX_TURNS ? prev.slice(-MAX_TURNS + 1) : prev
      return [...trimmedTurns, { id: uid(), role: "user", content: trimmed, thinking: false }]
    })

    sendMessage({ message: trimmed })
    setInput("")

    setTimeout(() => {
      isSendingRef.current = false
    }, 300)
  }, [input, isConnected, sendMessage])

  const handleSuggestionClick = useCallback((question: string) => {
    setInput(question)
    inputRef.current?.focus()
  }, [])

  const scrollToBottom = useCallback(() => {
    shouldAutoScrollRef.current = true
    setShowScrollButton(false)
    const viewport = viewportRef.current?.querySelector("[data-radix-scroll-area-viewport]")
    if (viewport) {
      (viewport as HTMLElement).scrollTo({ top: (viewport as HTMLElement).scrollHeight, behavior: "smooth" })
    }
  }, [])

  const connectionStatus = isReconnecting
    ? `Reconnecting… (${reconnectAttempts}/10)`
    : wsError
      ? wsError
      : !isConnected
        ? "Connecting…"
        : null

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-border px-4 py-2.5 sm:px-5" role="banner">
        <div className="flex items-center gap-3">
          <AgentAvatar state={agentState} />
          <div>
            <p className="text-sm font-semibold text-text-primary">DataLens Agent</p>
            <p className="text-[11px] text-text-muted" aria-live="polite">
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
      {connectionStatus && turns.length === 0 && (
        <div className={cn(
          "mx-4 mt-3 shrink-0 rounded border px-4 py-3 text-center text-sm",
          wsError ? "border-error/20 bg-error/5 text-error" : "border-warning/20 bg-warning/5 text-warning"
        )} role="alert">
          {connectionStatus}
        </div>
      )}

      {/* Chat messages — scrollable area */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <ScrollArea className="h-full" ref={viewportRef}>
          <div className="px-4 py-4 sm:px-5 sm:py-5">
            {/* Empty state */}
            {turns.length === 0 && isConnected && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <Sparkles className="mb-4 h-12 w-12 text-primary/20" aria-hidden="true" />
                <h3 className="mb-1.5 text-lg font-semibold text-text-primary">
                  Ask anything about your data
                </h3>
                <p className="mb-8 text-sm text-text-muted max-w-sm">
                  The agent reasons step-by-step and shows its work.
                </p>
                <div className="flex flex-wrap justify-center gap-2" role="list">
                  {SUGGESTIONS.map((q) => (
                    <button
                      key={q}
                      onClick={() => handleSuggestionClick(q)}
                      className="rounded-full border border-border bg-surface px-4 py-2 text-xs text-text-secondary transition-colors hover:border-border-hover hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                      role="listitem"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Turn rendering */}
            <div className="space-y-6">
              {turns.map((turn) => {
                if (turn.role === "user") {
                  return (
                    <div key={turn.id} className="flex justify-end">
                      <div className="max-w-[80%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground shadow-sm">
                        {turn.content}
                      </div>
                    </div>
                  )
                }

                return (
                  <div key={turn.id} className="space-y-3">
                    {/* Thinking block */}
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
                        "overflow-hidden rounded-lg border animate-fade-in-up",
                        turn.answer.startsWith("Error:")
                          ? "border-error/30 bg-error/5"
                          : "border-success/30 bg-success/[0.03]"
                      )}>
                        {!turn.answer.startsWith("Error:") && (
                          <div className="flex items-center gap-2 border-b border-border/50 px-4 py-2">
                            <div className="h-1.5 w-1.5 rounded-full bg-success" aria-hidden="true" />
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-success">
                              Answer
                            </span>
                          </div>
                        )}
                        <div className={cn(
                          "px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap",
                          turn.answer.startsWith("Error:") ? "text-error" : "text-text-primary"
                        )}>
                          {turn.answer}
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </ScrollArea>
      </div>

      {/* Scroll to bottom button (appears when user scrolls up) */}
      {showScrollButton && turns.length > 0 && (
        <div className="absolute bottom-24 right-6 z-10">
          <button
            onClick={scrollToBottom}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-surface border border-border text-text-muted shadow-lg transition-colors hover:text-text-primary"
            aria-label="Scroll to bottom"
          >
            <ArrowUp className="h-4 w-4 rotate-180" />
          </button>
        </div>
      )}

      {/* Input — sticky at bottom */}
      <div className="shrink-0 border-t border-border bg-background px-4 pb-4 pt-3 sm:px-5 sm:pb-5 sm:pt-4">
        <form onSubmit={handleSubmit} role="form" aria-label="Send a message">
          <div className="flex items-end gap-2">
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isConnected ? "Ask about your data…" : "Connecting…"}
              disabled={!isConnected}
              className="flex-1 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm text-text-primary placeholder:text-text-disabled outline-none transition-colors focus:border-primary focus:ring-1 focus:ring-primary/20 disabled:opacity-50"
              aria-label="Message input"
              autoComplete="off"
            />
            <button
              type="submit"
              disabled={!input.trim() || !isConnected}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-colors hover:bg-primary-hover disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              aria-label="Send message"
            >
              <ArrowUp className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
