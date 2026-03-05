"use client"

import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { AgentAvatar } from "./AgentAvatar"
import { ThoughtStep } from "./ThoughtStep"
import { ToolCallCard } from "./ToolCallCard"
import { useWebSocket } from "@/hooks/useWebSocket"
import type { AgentState, WSMessageType, VegaLiteSpec } from "@/lib/types"
import { Send, Sparkles } from "lucide-react"

interface AgentChatProps {
  sessionId: string
  onChartSpec?: (spec: VegaLiteSpec) => void
  onAgentStateChange?: (state: AgentState) => void
}

interface ChatMessage {
  id: string
  type: "user" | "thought" | "tool_call" | "tool_result" | "answer" | "error" | "chart"
  content: string | unknown
  tool?: string
  args?: Record<string, unknown>
  timestamp: Date
}

/**
 * AgentChat — Main chat interface for interacting with the DataLens agent.
 * Shows user messages, agent thoughts (typewriter), tool calls, and final answers.
 */
export function AgentChat({ sessionId, onChartSpec, onAgentStateChange }: AgentChatProps) {
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [agentState, setAgentState] = useState<AgentState>("idle")
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleAgentStateChange = (state: AgentState) => {
    setAgentState(state)
    onAgentStateChange?.(state)
  }

  const { sendMessage, isConnected } = useWebSocket({
    sessionId,
    onThought: (content) => {
      handleAgentStateChange("thinking")
      setMessages((prev) => [
        ...prev,
        {
          id: `thought-${Date.now()}`,
          type: "thought",
          content: content,
          timestamp: new Date(),
        },
      ])
    },
    onToolCall: (tool, args) => {
      handleAgentStateChange("executing")
      setMessages((prev) => [
        ...prev,
        {
          id: `tool-${Date.now()}`,
          type: "tool_call",
          tool: tool,
          args: args,
          content: "",
          timestamp: new Date(),
        },
      ])
    },
    onToolResult: (tool, result) => {
      setMessages((prev) => {
        // Find the last tool_call and update it with result
        const lastToolCallIndex = [...prev].reverse().findIndex((m) => m.type === "tool_call")
        if (lastToolCallIndex === -1) return prev

        const actualIndex = prev.length - 1 - lastToolCallIndex
        const updated = [...prev]
        updated[actualIndex] = { ...updated[actualIndex], type: "tool_result" as const, content: result }
        return updated
      })
    },
    onAnswer: (content) => {
      handleAgentStateChange("done")
      setMessages((prev) => [
        ...prev,
        {
          id: `answer-${Date.now()}`,
          type: "answer",
          content: content,
          timestamp: new Date(),
        },
      ])
      // Reset to idle after a delay
      setTimeout(() => handleAgentStateChange("idle"), 2000)
    },
    onChart: (spec) => {
      onChartSpec?.(spec as VegaLiteSpec)
    },
    onError: (message) => {
      handleAgentStateChange("error")
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          type: "error",
          content: message,
          timestamp: new Date(),
        },
      ])
      setTimeout(() => handleAgentStateChange("idle"), 2000)
    },
    onDone: () => {
      handleAgentStateChange("idle")
    },
  })

  // Auto-scroll to bottom
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || !isConnected) return

    // Add user message
    setMessages((prev) => [
      ...prev,
      {
        id: `user-${Date.now()}`,
        type: "user",
        content: input,
        timestamp: new Date(),
      },
    ])

    // Send to agent
    sendMessage({ message: input })
    setInput("")
  }

  const suggestedQueries = [
    "What columns does this dataset have?",
    "Find anomalies in the data",
    "Show me correlations between variables",
    "Cluster the data into groups",
  ]

  return (
    <div className="flex h-full flex-col">
      {/* Header with agent avatar */}
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-3">
          <AgentAvatar state={agentState} />
          <div>
            <h2 className="text-sm font-medium text-text-primary">DataLens Agent</h2>
            <p className="text-xs text-text-muted">
              {agentState === "thinking" && "Thinking..."}
              {agentState === "executing" && "Running analysis..."}
              {agentState === "done" && "Ready"}
              {agentState === "error" && "Encountered an error"}
              {agentState === "idle" && (isConnected ? "Connected" : "Connecting...")}
            </p>
          </div>
        </div>
        <div
          className={cn(
            "h-2 w-2 rounded-full",
            isConnected ? "bg-success animate-pulse" : "bg-text-muted"
          )}
        />
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 px-4 py-4">
        <div className="space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Sparkles className="mb-4 h-12 w-12 text-agent/50" />
              <h3 className="mb-2 text-lg font-medium text-text-primary">Start a conversation</h3>
              <p className="mb-6 max-w-md text-sm text-text-muted">
                Ask me anything about your data. I'll analyze it step by step.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {suggestedQueries.map((query) => (
                  <button
                    key={query}
                    onClick={() => {
                      setInput(query)
                      inputRef.current?.focus()
                    }}
                    className="rounded-full border border-border bg-surface/50 px-3 py-1.5 text-xs text-text-secondary transition-colors hover:border-agent/50 hover:text-text-primary"
                  >
                    {query}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, index) => {
            if (msg.type === "user") {
              return (
                <div key={msg.id} className="flex justify-end">
                  <div className="max-w-[80%] rounded-lg bg-primary px-4 py-2 text-sm text-background">
                    {String(msg.content)}
                  </div>
                </div>
              )
            }

            if (msg.type === "thought") {
              return <ThoughtStep key={msg.id} content={String(msg.content)} index={index} />
            }

            if (msg.type === "tool_call" || msg.type === "tool_result") {
              return (
                <ToolCallCard
                  key={msg.id}
                  tool={msg.tool || "unknown"}
                  args={msg.args}
                  result={msg.type === "tool_result" ? msg.content : undefined}
                  status={msg.type === "tool_call" ? "executing" : "completed"}
                  index={index}
                />
              )
            }

            if (msg.type === "answer") {
              return (
                <div
                  key={msg.id}
                  className="rounded-lg border border-success/30 bg-success/5 p-4 text-sm text-text-primary animate-fade-in-up"
                >
                  {String(msg.content)}
                </div>
              )
            }

            if (msg.type === "error") {
              return (
                <div
                  key={msg.id}
                  className="rounded-lg border border-error/30 bg-error/5 p-4 text-sm text-error animate-fade-in-up"
                >
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
      <form onSubmit={handleSubmit} className="border-t border-border p-4">
        <div className="flex gap-2">
          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isConnected ? "Ask anything about your data..." : "Connecting..."}
            disabled={!isConnected}
            className="flex-1 bg-surface/50 text-text-primary placeholder:text-text-muted"
          />
          <Button
            type="submit"
            disabled={!input.trim() || !isConnected}
            className="bg-agent hover:bg-agent-hover"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </form>
    </div>
  )
}
