"use client"

import { useState, useMemo } from "react"
import { cn } from "@/lib/utils"
import { ChevronDown, ChevronRight, BrainCircuit } from "lucide-react"

interface ThinkingBlockProps {
  thoughts: string[]
  isLive: boolean
  defaultOpen?: boolean
}

/**
 * Strip <think>...</think> XML tags from thinking content.
 * Returns only the meaningful reasoning text, not the parser noise.
 */
function stripThinkTags(text: string): string {
  // Remove full <think>...</think> blocks
  let cleaned = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim()
  // Remove partial tags
  cleaned = cleaned.replace(/<\/?think>/gi, "").trim()
  // Remove empty lines
  cleaned = cleaned.split("\n").filter((l) => l.trim()).join("\n")
  return cleaned
}

/**
 * Extract a one-line summary from a thought step.
 * Takes the first meaningful sentence (non-empty, non-tag content).
 */
function summarizeThought(text: string): string {
  const cleaned = stripThinkTags(text)
  if (!cleaned) return ""
  // Get first sentence or first 100 chars
  const firstSentence = cleaned.split(/[.!?]/)[0]?.trim()
  if (firstSentence && firstSentence.length < 120) return firstSentence
  return cleaned.slice(0, 100) + (cleaned.length > 100 ? "…" : "")
}

export function ThinkingBlock({ thoughts, isLive, defaultOpen = false }: ThinkingBlockProps) {
  const [open, setOpen] = useState(defaultOpen)

  // Process thoughts: strip tags, compute summaries
  const processedThoughts = useMemo(
    () => thoughts.map(stripThinkTags).filter(Boolean),
    [thoughts]
  )

  const summaries = useMemo(
    () => processedThoughts.map(summarizeThought).filter(Boolean),
    [processedThoughts]
  )

  const stepCount = processedThoughts.length

  // Auto-collapse when thinking finishes
  useState(() => {
    if (!isLive && !defaultOpen) setOpen(false)
  })

  const label = isLive
    ? `Thinking…`
    : `Thought for ${stepCount} step${stepCount !== 1 ? "s" : ""}`

  if (processedThoughts.length === 0 && !isLive) return null

  return (
    <div className="animate-fade-in-up">
      {/* Toggle header — always visible */}
      <button
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "flex w-full items-center gap-2 rounded px-3 py-2 text-left transition-colors",
          isLive
            ? "bg-agent/5 hover:bg-agent/10"
            : "bg-elevated/40 hover:bg-elevated/70"
        )}
        aria-expanded={open}
        aria-label={`Toggle thinking steps (${stepCount} steps)`}
      >
        {/* Status indicator */}
        <span className="flex h-4 w-4 shrink-0 items-center justify-center">
          {isLive ? (
            <span className="flex h-2 w-2 items-center justify-center">
              <span className="absolute h-2 w-2 rounded-full bg-agent opacity-75 animate-ping" />
              <span className="relative h-1.5 w-1.5 rounded-full bg-agent" />
            </span>
          ) : (
            <BrainCircuit className="h-3.5 w-3.5 text-agent/50" />
          )}
        </span>

        {/* Label */}
        <span className={cn(
          "flex-1 text-xs font-medium",
          isLive ? "text-agent" : "text-text-muted"
        )}>
          {label}
        </span>

        {/* Chevron */}
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-text-muted" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-text-muted" />
        )}
      </button>

      {/* Expandable body */}
      {open && processedThoughts.length > 0 && (
        <div className="mt-1.5 overflow-hidden rounded border-l-2 border-agent/20 bg-elevated/40 pl-3 pr-3 py-2.5 animate-fade-in-up">
          {/* Summarized steps — clean, no raw tags */}
          <div className="space-y-2">
            {summaries.map((summary, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-agent/40" />
                <p className="text-[12px] leading-relaxed text-text-secondary">
                  {summary}
                </p>
              </div>
            ))}
          </div>

          {/* Live indicator */}
          {isLive && (
            <div className="mt-3 flex items-center gap-1.5 text-[11px] text-agent">
              <span className="h-1.5 w-1.5 rounded-full bg-agent animate-pulse" />
              Still thinking…
            </div>
          )}
        </div>
      )}
    </div>
  )
}
