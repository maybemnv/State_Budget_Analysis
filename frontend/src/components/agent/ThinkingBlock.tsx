"use client"

import { useState, useEffect, useRef } from "react"
import { cn } from "@/lib/utils"
import { ChevronDown, BrainCircuit } from "lucide-react"

interface ThinkingBlockProps {
  thoughts: string[]
  isLive: boolean      // true = agent still thinking
  defaultOpen?: boolean
}

export function ThinkingBlock({ thoughts, isLive, defaultOpen = true }: ThinkingBlockProps) {
  const [open, setOpen] = useState(defaultOpen)
  const [displayed, setDisplayed] = useState("")
  const latestThought = thoughts[thoughts.length - 1] ?? ""
  const cursorRef = useRef(0)

  // Typewriter on the latest thought only
  useEffect(() => {
    setDisplayed("")
    cursorRef.current = 0
    const interval = setInterval(() => {
      if (cursorRef.current < latestThought.length) {
        setDisplayed(latestThought.slice(0, cursorRef.current + 1))
        cursorRef.current++
      } else {
        clearInterval(interval)
      }
    }, 1000 / 14)
    return () => clearInterval(interval)
  }, [latestThought])

  // Auto-collapse when thinking finishes
  useEffect(() => {
    if (!isLive) setOpen(false)
  }, [isLive])

  const label = isLive
    ? `Thinking...`
    : `Thought for ${thoughts.length} step${thoughts.length !== 1 ? "s" : ""}`

  return (
    <div className="animate-fade-in-up">
      {/* Toggle header */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 text-[11px] text-text-muted transition-colors hover:text-text-secondary"
      >
        {isLive && (
          <span className="flex h-2 w-2 items-center justify-center">
            <span className="absolute h-2 w-2 rounded-full bg-agent opacity-75 animate-ping" />
            <span className="h-1.5 w-1.5 rounded-full bg-agent" />
          </span>
        )}
        {!isLive && <BrainCircuit className="h-3.5 w-3.5 text-agent/60" />}
        <span className={cn(isLive ? "text-agent" : "text-text-muted")}>{label}</span>
        <ChevronDown
          className={cn("h-3 w-3 transition-transform duration-200", open && "rotate-180")}
        />
      </button>

      {/* Expandable body */}
      {open && (
        <div className="mt-2 overflow-hidden rounded border-l-2 border-agent/30 bg-elevated/60 pl-4 pr-3 py-3 animate-fade-in-up">
          {/* Past thoughts — compact, faded */}
          {thoughts.slice(0, -1).map((t, i) => (
            <p key={i} className="mb-2 font-mono text-[11px] leading-relaxed text-text-disabled line-clamp-2">
              {t}
            </p>
          ))}

          {/* Latest thought — streaming with cursor */}
          {latestThought && (
            <p className="font-mono text-[12px] leading-relaxed text-text-secondary">
              {displayed}
              {isLive && (
                <span className="inline-block h-3.5 w-px animate-cursor-blink bg-agent align-middle ml-px" />
              )}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
