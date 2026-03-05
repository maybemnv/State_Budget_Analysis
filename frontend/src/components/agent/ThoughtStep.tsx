"use client"

import { useEffect, useState, useRef } from "react"
import { cn } from "@/lib/utils"

interface ThoughtStepProps {
  content: string
  index?: number
  className?: string
  /** Typing speed in characters per second (default: 10 = ~60 WPM) */
  typingSpeed?: number
}

/**
 * ThoughtStep — Displays agent reasoning with typewriter animation.
 * Text appears character-by-character to create a sense of "live" thinking.
 */
export function ThoughtStep({
  content,
  index = 0,
  className,
  typingSpeed = 10,
}: ThoughtStepProps) {
  const [displayedText, setDisplayedText] = useState("")
  const [isTyping, setIsTyping] = useState(true)
  const indexRef = useRef(0)

  useEffect(() => {
    // Reset state when content changes
    setDisplayedText("")
    setIsTyping(true)
    indexRef.current = 0

    const charDelay = 1000 / typingSpeed // ms per character

    const typeInterval = setInterval(() => {
      if (indexRef.current < content.length) {
        setDisplayedText(content.slice(0, indexRef.current + 1))
        indexRef.current++
      } else {
        setIsTyping(false)
        clearInterval(typeInterval)
      }
    }, charDelay)

    return () => clearInterval(typeInterval)
  }, [content, typingSpeed])

  return (
    <div
      className={cn(
        "group relative rounded-lg border border-border bg-surface/50 p-4 transition-all hover:border-border-hover",
        "animate-fade-in-up",
        className
      )}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      {/* Left accent bar */}
      <div className="absolute left-0 top-3 h-6 w-1 rounded-r bg-agent/60 transition-colors group-hover:bg-agent" />

      {/* Label */}
      <div className="mb-2 flex items-center gap-2 text-xs font-medium text-text-muted uppercase tracking-wider">
        <span className="text-agent">Thought</span>
        {isTyping && (
          <span className="flex h-2 w-2 items-center justify-center">
            <span className="h-1.5 w-1.5 animate-pulse-fast rounded-full bg-agent" />
          </span>
        )}
      </div>

      {/* Content with typewriter effect */}
      <div className="font-body text-sm leading-relaxed text-text-secondary">
        {displayedText}
        {isTyping && (
          <span className="inline-block h-4 w-px animate-cursor-blink bg-agent" />
        )}
      </div>
    </div>
  )
}
