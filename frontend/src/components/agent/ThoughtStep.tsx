"use client"

import { useEffect, useState, useRef } from "react"
import { cn } from "@/lib/utils"

interface ThoughtStepProps {
  content: string
  index?: number
  typingSpeed?: number
}

export function ThoughtStep({ content, index = 0, typingSpeed = 12 }: ThoughtStepProps) {
  const [displayed, setDisplayed] = useState("")
  const [typing, setTyping] = useState(true)
  const cursor = useRef(0)

  useEffect(() => {
    setDisplayed("")
    setTyping(true)
    cursor.current = 0

    const ms = 1000 / typingSpeed
    const t = setInterval(() => {
      if (cursor.current < content.length) {
        setDisplayed(content.slice(0, cursor.current + 1))
        cursor.current++
      } else {
        setTyping(false)
        clearInterval(t)
      }
    }, ms)

    return () => clearInterval(t)
  }, [content, typingSpeed])

  return (
    <div
      className="relative rounded border-l-2 border-agent bg-elevated px-4 py-3 animate-fade-in-up"
      style={{ animationDelay: `${index * 80}ms` }}
    >
      <div className="mb-1.5 flex items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-agent">Thought</span>
        {typing && <span className="h-1.5 w-1.5 rounded-full bg-agent animate-pulse-fast" />}
      </div>
      <p className="font-mono text-sm leading-relaxed text-text-secondary">
        {displayed}
        {typing && <span className="inline-block h-4 w-px animate-cursor-blink bg-agent align-middle ml-0.5" />}
      </p>
    </div>
  )
}
