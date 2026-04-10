"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { cn } from "@/lib/utils"
import { Zap, X, ChevronRight, BarChart3 } from "lucide-react"

interface Insight {
  title: string
  body: string
}

interface AutoInsightModalProps {
  insights: Insight[]
  onDigDeeper: (insight: Insight) => void
  onShowVisualizations: () => void
  onDismiss: () => void
}

export function AutoInsightModal({ insights, onDigDeeper, onShowVisualizations, onDismiss }: AutoInsightModalProps) {
  const [expanded, setExpanded] = useState<number | null>(0)
  const dialogRef = useRef<HTMLDivElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  // Focus the close button on mount
  useEffect(() => {
    closeButtonRef.current?.focus()

    // Trap focus inside the modal
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onDismiss()
        return
      }

      if (e.key === "Tab" && dialogRef.current) {
        const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        )
        const first = focusable[0]
        const last = focusable[focusable.length - 1]

        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault()
            last?.focus()
          }
        } else {
          if (document.activeElement === last) {
            e.preventDefault()
            first?.focus()
          }
        }
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [onDismiss])

  const handleExpand = useCallback((i: number) => {
    setExpanded((prev) => (prev === i ? null : i))
  }, [])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-text-primary/10 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="insight-modal-title"
      aria-describedby="insight-modal-desc"
    >
      <div
        ref={dialogRef}
        className="w-full max-w-xl overflow-hidden rounded-lg border border-border bg-surface shadow-xl animate-scale-in"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded bg-primary/10">
              <Zap className="h-4 w-4 text-primary" aria-hidden="true" />
            </div>
            <div>
              <h2 id="insight-modal-title" className="text-sm font-semibold text-text-primary">
                Auto-Insight Mode
              </h2>
              <p id="insight-modal-desc" className="text-[11px] text-text-muted">
                {insights.length} patterns found in your data
              </p>
            </div>
          </div>
          <button
            ref={closeButtonRef}
            onClick={onDismiss}
            className="rounded p-1.5 text-text-muted transition-colors hover:bg-elevated hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            aria-label="Close insights"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>

        {/* Insights list */}
        <div className="divide-y divide-border">
          {insights.map((insight, i) => (
            <button
              key={i}
              onClick={() => handleExpand(i)}
              className="flex w-full items-start gap-3 px-6 py-4 text-left transition-colors hover:bg-elevated/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              aria-expanded={expanded === i}
            >
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-[11px] font-bold text-primary">
                {i + 1}
              </span>
              <div className="flex-1">
                <p className="text-sm font-medium text-text-primary">{insight.title}</p>
                {expanded === i && (
                  <p className="mt-2 text-sm leading-relaxed text-text-secondary animate-fade-in-up">
                    {insight.body}
                  </p>
                )}
              </div>
              <ChevronRight
                className={cn(
                  "mt-0.5 h-4 w-4 shrink-0 text-text-muted transition-transform duration-200",
                  expanded === i && "rotate-90"
                )}
                aria-hidden="true"
              />
            </button>
          ))}
        </div>

        {/* Actions */}
        <div className="flex flex-wrap items-center justify-end gap-2 border-t border-border px-6 py-4">
          <button
            onClick={onDismiss}
            className="rounded px-4 py-2 text-sm text-text-muted transition-colors hover:text-text-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            Dismiss
          </button>
          <button
            onClick={onShowVisualizations}
            className="flex items-center gap-1.5 rounded border border-border px-4 py-2 text-sm font-medium text-text-secondary transition-colors hover:border-border-hover hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            <BarChart3 className="h-3.5 w-3.5" aria-hidden="true" />
            Visualizations
          </button>
          {expanded !== null && (
            <button
              onClick={() => onDigDeeper(insights[expanded])}
              className="flex items-center gap-1.5 rounded bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            >
              Dig Deeper
              <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
