"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { Zap, X, ChevronRight, BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Insight {
  title: string
  body: string
}

interface AutoInsightModalProps {
  insights: Insight[]
  onDigDeeper: (insight: Insight) => void
  onShowVisualizations: () => void
  onDismiss: () => void
  className?: string
}

/**
 * AutoInsightModal — "Holy shit" moment: shows 3 non-obvious patterns
 * discovered automatically after dataset upload.
 */
export function AutoInsightModal({
  insights,
  onDigDeeper,
  onShowVisualizations,
  onDismiss,
  className,
}: AutoInsightModalProps) {
  const [expanded, setExpanded] = useState<number | null>(0)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <div
        className={cn(
          "w-full max-w-2xl overflow-hidden rounded-xl border border-border bg-surface shadow-2xl animate-scale-in",
          className
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#FF6B35]/10">
              <Zap className="h-5 w-5 text-[#FF6B35]" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-text-primary">Auto-Insight Mode</h2>
              <p className="text-xs text-text-muted">{insights.length} patterns found in your data</p>
            </div>
          </div>
          <button
            onClick={onDismiss}
            className="rounded-md p-1.5 text-text-muted transition-colors hover:bg-elevated hover:text-text-primary"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Insights */}
        <div className="divide-y divide-border">
          {insights.map((insight, i) => (
            <button
              key={i}
              onClick={() => setExpanded(expanded === i ? null : i)}
              className="w-full px-6 py-4 text-left transition-colors hover:bg-elevated/50"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#FF6B35]/10 text-xs font-bold text-[#FF6B35]">
                    {i + 1}
                  </span>
                  <div>
                    <p className="text-sm font-medium text-text-primary">{insight.title}</p>
                    {expanded === i && (
                      <p className="mt-2 text-sm leading-relaxed text-text-secondary animate-fade-in-up">
                        {insight.body}
                      </p>
                    )}
                  </div>
                </div>
                <ChevronRight
                  className={cn(
                    "mt-0.5 h-4 w-4 shrink-0 text-text-muted transition-transform",
                    expanded === i && "rotate-90"
                  )}
                />
              </div>
            </button>
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end gap-2 border-t border-border px-6 py-4">
          <Button
            variant="outline"
            size="sm"
            onClick={onDismiss}
            className="border-border bg-surface/50 text-text-secondary"
          >
            Dismiss
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onShowVisualizations}
            className="border-agent/40 text-agent hover:bg-agent/10"
          >
            <BarChart3 className="mr-1.5 h-3.5 w-3.5" />
            Visualizations
          </Button>
          {expanded !== null && (
            <Button
              size="sm"
              onClick={() => onDigDeeper(insights[expanded])}
              className="bg-[#FF6B35] text-white hover:bg-[#FF8555]"
            >
              Dig Deeper
              <ChevronRight className="ml-1 h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
