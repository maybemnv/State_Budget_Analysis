"use client"

import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import { Terminal, CheckCircle2, CircleDashed } from "lucide-react"

interface ToolCallCardProps {
  tool: string
  args?: Record<string, unknown>
  result?: unknown
  status?: "pending" | "executing" | "completed" | "error"
  index?: number
}

const STATUS_MAP = {
  pending:   { icon: CircleDashed, color: "text-text-muted",    label: "Pending"   },
  executing: { icon: CircleDashed, color: "text-primary",       label: "Executing" },
  completed: { icon: CheckCircle2, color: "text-success",       label: "Done"      },
  error:     { icon: CircleDashed, color: "text-error",         label: "Error"     },
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "null"
  if (typeof v === "object") return JSON.stringify(v, null, 2).slice(0, 300)
  return String(v)
}

function resultSummary(r: unknown): string {
  if (typeof r === "object" && r !== null) {
    const o = r as Record<string, unknown>
    if ("r2" in o)            return `R² = ${(o.r2 as number).toFixed(3)}`
    if ("accuracy" in o)      return `Accuracy = ${(o.accuracy as number).toFixed(2)}`
    if ("anomaly_count" in o) return `${o.anomaly_count} anomalies found`
    if ("shape" in o)         return `Shape: ${(o.shape as number[]).join(" × ")}`
    if ("result" in o && Array.isArray(o.result)) return `${o.result.length} rows returned`
    return JSON.stringify(r).slice(0, 120) + "…"
  }
  return formatValue(r)
}

export function ToolCallCard({ tool, args, result, status = "executing", index = 0 }: ToolCallCardProps) {
  const [showArgs, setShowArgs] = useState(false)
  const [showResult, setShowResult] = useState(false)

  useEffect(() => { const t = setTimeout(() => setShowArgs(true), 180); return () => clearTimeout(t) }, [])
  useEffect(() => {
    if (result !== undefined) { const t = setTimeout(() => setShowResult(true), 0); return () => clearTimeout(t) }
  }, [result])

  const cfg = STATUS_MAP[status]
  const Icon = cfg.icon

  return (
    <div
      className="overflow-hidden rounded border border-border bg-elevated animate-scale-in"
      style={{ animationDelay: `${index * 100}ms` }}
    >
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-border px-4 py-2.5">
        <Terminal className={cn("h-4 w-4 shrink-0", cfg.color)} />
        <code className="flex-1 text-sm font-semibold text-text-primary">{tool}</code>
        <span className={cn("flex items-center gap-1 text-[10px] font-semibold uppercase tracking-widest", cfg.color)}>
          <Icon className="h-3 w-3" />
          {cfg.label}
        </span>
      </div>

      {/* Args */}
      <div
        className={cn("px-4 py-3 transition-all duration-400", showArgs ? "opacity-100" : "opacity-0 translate-y-1")}
      >
        <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-text-muted">Input</p>
        {args && (
          <pre className="overflow-x-auto rounded bg-surface px-3 py-2 font-mono text-[11px] text-text-secondary">
            {formatValue(args)}
          </pre>
        )}
      </div>

      {/* Result */}
      {result !== undefined && (
        <div
          className={cn("border-t border-border px-4 py-3 transition-all duration-400", showResult ? "opacity-100" : "opacity-0 translate-y-1")}
        >
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-text-muted">Result</p>
          <div className={cn(
            "rounded px-3 py-2 text-sm",
            status === "error"
              ? "border border-error/20 bg-error/5 text-error"
              : "border border-success/20 bg-success/5 text-text-primary"
          )}>
            {resultSummary(result)}
          </div>
        </div>
      )}
    </div>
  )
}
