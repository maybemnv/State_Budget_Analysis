"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { Terminal, CheckCircle2, CircleDashed, ChevronDown, ChevronRight, Code2 } from "lucide-react"

interface ToolCallCardProps {
  tool: string
  args?: Record<string, unknown>
  result?: unknown
  status?: "pending" | "executing" | "completed" | "error"
  index?: number
}

const STATUS_MAP = {
  pending:   { icon: CircleDashed, color: "text-text-muted",  label: "Pending"   },
  executing: { icon: CircleDashed, color: "text-primary",     label: "Running"   },
  completed: { icon: CheckCircle2, color: "text-success",     label: "Done"      },
  error:     { icon: CircleDashed, color: "text-error",       label: "Error"     },
}

/**
 * Format a value concisely. Truncates long strings and objects.
 */
function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "null"
  if (typeof v === "boolean") return v ? "true" : "false"
  if (typeof v === "number") return String(v)
  if (typeof v === "string") {
    if (v.length > 200) return v.slice(0, 200) + "…"
    return v
  }
  if (Array.isArray(v)) return `[${v.length} item${v.length !== 1 ? "s" : ""}]`
  if (typeof v === "object") {
    const keys = Object.keys(v)
    return `{${keys.length} key${keys.length !== 1 ? "s" : ""}}`
  }
  return String(v)
}

/**
 * Create a human-readable summary of a tool call result.
 * Handles common patterns: shape, stats, row counts, error messages.
 */
function resultSummary(r: unknown): { text: string; isError: boolean } {
  if (r === null || r === undefined) return { text: "No result", isError: false }

  // Error case
  if (typeof r === "object" && r !== null) {
    const o = r as Record<string, unknown>
    if ("error" in o && o.error) return { text: String(o.error), isError: true }
    if ("message" in o && typeof o.message === "string" && o.message.toLowerCase().includes("error")) {
      return { text: o.message, isError: true }
    }
  }

  if (typeof r === "string") {
    // Check if it's an error message
    if (r.toLowerCase().includes("error") || r.toLowerCase().includes("exception")) {
      return { text: r.slice(0, 200), isError: true }
    }
    return { text: r.slice(0, 200) + (r.length > 200 ? "…" : ""), isError: false }
  }

  if (typeof r === "object" && r !== null) {
    const o = r as Record<string, unknown>

    // Statistical results
    if ("r2" in o) return { text: `R² = ${(o.r2 as number).toFixed(3)}`, isError: false }
    if ("accuracy" in o) return { text: `Accuracy = ${((o.accuracy as number) * 100).toFixed(1)}%`, isError: false }
    if ("anomaly_count" in o) return { text: `${o.anomaly_count} anomalies found`, isError: false }
    if ("shape" in o && Array.isArray(o.shape)) return { text: `Shape: ${o.shape.join(" × ")}`, isError: false }

    // Data results
    if ("result" in o && Array.isArray(o.result)) {
      const rows = o.result.length
      if (rows === 0) return { text: "No results", isError: false }
      return { text: `${rows} row${rows !== 1 ? "s" : ""} returned`, isError: false }
    }
    if ("rows" in o) return { text: `${o.rows} rows`, isError: false }
    if ("count" in o) return { text: `${o.count} items`, isError: false }

    // Generic object — show key count
    const keys = Object.keys(o)
    if (keys.length <= 3) {
      const preview = keys.map((k) => `${k}: ${formatValue(o[k])}`).join(", ")
      return { text: preview, isError: false }
    }
    return { text: `${keys.length} fields returned`, isError: false }
  }

  return { text: formatValue(r), isError: false }
}

export function ToolCallCard({ tool, args, result, status = "executing", index = 0 }: ToolCallCardProps) {
  const [showDetails, setShowDetails] = useState(false)

  const cfg = STATUS_MAP[status]
  const Icon = cfg.icon
  const summary = result !== undefined ? resultSummary(result) : null

  // Clean tool name for display
  const displayName = tool.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())

  return (
    <div
      className="overflow-hidden rounded border border-border bg-surface animate-scale-in"
      style={{ animationDelay: `${index * 80}ms` }}
    >
      {/* Header — always visible */}
      <div className="flex items-center gap-2.5 px-3 py-2">
        <Terminal className={cn("h-3.5 w-3.5 shrink-0", cfg.color)} />
        <code className="flex-1 text-xs font-medium text-text-primary">{displayName}</code>
        <span className={cn("flex items-center gap-1 text-[10px] font-medium", cfg.color)}>
          <Icon className="h-3 w-3" />
          {cfg.label}
        </span>
        {/* Expand toggle */}
        <button
          onClick={() => setShowDetails((d) => !d)}
          className="flex h-5 w-5 items-center justify-center rounded text-text-muted hover:text-text-primary transition-colors"
          aria-label={showDetails ? "Hide details" : "Show details"}
          aria-expanded={showDetails}
        >
          {showDetails ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
        </button>
      </div>

      {/* Expandable details */}
      {showDetails && (
        <div className="border-t border-border animate-fade-in-up">
          {/* Args */}
          {args && Object.keys(args).length > 0 && (
            <div className="px-3 py-2">
              <p className="mb-1 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
                <Code2 className="h-3 w-3" />
                Input
              </p>
              <pre className="overflow-x-auto rounded bg-elevated px-2.5 py-2 font-mono text-[11px] text-text-secondary max-h-32">
                {JSON.stringify(args, null, 2).slice(0, 500)}
              </pre>
            </div>
          )}

          {/* Result */}
          {result !== undefined && (
            <div className="border-t border-border px-3 py-2">
              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
                Output
              </p>
              <div className={cn(
                "rounded px-2.5 py-2 text-xs font-mono max-h-40 overflow-y-auto",
                summary?.isError
                  ? "bg-error/5 text-error"
                  : "bg-elevated text-text-secondary"
              )}>
                {summary?.text ?? "No output"}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
