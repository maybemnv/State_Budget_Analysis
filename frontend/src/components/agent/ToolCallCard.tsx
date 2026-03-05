"use client"

import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Terminal, CheckCircle2, CircleDashed } from "lucide-react"

interface ToolCallCardProps {
  tool: string
  args?: Record<string, unknown>
  result?: unknown
  status?: "pending" | "executing" | "completed" | "error"
  index?: number
  className?: string
}

/**
 * ToolCallCard — Displays a tool call with build-in animation.
 * Components appear sequentially: tool name → args → result
 */
export function ToolCallCard({
  tool,
  args,
  result,
  status = "executing",
  index = 0,
  className,
}: ToolCallCardProps) {
  const [showArgs, setShowArgs] = useState(false)
  const [showResult, setShowResult] = useState(false)

  useEffect(() => {
    // Animate in args after a short delay
    const argsTimer = setTimeout(() => setShowArgs(true), 200)
    return () => clearTimeout(argsTimer)
  }, [])

  useEffect(() => {
    // Show result when it arrives
    if (result !== undefined) {
      const resultTimer = setTimeout(() => setShowResult(true), 0)
      return () => clearTimeout(resultTimer)
    }
  }, [result])

  const statusConfig = {
    pending: {
      icon: CircleDashed,
      color: "text-text-muted",
      bg: "bg-border/50",
      label: "Pending",
    },
    executing: {
      icon: CircleDashed,
      color: "text-[#FF8555]",
      bg: "bg-[#FF8555]/10",
      label: "Executing",
    },
    completed: {
      icon: CheckCircle2,
      color: "text-success",
      bg: "bg-success/10",
      label: "Done",
    },
    error: {
      icon: CircleDashed,
      color: "text-error",
      bg: "bg-error/10",
      label: "Error",
    },
  }

  const config = statusConfig[status]
  const StatusIcon = config.icon

  const formatValue = (value: unknown): string => {
    if (value === null || value === undefined) return "null"
    if (typeof value === "object") {
      return JSON.stringify(value, null, 2).slice(0, 200)
    }
    return String(value)
  }

  const getResultSummary = (result: unknown): string => {
    if (typeof result === "object" && result !== null) {
      const obj = result as Record<string, unknown>
      // Extract key metrics based on tool type
      if ("r2" in obj) return `R² = ${(obj.r2 as number).toFixed(3)}`
      if ("accuracy" in obj) return `Accuracy = ${(obj.accuracy as number).toFixed(1)}`
      if ("anomaly_count" in obj) return `${obj.anomaly_count} anomalies detected`
      if ("shape" in obj) return `Shape: ${(obj.shape as number[]).join(" × ")}`
      if ("result" in obj && Array.isArray(obj.result)) return `${obj.result.length} rows`
      return JSON.stringify(result).slice(0, 100) + "..."
    }
    return formatValue(result)
  }

  return (
    <Card
      className={cn(
        "overflow-hidden border-border bg-surface/30 transition-all",
        "animate-scale-in",
        className
      )}
      style={{ animationDelay: `${index * 150}ms` }}
    >
      <CardHeader className="border-b border-border/50 bg-surface/50 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={cn("flex h-8 w-8 items-center justify-center rounded-md", config.bg)}>
              <Terminal className={cn("h-4 w-4", config.color)} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <code className="text-sm font-medium text-text-primary">{tool}</code>
                <Badge
                  variant="outline"
                  className={cn(
                    "h-5 text-xs",
                    status === "executing" && "border-[#FF8555]/50 text-[#FF8555]",
                    status === "completed" && "border-success/50 text-success",
                    status === "error" && "border-error/50 text-error"
                  )}
                >
                  <StatusIcon className="mr-1 h-3 w-3" />
                  {config.label}
                </Badge>
              </div>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="px-4 py-3">
        {/* Args section */}
        <div
          className={cn(
            "transition-all duration-500",
            showArgs ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"
          )}
        >
          <div className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Input
          </div>
          {args && (
            <pre className="overflow-x-auto rounded-md bg-elevated p-3 text-xs text-text-secondary">
              <code>{formatValue(args)}</code>
            </pre>
          )}
        </div>

        {/* Result section */}
        {result !== undefined && (
          <div
            className={cn(
              "mt-4 transition-all duration-500",
              showResult ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"
            )}
          >
            <div className="mb-2 text-xs font-medium uppercase tracking-wider text-text-muted">
              Result
            </div>
            <div
              className={cn(
                "rounded-md border px-3 py-2 text-sm",
                status === "error"
                  ? "border-error/30 bg-error/5 text-error"
                  : "border-success/30 bg-success/5 text-text-primary"
              )}
            >
              {getResultSummary(result)}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
