"use client"

import { cn } from "@/lib/utils"
import { Sidebar } from "@/components/layout/sidebar"
import { AgentChat } from "@/components/agent/AgentChat"
import { VizPanel } from "@/components/layout/VizPanel"
import { useState } from "react"
import type { AgentState, VegaLiteSpec } from "@/lib/types"

interface ThreePanelLayoutProps {
  sessionId: string
  sessionInfo?: {
    filename: string
    shape: [number, number]
    columns: string[]
    dtypes: Record<string, string>
  }
  className?: string
}

/**
 * ThreePanelLayout — Main workspace layout with three panels:
 * - Left: Sidebar (file info, columns, stats)
 * - Center: Agent chat (thoughts, tool calls, answers)
 * - Right: Visualization canvas (3D/2D charts)
 */
export function ThreePanelLayout({ sessionId, sessionInfo, className }: ThreePanelLayoutProps) {
  const [chartSpec, setChartSpec] = useState<VegaLiteSpec | null>(null)
  const [agentState, setAgentState] = useState<AgentState>("idle")
  const [timelineSteps, setTimelineSteps] = useState<{ label: string; timestamp: string }[]>([])

  return (
    <div className={cn("flex h-screen w-screen overflow-hidden bg-background", className)}>
      {/* Left Sidebar — 280px fixed */}
      <div className="w-[280px] flex-shrink-0 border-r border-border">
        <Sidebar sessionInfo={sessionInfo} />
      </div>

      {/* Center — Agent Chat — Flexible width */}
      <div className="flex min-w-0 flex-1 flex-col border-r border-border">
        <AgentChat
          sessionId={sessionId}
          onChartSpec={(spec) => setChartSpec(spec)}
          onAgentStateChange={setAgentState}
          onTimelineStep={(step) =>
            setTimelineSteps((prev) => [...prev.slice(-8), step])
          }
        />

        {/* Agent Timeline — dynamic, click to jump */}
        <div className="border-t border-border bg-surface/30 px-4 py-2">
          <div className="flex items-center gap-1 overflow-x-auto">
            {timelineSteps.length === 0 ? (
              <span className="text-xs text-text-disabled">Agent timeline will appear here</span>
            ) : (
              timelineSteps.map((step, i) => (
                <div key={i} className="flex shrink-0 items-center">
                  <div className="flex flex-col items-center gap-0.5">
                    <div
                      className={cn(
                        "h-2.5 w-2.5 rounded-full transition-colors",
                        i === timelineSteps.length - 1
                          ? agentState !== "idle"
                            ? "bg-[#FF6B35] animate-pulse"
                            : "bg-success"
                          : "bg-success"
                      )}
                    />
                    <span className="max-w-[64px] truncate text-[9px] text-text-muted">{step.label}</span>
                    <span className="text-[8px] tabular-nums text-text-disabled">{step.timestamp}</span>
                  </div>
                  {i < timelineSteps.length - 1 && (
                    <div className="mx-1 h-px w-4 bg-border" />
                  )}
                </div>
              ))
            )}
            {agentState === "idle" && timelineSteps.length > 0 && (
              <span className="ml-auto shrink-0 text-xs text-success">✓ Done</span>
            )}
          </div>
        </div>
      </div>

      {/* Right Viz Panel — 360px fixed */}
      <div className="w-[360px] flex-shrink-0">
        <VizPanel chartSpec={chartSpec} isConnected={agentState !== "idle"} />
      </div>
    </div>
  )
}
