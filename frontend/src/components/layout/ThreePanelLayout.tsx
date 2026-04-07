"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { Sidebar } from "@/components/layout/sidebar"
import { AgentChat } from "@/components/agent/AgentChat"
import { VizPanel } from "@/components/layout/VizPanel"
import type { AgentState, VegaLiteSpec } from "@/lib/types"

interface ThreePanelLayoutProps {
  sessionId: string
  sessionInfo?: {
    filename: string
    shape: [number, number]
    columns: string[]
    dtypes: Record<string, string>
  }
}

export function ThreePanelLayout({ sessionId, sessionInfo }: ThreePanelLayoutProps) {
  const [chartSpec, setChartSpec] = useState<VegaLiteSpec | null>(null)
  const [agentState, setAgentState] = useState<AgentState>("idle")
  const [timelineSteps, setTimelineSteps] = useState<{ label: string; timestamp: string }[]>([])

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      {/* Sidebar — 260px */}
      <div className="w-[260px] flex-shrink-0 bg-surface">
        <Sidebar sessionInfo={sessionInfo} />
      </div>

      {/* Agent Chat — flex grow */}
      <div className="flex min-w-0 flex-1 flex-col bg-background">
        <AgentChat
          sessionId={sessionId}
          onChartSpec={setChartSpec}
          onAgentStateChange={setAgentState}
          onTimelineStep={(step) => setTimelineSteps((p) => [...p.slice(-9), step])}
        />

        {/* Timeline bar */}
        <div className="border-t border-border bg-surface px-5 py-2">
          <div className="flex items-center gap-1 overflow-x-auto">
            {timelineSteps.length === 0 ? (
              <span className="text-[11px] text-text-disabled">
                Agent timeline — steps appear here as the agent works
              </span>
            ) : (
              timelineSteps.map((step, i) => (
                <div key={i} className="flex shrink-0 items-center">
                  <div className="flex flex-col items-center gap-0.5">
                    <div
                      className={cn(
                        "h-2 w-2 rounded-full",
                        i === timelineSteps.length - 1 && agentState !== "idle"
                          ? "bg-primary animate-pulse-fast"
                          : "bg-success"
                      )}
                    />
                    <span className="max-w-[72px] truncate text-[10px] text-text-muted">{step.label}</span>
                    <span className="font-mono text-[9px] tabular-nums text-text-disabled">{step.timestamp}</span>
                  </div>
                  {i < timelineSteps.length - 1 && (
                    <div className="mx-1.5 h-px w-5 bg-border" />
                  )}
                </div>
              ))
            )}
            {agentState === "idle" && timelineSteps.length > 0 && (
              <span className="ml-auto shrink-0 text-[11px] text-success">✓ Done</span>
            )}
          </div>
        </div>
      </div>

      {/* Viz Panel — 340px */}
      <div className="w-[340px] flex-shrink-0 bg-elevated">
        <VizPanel chartSpec={chartSpec} isConnected={agentState !== "idle"} />
      </div>
    </div>
  )
}
