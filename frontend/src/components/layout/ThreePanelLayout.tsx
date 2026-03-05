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
        />
        
        {/* Agent Timeline — Bottom scrubber */}
        <div className="border-t border-border bg-surface/30 px-4 py-2">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              {/* Timeline steps */}
              {["desc", "stats", "anomaly", "chart", "answer"].map((step, i, arr) => (
                <div key={step} className="flex items-center">
                  <button
                    className={cn(
                      "flex h-6 w-6 items-center justify-center rounded-full text-xs transition-colors",
                      i <= 2
                        ? "bg-[#FF6B35] text-white"
                        : "bg-border text-text-muted hover:bg-surface"
                    )}
                  >
                    {i + 1}
                  </button>
                  {i < arr.length - 1 && (
                    <div
                      className={cn(
                        "mx-1 h-0.5 w-8 transition-colors",
                        i < 2 ? "bg-[#FF6B35]" : "bg-border"
                      )}
                    />
                  )}
                </div>
              ))}
            </div>
            <div className="ml-auto text-xs text-text-muted">
              {agentState === "thinking" && "Agent is thinking..."}
              {agentState === "executing" && "Running analysis..."}
              {agentState === "done" && "Analysis complete"}
              {agentState === "idle" && "Ready"}
            </div>
          </div>
        </div>
      </div>

      {/* Right Viz Panel — 360px fixed */}
      <div className="w-[360px] flex-shrink-0">
        <VizPanel chartSpec={chartSpec} />
      </div>
    </div>
  )
}
