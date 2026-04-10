"use client"

import { useState, useCallback } from "react"
import { cn } from "@/lib/utils"
import { Sidebar } from "@/components/layout/sidebar"
import { AgentChat } from "@/components/agent/AgentChat"
import { VizPanel } from "@/components/layout/VizPanel"
import { useWorkspaceStore } from "@/lib/store"
import type { VegaLiteSpec } from "@/lib/types"
import { PanelLeftClose, PanelLeftOpen, X } from "lucide-react"

interface ThreePanelLayoutProps {
  sessionId: string
  sessionInfo?: {
    filename: string
    shape: [number, number]
    columns: string[]
    dtypes: Record<string, string>
    missing_values?: number
  }
}

/**
 * ThreePanelLayout — responsive workspace layout.
 * - Desktop (>1024px): 3-column layout with sidebar, chat, viz panel
 * - Tablet (768-1024px): 2-column, sidebar collapses, viz panel overlays
 * - Mobile (<768px): single column, both panels overlay
 */
export function ThreePanelLayout({ sessionId, sessionInfo }: ThreePanelLayoutProps) {
  const setChartSpec = useWorkspaceStore((s) => s.setChartSpec)
  const addTimelineStep = useWorkspaceStore((s) => s.addTimelineStep)
  const timelineSteps = useWorkspaceStore((s) => s.timelineSteps)
  const agentState = useWorkspaceStore((s) => s.agentState)
  const sidebarOpen = useWorkspaceStore((s) => s.sidebarOpen)
  const setSidebarOpen = useWorkspaceStore((s) => s.setSidebarOpen)
  const vizPanelFullscreen = useWorkspaceStore((s) => s.vizPanelFullscreen)

  // Set session info on mount
  useState(() => {
    if (sessionInfo) {
      useWorkspaceStore.getState().setSessionInfo({
        filename: sessionInfo.filename,
        shape: sessionInfo.shape,
        columns: sessionInfo.columns,
        dtypes: sessionInfo.dtypes,
        missing_values: sessionInfo.missing_values,
      })
    }
  })

  const handleChartSpec = useCallback((spec: VegaLiteSpec) => {
    setChartSpec(spec)
  }, [setChartSpec])

  const handleTimelineStep = useCallback((step: { label: string; timestamp: string }) => {
    addTimelineStep(step)
  }, [addTimelineStep])

  return (
    <div className="relative flex h-screen w-screen overflow-hidden bg-background">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-background/60 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar — responsive width */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex h-full w-[280px] flex-shrink-0 flex-col bg-surface transition-transform duration-200 ease-in-out lg:relative lg:translate-x-0 lg:w-[260px]",
          sidebarOpen ? "translate-x-0" : "-translate-x-full lg:w-0 lg:overflow-hidden"
        )}
        aria-label="Session sidebar"
      >
        <Sidebar />
      </aside>

      {/* Toggle sidebar button — visible on all sizes */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className={cn(
          "absolute top-3 z-50 flex h-8 w-8 items-center justify-center rounded border border-border bg-surface text-text-muted transition-colors hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
          sidebarOpen ? "left-[268px] lg:left-auto" : "left-3"
        )}
        aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
      >
        {sidebarOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
      </button>

      {/* Agent Chat — flex grow */}
      <main className="flex min-w-0 flex-1 flex-col bg-background" role="main">
        <AgentChat
          sessionId={sessionId}
          onChartSpec={handleChartSpec}
          onTimelineStep={handleTimelineStep}
        />

        {/* Timeline bar */}
        <div className="border-t border-border bg-surface px-4 py-2">
          <div className="flex items-center gap-1 overflow-x-auto" role="list" aria-label="Agent timeline">
            {timelineSteps.length === 0 ? (
              <span className="text-[11px] text-text-disabled">
                Agent timeline — steps appear here as the agent works
              </span>
            ) : (
              timelineSteps.map((step, i) => (
                <div key={`${step.label}-${i}`} className="flex shrink-0 items-center" role="listitem">
                  <div className="flex flex-col items-center gap-0.5">
                    <div
                      className={cn(
                        "h-2 w-2 rounded-full",
                        i === timelineSteps.length - 1 && agentState !== "idle"
                          ? "bg-primary animate-pulse-fast"
                          : "bg-success"
                      )}
                      aria-hidden="true"
                    />
                    <span className="max-w-[72px] truncate text-[10px] text-text-muted" title={step.label}>
                      {step.label}
                    </span>
                    <span className="font-mono text-[9px] tabular-nums text-text-disabled">
                      {step.timestamp}
                    </span>
                  </div>
                  {i < timelineSteps.length - 1 && (
                    <div className="mx-1.5 h-px w-5 bg-border" aria-hidden="true" />
                  )}
                </div>
              ))
            )}
            {agentState === "idle" && timelineSteps.length > 0 && (
              <span className="ml-auto shrink-0 text-[11px] text-success" aria-label="Analysis complete">
                ✓ Done
              </span>
            )}
          </div>
        </div>
      </main>

      {/* Viz Panel — responsive: overlay on mobile, fixed on desktop */}
      {vizPanelFullscreen ? (
        <div className="fixed inset-0 z-50 bg-elevated">
          <button
            onClick={() => useWorkspaceStore.getState().setVizPanelFullscreen(false)}
            className="absolute right-4 top-4 z-10 flex h-8 w-8 items-center justify-center rounded bg-surface/90 text-text-muted transition-colors hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            aria-label="Exit fullscreen"
          >
            <X className="h-4 w-4" />
          </button>
          <VizPanel />
        </div>
      ) : (
        <aside
          className="hidden h-full flex-shrink-0 bg-elevated xl:block xl:w-[380px]"
          aria-label="Visualization panel"
        >
          <VizPanel />
        </aside>
      )}
    </div>
  )
}
