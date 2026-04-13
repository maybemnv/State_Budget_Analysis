"use client"

import { useCallback, useMemo } from "react"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Maximize2, Minimize2, ImageDown, BarChart3, ScatterChart, Wifi, WifiOff } from "lucide-react"
import { useWorkspaceStore } from "@/lib/store"
import type { VegaLiteSpec } from "@/lib/types"

interface VizPanelProps {
  chartSpec?: VegaLiteSpec | null
}

const TABS = ["PCA", "Clusters", "Forecast"] as const

/**
 * VizPanel — visualization panel for the workspace.
 * - 3D view: React Three Fiber scene (for spatial/cluster data)
 * - 2D view: Renders Vega-Lite specs as JSON preview or chart placeholder
 * - No hardcoded demo data — only shows what the backend sends
 */
export function VizPanel({ chartSpec }: VizPanelProps) {
  // Zustand store — shared UI state
  const view = useWorkspaceStore((s) => s.vizPanelView)
  const setView = useWorkspaceStore((s) => s.setVizPanelView)
  const fullscreen = useWorkspaceStore((s) => s.vizPanelFullscreen)
  const setFullscreen = useWorkspaceStore((s) => s.setVizPanelFullscreen)
  const agentState = useWorkspaceStore((s) => s.agentState)
  const isActive = agentState === "thinking" || agentState === "executing"

  const isConnected = agentState !== "idle" || chartSpec !== null

  const handleToggleFullscreen = useCallback(() => {
    setFullscreen(!fullscreen)
  }, [fullscreen, setFullscreen])

  const handleExport = useCallback(() => {
    if (!chartSpec) return
    const blob = new Blob([JSON.stringify(chartSpec, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "chart-spec.json"
    a.click()
    URL.revokeObjectURL(url)
  }, [chartSpec])

  // Render 2D view — either chart spec preview or empty state
  const render2DView = useMemo(() => {
    if (chartSpec) {
      return (
        <div className="h-full p-4">
          <div className="rounded border border-border bg-surface p-4">
            <div className="mb-3 flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-primary" aria-hidden="true" />
              <p className="text-sm font-semibold text-text-primary">
                {typeof chartSpec.title === "string"
                  ? chartSpec.title
                  : chartSpec.title && typeof chartSpec.title === "object" && "text" in chartSpec.title
                    ? String(chartSpec.title.text)
                    : "Chart"}
              </p>
            </div>

            {/* Chart data preview */}
            {chartSpec.data?.values && chartSpec.data.values.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs" aria-label="Chart data preview">
                  <thead>
                    <tr className="border-b border-border">
                      {Object.keys(chartSpec.data.values[0]).map((key) => (
                        <th key={key} className="px-2 py-1.5 font-medium text-text-muted">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {chartSpec.data.values.slice(0, 10).map((row, i) => (
                      <tr key={i} className="border-b border-border/50">
                        {Object.values(row).map((val, j) => (
                          <td key={j} className="px-2 py-1.5 text-text-secondary">
                            {String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {chartSpec.data.values.length > 10 && (
                  <p className="mt-2 text-center text-[11px] text-text-muted">
                    Showing 10 of {chartSpec.data.values.length} rows
                  </p>
                )}
              </div>
            ) : (
              <div className="rounded border border-border bg-elevated p-3 font-mono text-[11px] text-text-muted">
                <pre className="overflow-x-auto">
                  {JSON.stringify(chartSpec, null, 2).slice(0, 500)}
                  {JSON.stringify(chartSpec, null, 2).length > 500 ? "…" : ""}
                </pre>
              </div>
            )}
          </div>
        </div>
      )
    }

    // No chart spec yet
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <BarChart3 className="mb-3 h-10 w-10 text-text-disabled" aria-hidden="true" />
        <p className="text-sm font-medium text-text-secondary">No chart yet</p>
        <p className="mt-1 text-xs text-text-muted">Ask the agent to create a visualization</p>
      </div>
    )
  }, [chartSpec])

  return (
    <div
      className={cn(
        "flex flex-col",
        fullscreen ? "fixed inset-0 z-50 bg-elevated" : "h-full"
      )}
      role="region"
      aria-label="Visualization panel"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 sm:px-4 sm:py-2.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-widest text-text-muted">VIZ</span>
          <span
            className={cn(
              "flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold",
              isConnected ? "bg-success/10 text-success" : "bg-elevated text-text-disabled"
            )}
            aria-label={isConnected ? "Agent active" : "Standby"}
          >
            {isConnected ? <Wifi className="h-2.5 w-2.5" aria-hidden="true" /> : <WifiOff className="h-2.5 w-2.5" aria-hidden="true" />}
            {isConnected ? (isActive ? "Active" : "Ready") : "Standby"}
          </span>
        </div>

        <div className="flex items-center gap-0.5" role="toolbar" aria-label="Visualization controls">
          {[
            { id: "3d" as const, title: "3D view", Icon: ScatterChart },
            { id: "2d" as const, title: "2D view", Icon: BarChart3 },
          ].map(({ id, title, Icon }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              title={title}
              aria-label={title}
              aria-pressed={view === id}
              className={cn(
                "flex h-7 w-7 items-center justify-center rounded transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
                view === id ? "bg-elevated text-text-primary" : "text-text-muted hover:text-text-secondary"
              )}
            >
              <Icon className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          ))}
          <button
            title="Export chart spec"
            onClick={handleExport}
            disabled={!chartSpec}
            className="flex h-7 w-7 items-center justify-center rounded text-text-muted transition-colors hover:text-text-secondary disabled:opacity-30 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            aria-label="Export chart specification as JSON"
          >
            <ImageDown className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
          <button
            title={fullscreen ? "Exit fullscreen" : "Fullscreen"}
            onClick={handleToggleFullscreen}
            className="flex h-7 w-7 items-center justify-center rounded text-text-muted transition-colors hover:text-text-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            aria-label={fullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {fullscreen ? <Minimize2 className="h-3.5 w-3.5" aria-hidden="true" /> : <Maximize2 className="h-3.5 w-3.5" aria-hidden="true" />}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 overflow-hidden border-t border-border">
        {view === "3d" ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-text-muted">
              3D view requires real data. Ask the agent to run PCA or clustering.
            </p>
          </div>
        ) : (
          <ScrollArea className="h-full">
            {render2DView}
          </ScrollArea>
        )}
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-t border-border px-2 py-1.5 sm:px-3 sm:py-2" role="tablist" aria-label="Visualization types">
        {TABS.map((tab) => (
          <button
            key={tab}
            role="tab"
            aria-selected={false}
            className={cn(
              "rounded px-3 py-1 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
              "text-text-muted hover:text-text-secondary"
            )}
          >
            {tab}
          </button>
        ))}
      </div>
    </div>
  )
}
