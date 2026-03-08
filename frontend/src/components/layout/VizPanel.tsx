"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Scene3D } from "@/components/viz/Scene3D"
import { PCAScatter3D } from "@/components/viz/PCAScatter3D"
import { Maximize2, ImageDown, BarChart3, ScatterChart, Minimize2, Wifi, WifiOff } from "lucide-react"
import { BackendStatusIndicator } from "@/components/ui/BackendStatusIndicator"
import type { VegaLiteSpec } from "@/lib/types"

interface VizPanelProps {
  chartSpec?: VegaLiteSpec | null
  isConnected?: boolean
  className?: string
}

/**
 * VizPanel — Right panel showing visualizations.
 * Supports 3D scenes (PCA, clusters) and 2D charts (Recharts).
 */
export function VizPanel({ chartSpec, isConnected = false, className }: VizPanelProps) {
  const [activeTab, setActiveTab] = useState<"3d" | "2d">("3d")
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Demo data for PCA visualization (replace with real data from backend)
  const demoPoints = Array.from({ length: 50 }, (_, i) => ({
    x: (Math.random() - 0.5) * 2,
    y: (Math.random() - 0.5) * 2,
    z: (Math.random() - 0.5) * 2,
    group: Math.floor(Math.random() * 3),
  }))

  return (
    <div
      className={cn(
        "flex flex-col border-l border-border bg-background transition-all",
        isFullscreen ? "fixed inset-0 z-50" : "h-full",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-medium text-text-primary">VIZ</h2>
          <div className={cn(
            "flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium",
            isConnected
              ? "bg-success/10 text-success"
              : "bg-border text-text-muted"
          )}>
            {isConnected ? <Wifi className="h-2.5 w-2.5" /> : <WifiOff className="h-2.5 w-2.5" />}
            {isConnected ? "Active" : "Standby"}
          </div>
        </div>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className={cn("h-7 w-7", activeTab === "3d" ? "bg-surface text-text-primary" : "text-text-muted")}
            onClick={() => setActiveTab("3d")}
            title="3D View"
          >
            <ScatterChart className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={cn("h-7 w-7", activeTab === "2d" ? "bg-surface text-text-primary" : "text-text-muted")}
            onClick={() => setActiveTab("2d")}
            title="2D Charts"
          >
            <BarChart3 className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="icon" className="h-7 w-7 text-text-muted" title="Export">
            <ImageDown className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-text-muted"
            title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            onClick={() => setIsFullscreen((f) => !f)}
          >
            {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "3d" ? (
          <Scene3D showControls={false}>
            <PCAScatter3D points={demoPoints} pointSize={0.12} />
          </Scene3D>
        ) : (
          <ScrollArea className="h-full p-4">
            {chartSpec ? (
              <div className="space-y-4">
                <div className="rounded-lg border border-border bg-surface/50 p-4">
                  <h3 className="mb-2 text-sm font-medium text-text-primary">
                    {chartSpec.title?.toString() || "Chart"}
                  </h3>
                  <div className="text-xs text-text-muted">
                    Chart spec received from agent
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <BarChart3 className="mb-4 h-12 w-12 text-agent/50" />
                <h3 className="mb-2 text-lg font-medium text-text-primary">No chart yet</h3>
                <p className="text-sm text-text-muted">
                  Ask the agent to create a visualization
                </p>
              </div>
            )}
          </ScrollArea>
        )}
      </div>

      {/* Chart history tabs */}
      <div className="border-t border-border px-2 py-2">
        <div className="flex gap-1 overflow-x-auto">
          {["PCA", "Clusters", "Forecast"].map((tab, i) => (
            <button
              key={tab}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs whitespace-nowrap transition-colors",
                i === 0
                  ? "bg-surface text-text-primary"
                  : "text-text-muted hover:bg-surface/50 hover:text-text-primary"
              )}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
