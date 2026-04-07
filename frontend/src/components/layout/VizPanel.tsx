"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Scene3D } from "@/components/viz/Scene3D"
import { PCAScatter3D } from "@/components/viz/PCAScatter3D"
import { Maximize2, Minimize2, ImageDown, BarChart3, ScatterChart, Wifi, WifiOff } from "lucide-react"
import type { VegaLiteSpec } from "@/lib/types"

interface VizPanelProps {
  chartSpec?: VegaLiteSpec | null
  isConnected?: boolean
}

const DEMO_POINTS = Array.from({ length: 50 }, () => ({
  x: (Math.random() - 0.5) * 2,
  y: (Math.random() - 0.5) * 2,
  z: (Math.random() - 0.5) * 2,
  group: Math.floor(Math.random() * 3),
}))

const TABS = ["PCA", "Clusters", "Forecast"] as const

export function VizPanel({ chartSpec, isConnected = false }: VizPanelProps) {
  const [view, setView] = useState<"3d" | "2d">("3d")
  const [fullscreen, setFullscreen] = useState(false)
  const [activeTab, setActiveTab] = useState<(typeof TABS)[number]>("PCA")

  return (
    <div
      className={cn(
        "flex flex-col",
        fullscreen ? "fixed inset-0 z-50 bg-elevated" : "h-full"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-widest text-text-muted">VIZ</span>
          <span
            className={cn(
              "flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold",
              isConnected ? "bg-success/10 text-success" : "bg-elevated text-text-disabled"
            )}
          >
            {isConnected ? <Wifi className="h-2.5 w-2.5" /> : <WifiOff className="h-2.5 w-2.5" />}
            {isConnected ? "Live" : "Standby"}
          </span>
        </div>

        <div className="flex items-center gap-0.5">
          {[
            { icon: ScatterChart, id: "3d" as const, title: "3D" },
            { icon: BarChart3, id: "2d" as const, title: "2D" },
          ].map(({ icon: Icon, id, title }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              title={title}
              className={cn(
                "flex h-7 w-7 items-center justify-center rounded transition-colors",
                view === id ? "bg-elevated text-text-primary" : "text-text-muted hover:text-text-secondary"
              )}
            >
              <Icon className="h-3.5 w-3.5" />
            </button>
          ))}
          <button title="Export" className="flex h-7 w-7 items-center justify-center rounded text-text-muted transition-colors hover:text-text-secondary">
            <ImageDown className="h-3.5 w-3.5" />
          </button>
          <button
            title={fullscreen ? "Exit fullscreen" : "Fullscreen"}
            onClick={() => setFullscreen((f) => !f)}
            className="flex h-7 w-7 items-center justify-center rounded text-text-muted transition-colors hover:text-text-secondary"
          >
            {fullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 overflow-hidden border-t border-border">
        {view === "3d" ? (
          <Scene3D showControls={false}>
            <PCAScatter3D points={DEMO_POINTS} pointSize={0.12} />
          </Scene3D>
        ) : (
          <ScrollArea className="h-full p-4">
            {chartSpec ? (
              <div className="rounded border border-border bg-surface p-4">
                <p className="mb-1 text-sm font-semibold text-text-primary">
                  {chartSpec.title?.toString() ?? "Chart"}
                </p>
                <p className="text-xs text-text-muted">Chart spec received from agent</p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <BarChart3 className="mb-3 h-10 w-10 text-text-disabled" />
                <p className="text-sm font-medium text-text-secondary">No chart yet</p>
                <p className="mt-1 text-xs text-text-muted">Ask the agent to create a visualization</p>
              </div>
            )}
          </ScrollArea>
        )}
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-t border-border px-3 py-2">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              "rounded px-3 py-1 text-xs font-medium transition-colors",
              activeTab === tab
                ? "bg-elevated text-text-primary"
                : "text-text-muted hover:text-text-secondary"
            )}
          >
            {tab}
          </button>
        ))}
      </div>
    </div>
  )
}
