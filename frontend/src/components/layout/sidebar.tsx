"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  FileText,
  Table,
  Hash,
  Type,
  Calendar,
  ChevronRight,
  ChevronDown,
  TrendingUp,
  AlertCircle,
  Upload,
  Settings,
  History,
} from "lucide-react"

interface SidebarProps {
  sessionInfo?: {
    filename: string
    shape: [number, number]
    columns: string[]
    dtypes: Record<string, string>
  }
  className?: string
}

interface ColumnGroup {
  name: string
  icon: React.ReactNode
  columns: string[]
  color: string
}

/**
 * Sidebar — Left panel showing file info, column browser, and quick stats.
 */
export function Sidebar({ sessionInfo, className }: SidebarProps) {
  const [expandedSections, setExpandedSections] = useState({
    columns: true,
    stats: false,
    history: false,
  })

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }))
  }

  // Group columns by type
  const columnGroups: ColumnGroup[] = [
    {
      name: "Numeric",
      icon: <Hash className="h-3 w-3" />,
      columns:
        sessionInfo?.columns.filter((col) =>
          sessionInfo.dtypes[col]?.match(/int|float|number/i)
        ) || [],
      color: "text-[#00DCB4]",
    },
    {
      name: "Categorical",
      icon: <Type className="h-3 w-3" />,
      columns:
        sessionInfo?.columns.filter((col) =>
          sessionInfo.dtypes[col]?.match(/object|string|category/i)
        ) || [],
      color: "text-[#9D4EDD]",
    },
    {
      name: "Date/Time",
      icon: <Calendar className="h-3 w-3" />,
      columns:
        sessionInfo?.columns.filter((col) =>
          sessionInfo.dtypes[col]?.match(/date|time|datetime/i)
        ) || [],
      color: "text-[#FF6B35]",
    },
  ]

  return (
    <div className={cn("flex h-full flex-col border-r border-border bg-background", className)}>
      {/* Header */}
      <div className="border-b border-border px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface">
            <FileText className="h-5 w-5 text-[#FF6B35]" />
          </div>
          <div className="flex-1 overflow-hidden">
            <h3 className="truncate text-sm font-medium text-text-primary">
              {sessionInfo?.filename || "No file uploaded"}
            </h3>
            {sessionInfo && (
              <p className="text-xs text-text-muted">
                {sessionInfo.shape[0].toLocaleString()} rows ×{" "}
                {sessionInfo.shape[1].toLocaleString()} cols
              </p>
            )}
          </div>
        </div>
      </div>

      <ScrollArea className="flex-1">
        {/* Quick Stats */}
        <div className="border-b border-border">
          <button
            onClick={() => toggleSection("stats")}
            className="flex w-full items-center justify-between px-4 py-2 text-xs font-medium uppercase tracking-wider text-text-muted hover:bg-surface/50"
          >
            <span className="flex items-center gap-2">
              <TrendingUp className="h-3 w-3" />
              Quick Stats
            </span>
            {expandedSections.stats ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </button>

          {expandedSections.stats && sessionInfo && (
            <div className="space-y-2 px-4 pb-3">
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md bg-surface/50 p-2">
                  <div className="text-xs text-text-muted">Numeric</div>
                  <div className="text-lg font-semibold text-[#00DCB4]">
                    {columnGroups[0].columns.length}
                  </div>
                </div>
                <div className="rounded-md bg-surface/50 p-2">
                  <div className="text-xs text-text-muted">Categorical</div>
                  <div className="text-lg font-semibold text-[#9D4EDD]">
                    {columnGroups[1].columns.length}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 rounded-md bg-surface/50 p-2 text-xs text-text-muted">
                <AlertCircle className="h-3 w-3" />
                <span>Missing values: 0</span>
              </div>
            </div>
          )}
        </div>

        {/* Column Browser */}
        <div className="border-b border-border">
          <button
            onClick={() => toggleSection("columns")}
            className="flex w-full items-center justify-between px-4 py-2 text-xs font-medium uppercase tracking-wider text-text-muted hover:bg-surface/50"
          >
            <span className="flex items-center gap-2">
              <Table className="h-3 w-3" />
              Columns
            </span>
            {expandedSections.columns ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </button>

          {expandedSections.columns && (
            <div className="px-2 pb-2">
              {columnGroups.map((group) =>
                group.columns.length > 0 ? (
                  <div key={group.name} className="mb-3">
                    <div className="mb-1 flex items-center gap-2 px-2 text-xs font-medium text-text-muted">
                      <span className={group.color}>{group.icon}</span>
                      <span>{group.name}</span>
                      <Badge variant="outline" className="ml-auto h-5 text-xs">
                        {group.columns.length}
                      </Badge>
                    </div>
                    <div className="space-y-0.5">
                      {group.columns.map((col) => (
                        <button
                          key={col}
                          className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm text-text-secondary transition-colors hover:bg-surface hover:text-text-primary"
                        >
                          <span className="truncate">{col}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null
              )}
            </div>
          )}
        </div>

        {/* Session History */}
        <div>
          <button
            onClick={() => toggleSection("history")}
            className="flex w-full items-center justify-between px-4 py-2 text-xs font-medium uppercase tracking-wider text-text-muted hover:bg-surface/50"
          >
            <span className="flex items-center gap-2">
              <History className="h-3 w-3" />
              History
            </span>
            {expandedSections.history ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </button>

          {expandedSections.history && (
            <div className="px-2 pb-2">
              <div className="space-y-1">
                {["Q4 2024", "Q3 2024", "Q2 2024", "Q1 2024"].map((session, i) => (
                  <button
                    key={session}
                    className={cn(
                      "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-surface",
                      i === 0
                        ? "bg-surface text-text-primary"
                        : "text-text-secondary"
                    )}
                  >
                    <div
                      className={cn(
                        "h-2 w-2 rounded-full",
                        i === 0 ? "bg-[#FF6B35]" : "bg-border"
                      )}
                    />
                    <span>{session}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Footer actions */}
      <div className="border-t border-border p-3">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 border-border bg-surface/50 text-text-secondary hover:bg-surface hover:text-text-primary"
          >
            <Upload className="mr-2 h-4 w-4" />
            Upload
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-9 w-9 border-border bg-surface/50 p-0 text-text-secondary hover:bg-surface hover:text-text-primary"
          >
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
