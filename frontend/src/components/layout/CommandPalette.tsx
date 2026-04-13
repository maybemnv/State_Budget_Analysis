"use client"

import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Search,
  X,
  Command,
  Sparkles,
  BarChart3,
  TrendingUp,
  PieChart,
  Filter,
  Download,
  Trash2,
  Settings,
  ChevronRight,
} from "lucide-react"

interface CommandPaletteProps {
  isOpen: boolean
  onClose: () => void
  onCommand?: (command: string) => void
}

interface CommandItem {
  id: string
  label: string
  icon: React.ReactNode
  category: "suggested" | "tools" | "session" | "navigation"
  shortcut?: string
}

/**
 * CommandPalette — Cmd+K command palette for quick actions.
 * Context-aware suggestions based on data type.
 */
export function CommandPalette({ isOpen, onClose, onCommand }: CommandPaletteProps) {
  const [query, setQuery] = useState("")
  const [selectedIndex, setSelectedIndex] = useState(0)

  // Sample commands — in real app, these would be context-aware
  const commands: CommandItem[] = [
    {
      id: "anomalies",
      label: "Find anomalies in spending",
      icon: <Sparkles className="h-4 w-4 text-[#FF6B35]" />,
      category: "suggested",
    },
    {
      id: "correlation",
      label: "Show correlation matrix",
      icon: <BarChart3 className="h-4 w-4 text-[#00DCB4]" />,
      category: "suggested",
    },
    {
      id: "forecast",
      label: "Forecast next fiscal year",
      icon: <TrendingUp className="h-4 w-4 text-[#9D4EDD]" />,
      category: "suggested",
    },
    {
      id: "cluster",
      label: "Cluster agencies by spending",
      icon: <PieChart className="h-4 w-4 text-[#F59E0B]" />,
      category: "suggested",
    },
    {
      id: "trend",
      label: "Detect trends over time",
      icon: <TrendingUp className="h-4 w-4 text-[#FF6B35]" />,
      category: "suggested",
    },
    {
      id: "run_pca",
      label: "run_pca — Reduce dimensions",
      icon: <Filter className="h-4 w-4 text-text-muted" />,
      category: "tools",
    },
    {
      id: "run_kmeans",
      label: "run_kmeans — Cluster data",
      icon: <Filter className="h-4 w-4 text-text-muted" />,
      category: "tools",
    },
    {
      id: "run_regression",
      label: "run_regression — Predict numeric",
      icon: <Filter className="h-4 w-4 text-text-muted" />,
      category: "tools",
    },
    {
      id: "run_forecast",
      label: "run_forecast — Time series",
      icon: <Filter className="h-4 w-4 text-text-muted" />,
      category: "tools",
    },
    {
      id: "upload",
      label: "Upload new dataset",
      icon: <Download className="h-4 w-4 text-text-muted" />,
      category: "session",
    },
    {
      id: "clear",
      label: "Clear current session",
      icon: <Trash2 className="h-4 w-4 text-error" />,
      category: "session",
    },
    {
      id: "settings",
      label: "Settings",
      icon: <Settings className="h-4 w-4 text-text-muted" />,
      category: "navigation",
    },
  ]

  const filteredCommands = commands.filter((cmd) =>
    cmd.label.toLowerCase().includes(query.toLowerCase())
  )

  const groupedCommands = filteredCommands.reduce(
    (acc, cmd) => {
      if (!acc[cmd.category]) acc[cmd.category] = []
      acc[cmd.category].push(cmd)
      return acc
    },
    {} as Record<string, CommandItem[]>
  )

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return

      if (e.key === "Escape") {
        onClose()
        return
      }

      if (e.key === "ArrowDown") {
        e.preventDefault()
        setSelectedIndex((prev) => (prev + 1) % filteredCommands.length)
      }

      if (e.key === "ArrowUp") {
        e.preventDefault()
        setSelectedIndex((prev) => (prev - 1 + filteredCommands.length) % filteredCommands.length)
      }

      if (e.key === "Enter") {
        e.preventDefault()
        const selected = filteredCommands[selectedIndex]
        if (selected) {
          onCommand?.(selected.id)
          onClose()
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [isOpen, onClose, filteredCommands, selectedIndex, onCommand])

  // Cmd+K to open
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault()
        if (!isOpen) onClose()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-background/80 backdrop-blur-sm pt-24">
      <div className="w-full max-w-2xl overflow-hidden rounded-xl border border-border bg-surface shadow-2xl">
        {/* Search input */}
        <div className="flex items-center border-b border-border px-4 py-3">
          <Search className="mr-3 h-5 w-5 text-text-muted" />
          <Input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setSelectedIndex(0)
            }}
            placeholder="Type a command or search..."
            className="flex-1 border-0 bg-transparent p-0 text-text-primary placeholder:text-text-muted focus-visible:ring-0"
            autoFocus
          />
          <button
            onClick={onClose}
            className="rounded-md p-1 text-text-muted hover:bg-surface hover:text-text-primary"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Commands */}
        <ScrollArea className="max-h-[400px]">
          <div className="p-2">
            {Object.entries(groupedCommands).map(([category, cmds]) => (
              <div key={category} className="mb-4">
                <div className="mb-2 px-2 text-xs font-medium uppercase tracking-wider text-text-muted">
                  {category}
                </div>
                <div className="space-y-0.5">
                  {cmds.map((cmd) => {
                    const globalIndex = filteredCommands.indexOf(cmd)
                    const isSelected = globalIndex === selectedIndex

                    return (
                      <button
                        key={cmd.id}
                        onClick={() => {
                          onCommand?.(cmd.id)
                          onClose()
                        }}
                        className={cn(
                          "flex w-full items-center justify-between rounded-md px-2 py-2.5 text-left transition-colors",
                          isSelected
                            ? "bg-[#FF6B35]/20 text-text-primary"
                            : "text-text-secondary hover:bg-surface hover:text-text-primary"
                        )}
                      >
                        <div className="flex items-center gap-3">
                          {cmd.icon}
                          <span className="text-sm">{cmd.label}</span>
                        </div>
                        {cmd.shortcut && (
                          <div className="flex items-center gap-1">
                            <kbd className="rounded border border-border bg-surface px-1.5 py-0.5 text-xs text-text-muted">
                              {cmd.shortcut}
                            </kbd>
                          </div>
                        )}
                        {category === "suggested" && (
                          <ChevronRight className="h-4 w-4 text-text-muted" />
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>
            ))}

            {filteredCommands.length === 0 && (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Command className="mb-3 h-8 w-8 text-text-muted" />
                <p className="text-sm text-text-muted">No commands found</p>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-border px-4 py-2 text-xs text-text-muted">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="rounded border border-border bg-surface px-1.5 py-0.5">↑↓</kbd>
              to navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="rounded border border-border bg-surface px-1.5 py-0.5">↵</kbd>
              to select
            </span>
          </div>
          <div className="flex items-center gap-1">
            <kbd className="rounded border border-border bg-surface px-1.5 py-0.5">esc</kbd>
            to close
          </div>
        </div>
      </div>
    </div>
  )
}

// Hook to use command palette globally
let globalOpenCallback: (() => void) | null = null
// eslint-disable-next-line @typescript-eslint/no-unused-vars
let globalCloseCallback: (() => void) | null = null

export function useCommandPalette(onCommand?: (command: string) => void) {
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    globalOpenCallback = () => setIsOpen(true)
    globalCloseCallback = () => setIsOpen(false)
    return () => {
      globalOpenCallback = null
      globalCloseCallback = null
    }
  }, [])

  const handleCommand = (command: string) => {
    onCommand?.(command)
    setIsOpen(false)
  }

  return {
    isOpen,
    open: () => setIsOpen(true),
    close: () => setIsOpen(false),
    toggle: () => setIsOpen((prev) => !prev),
    CommandPalette: (
      <CommandPalette
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        onCommand={handleCommand}
      />
    ),
  }
}

// Global trigger component
export function CommandPaletteProvider({
  children,
  onCommand,
}: {
  children: React.ReactNode
  onCommand?: (command: string) => void
}) {
  const { CommandPalette: Palette } = useCommandPalette(onCommand)

  // Trigger open on Cmd+K from anywhere
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault()
        globalOpenCallback?.()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  return (
    <>
      {children}
      {Palette}
    </>
  )
}
