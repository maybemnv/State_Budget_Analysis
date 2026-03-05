"use client"

import { useBackendStatus } from "@/hooks/useBackendStatus"
import { cn } from "@/lib/utils"
import { Wifi, WifiOff, AlertCircle, Server } from "lucide-react"

export function BackendStatusIndicator() {
  const status = useBackendStatus()

  return (
    <div className="flex items-center gap-2 text-xs">
      {/* Backend Connection */}
      <div
        className={cn(
          "flex items-center gap-1.5 px-2 py-1 rounded-full border",
          status.connected
            ? "bg-success/10 border-success/30 text-success"
            : "bg-error/10 border-error/30 text-error"
        )}
        title={status.connected ? "Backend connected" : status.error}
      >
        {status.connected ? (
          <Wifi className="h-3 w-3" />
        ) : (
          <WifiOff className="h-3 w-3" />
        )}
        <span className="font-medium">
          {status.connected ? "Backend OK" : "Backend Error"}
        </span>
      </div>

      {/* Session Count */}
      {status.connected && status.sessionCount !== undefined && (
        <div
          className="flex items-center gap-1.5 px-2 py-1 rounded-full border bg-surface/50 border-border text-text-muted"
          title="Active sessions"
        >
          <Server className="h-3 w-3" />
          <span>{status.sessionCount} session{status.sessionCount !== 1 ? 's' : ''}</span>
        </div>
      )}

      {/* Error Details */}
      {!status.connected && status.error && (
        <div
          className="flex items-center gap-1.5 px-2 py-1 rounded-full border bg-error/10 border-error/30 text-error"
          title={status.error}
        >
          <AlertCircle className="h-3 w-3" />
          <span className="max-w-[200px] truncate">{status.error}</span>
        </div>
      )}

      {/* URLs (dev only) */}
      <div className="hidden md:flex items-center gap-2 text-text-muted ml-2">
        <code className="text-[10px]">{status.backendUrl}</code>
      </div>
    </div>
  )
}
