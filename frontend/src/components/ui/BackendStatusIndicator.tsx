"use client"

import { useBackendStatus } from "@/hooks/useBackendStatus"
import { cn } from "@/lib/utils"
import { Wifi, WifiOff } from "lucide-react"

export function BackendStatusIndicator() {
  const { connected, error } = useBackendStatus()

  return (
    <div
      title={connected ? "Backend connected" : error ?? "Backend unreachable"}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium",
        connected
          ? "bg-success/10 text-success"
          : "bg-error/10 text-error"
      )}
    >
      {connected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
      {connected ? "Backend OK" : "Backend Error"}
    </div>
  )
}
