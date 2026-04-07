"use client"

import { cn } from "@/lib/utils"
import type { AgentState } from "@/lib/types"

const STATE_CONFIG: Record<AgentState, { bg: string; glow: string; pulse: string; ringColor: string }> = {
  idle:      { bg: "bg-agent/40",  glow: "[box-shadow:0_0_16px_2px_rgba(107,63,160,0.2)]",  pulse: "animate-pulse-slow", ringColor: "rgba(107,63,160,0.3)" },
  thinking:  { bg: "bg-agent",     glow: "[box-shadow:0_0_24px_6px_rgba(107,63,160,0.4)]",  pulse: "animate-pulse-slow", ringColor: "rgba(107,63,160,0.4)" },
  executing: { bg: "bg-primary",   glow: "[box-shadow:0_0_24px_6px_rgba(228,77,10,0.35)]",  pulse: "animate-pulse-fast", ringColor: "rgba(228,77,10,0.35)" },
  done:      { bg: "bg-success",   glow: "[box-shadow:0_0_16px_2px_rgba(42,122,82,0.3)]",   pulse: "",                   ringColor: "" },
  error:     { bg: "bg-error",     glow: "[box-shadow:0_0_16px_4px_rgba(192,57,43,0.4)]",   pulse: "",                   ringColor: "" },
}

export function AgentAvatar({ state = "idle", className }: { state?: AgentState; className?: string }) {
  const cfg = STATE_CONFIG[state]

  return (
    <div
      role="status"
      aria-label={`Agent is ${state}`}
      className={cn(
        "relative flex h-9 w-9 items-center justify-center rounded-full transition-all duration-500",
        cfg.bg,
        cfg.glow,
        cfg.pulse,
        className
      )}
    >
      <div className="h-3.5 w-3.5 rounded-full bg-white/70 blur-[1px]" />
    </div>
  )
}
