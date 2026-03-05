"use client"

import { cn } from "@/lib/utils"
import { AgentState } from "@/lib/types"

interface AgentAvatarProps {
  state?: AgentState
  className?: string
}

/**
 * AgentAvatar — An abstract animated orb that represents the agent's presence.
 * Changes color and pulse based on agent state.
 * 
 * States:
 * - idle: subtle purple glow, slow pulse
 * - thinking: bright purple, slow pulse (~1Hz)
 * - executing: orange, fast pulse (~3Hz)
 * - done: teal, steady glow
 * - error: red, flickering
 */
export function AgentAvatar({ state = "idle", className }: AgentAvatarProps) {
  const stateStyles = {
    idle: "bg-agent/60 shadow-[0_0_20px_4px_rgba(157,78,221,0.3)] animate-pulse-slow",
    thinking: "bg-agent shadow-[0_0_30px_8px_rgba(157,78,221,0.5)] animate-pulse",
    executing: "bg-[#FF8555] shadow-[0_0_30px_8px_rgba(255,133,85,0.5)] animate-pulse-fast",
    done: "bg-success shadow-[0_0_20px_4px_rgba(0,220,180,0.3)]",
    error: "bg-error shadow-[0_0_20px_4px_rgba(239,68,68,0.5)] animate-flicker",
  }

  return (
    <div
      className={cn(
        "relative flex h-10 w-10 items-center justify-center rounded-full transition-all duration-500",
        stateStyles[state],
        className
      )}
      role="status"
      aria-label={`Agent is ${state}`}
    >
      {/* Inner core */}
      <div className="h-4 w-4 rounded-full bg-white/80 blur-[1px]" />
      
      {/* Outer glow ring */}
      <div
        className={cn(
          "absolute inset-0 rounded-full opacity-30",
          state === "thinking" && "animate-ping-slow",
          state === "executing" && "animate-ping-fast"
        )}
        style={{
          background: state === "executing" 
            ? "radial-gradient(circle, rgba(255,133,85,0.4) 0%, transparent 70%)"
            : "radial-gradient(circle, rgba(157,78,221,0.4) 0%, transparent 70%)"
        }}
      />
    </div>
  )
}
