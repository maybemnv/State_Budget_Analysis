import type { HTMLAttributes } from "react"
import { cn } from "@/lib/utils"

interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  variant?: "text" | "circular" | "rectangular" | "rounded"
  width?: string | number
  height?: string | number
  animation?: "pulse" | "wave" | "none"
}

/**
 * Skeleton — placeholder loading for content that hasn't loaded yet.
 * Matches the warm editorial theme.
 */
export function Skeleton({
  variant = "rounded",
  width,
  height,
  animation = "wave",
  className,
  style,
  ...props
}: SkeletonProps) {
  return (
    <div
      className={cn(
        "bg-elevated",
        {
          "rounded-sm": variant === "text",
          "rounded-full": variant === "circular",
          "rounded": variant === "rounded",
          "animate-pulse": animation === "pulse" || animation === "wave",
        },
        className
      )}
      style={{
        width,
        height,
        ...style,
      }}
      aria-hidden="true"
      {...props}
    />
  )
}
