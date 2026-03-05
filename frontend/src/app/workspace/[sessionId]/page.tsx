"use client"

import { useParams } from "next/navigation"
import { ThreePanelLayout } from "@/components/layout/ThreePanelLayout"
import { useEffect, useState } from "react"

// Mock session info — in real app, fetch from backend
const mockSessionInfo = {
  filename: "state_budget_2024.csv",
  shape: [12438, 8] as [number, number],
  columns: [
    "date",
    "agency_id",
    "agency_name",
    "category",
    "program",
    "amount",
    "fiscal_year",
    "region",
  ],
  dtypes: {
    date: "datetime64[ns]",
    agency_id: "object",
    agency_name: "object",
    category: "object",
    program: "object",
    amount: "float64",
    fiscal_year: "int64",
    region: "object",
  },
}

export default function WorkspacePage() {
  const params = useParams()
  const sessionId = params.sessionId as string
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate loading session data
    const timer = setTimeout(() => setIsLoading(false), 500)
    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="h-12 w-12 animate-pulse-fast rounded-full bg-[#FF6B35] shadow-[0_0_30px_8px_rgba(255,133,85,0.5)]" />
          <p className="text-sm text-text-muted">Loading session...</p>
        </div>
      </div>
    )
  }

  return (
    <ThreePanelLayout
      sessionId={sessionId}
      sessionInfo={mockSessionInfo}
    />
  )
}
