"use client"

import { useParams, useRouter } from "next/navigation"
import { ThreePanelLayout } from "@/components/layout/ThreePanelLayout"
import { useEffect, useState } from "react"
import { api, type SessionInfo } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { AlertCircle, WifiOff } from "lucide-react"
import { BackendStatusIndicator } from "@/components/ui/BackendStatusIndicator"
import { AutoInsightModal } from "@/components/agent/AutoInsightModal"

export default function WorkspacePage() {
  const params = useParams()
  const router = useRouter()
  const sessionId = params.sessionId as string
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState("")
  const [showInsights, setShowInsights] = useState(false)

  useEffect(() => {
    async function loadSession() {
      try {
        setIsLoading(true)
        const info = await api.getSessionInfo(sessionId)
        setSessionInfo(info)
        setTimeout(() => setShowInsights(true), 1500)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Session not found')
      } finally {
        setIsLoading(false)
      }
    }

    loadSession()
  }, [sessionId])

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

  if (error || !sessionInfo) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-6 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-error/10">
            <WifiOff className="h-8 w-8 text-error" />
          </div>
          <div className="space-y-2">
            <h2 className="text-xl font-semibold text-text-primary">
              {error || 'Session not found'}
            </h2>
            <p className="text-sm text-text-muted max-w-md">
              The session doesn't exist or has expired. This happens when the backend restarts.
            </p>
          </div>
          
          {/* Backend Status */}
          <div className="flex items-center gap-2">
            <BackendStatusIndicator />
          </div>
          
          <div className="flex gap-3 pt-2">
            <Button
              onClick={() => router.push('/')}
              className="bg-[#FF6B35] hover:bg-[#FF8555]"
            >
              Upload New Dataset
            </Button>
            <Button
              variant="outline"
              onClick={() => window.location.reload()}
              className="border-border bg-surface/50"
            >
              Retry
            </Button>
          </div>
          
          {/* Debug info */}
          <div className="mt-4 rounded-md bg-surface/50 p-3 text-left">
            <p className="text-xs text-text-muted mb-2 font-medium">Debug Info:</p>
            <code className="text-[10px] text-text-secondary block">
              Session ID: {sessionId}<br/>
              Backend: {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}<br/>
              Error: {error}
            </code>
          </div>
        </div>
      </div>
    )
  }

  const autoInsights = [
    {
      title: "Spending variance is increasing over time",
      body: "While total budget grew over the period, variance between agencies is accelerating — suggesting diverging budget priorities across departments.",
    },
    {
      title: "One category dominates disproportionate budget share",
      body: "A small subset of categories consumes the majority of total expenditure relative to their program count, indicating a high cost-per-program ratio.",
    },
    {
      title: "Q4 spending spikes correlate with lower outcome scores",
      body: "Agencies that spend heavily in Q4 show lower output metrics — a pattern consistent with rushed end-of-year budget consumption.",
    },
  ]

  return (
    <>
      {showInsights && (
        <AutoInsightModal
          insights={autoInsights}
          onDigDeeper={(insight) => {
            setShowInsights(false)
          }}
          onShowVisualizations={() => setShowInsights(false)}
          onDismiss={() => setShowInsights(false)}
        />
      )}
      <ThreePanelLayout
        sessionId={sessionId}
        sessionInfo={{
          filename: sessionInfo!.filename,
          shape: sessionInfo!.shape as [number, number],
          columns: sessionInfo!.columns,
          dtypes: sessionInfo!.dtypes,
        }}
      />
    </>
  )
}
