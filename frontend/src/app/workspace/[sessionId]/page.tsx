"use client"

import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { ThreePanelLayout } from "@/components/layout/ThreePanelLayout"
import { AutoInsightModal } from "@/components/agent/AutoInsightModal"
import { BackendStatusIndicator } from "@/components/ui/BackendStatusIndicator"
import { api, type SessionInfo } from "@/lib/api"
import { AlertCircle } from "lucide-react"

const AUTO_INSIGHTS = [
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
    body: "Agencies that spend heavily in Q4 show lower output metrics — consistent with rushed end-of-year budget consumption.",
  },
]

export default function WorkspacePage() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const router = useRouter()
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [showInsights, setShowInsights] = useState(false)

  useEffect(() => {
    api.getSessionInfo(sessionId)
      .then((info) => {
        setSessionInfo(info)
        setTimeout(() => setShowInsights(true), 1500)
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Session not found"))
      .finally(() => setLoading(false))
  }, [sessionId])

  if (loading) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-primary animate-pulse-fast" />
          <p className="text-sm text-text-muted">Loading session...</p>
        </div>
      </div>
    )
  }

  if (error || !sessionInfo) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-6 text-center">
          <AlertCircle className="h-12 w-12 text-error/60" />
          <div>
            <h2 className="text-lg font-semibold text-text-primary">{error || "Session not found"}</h2>
            <p className="mt-1 max-w-sm text-sm text-text-muted">
              The session may have expired or the backend restarted.
            </p>
          </div>
          <BackendStatusIndicator />
          <div className="flex gap-2">
            <button
              onClick={() => router.push("/")}
              className="rounded bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
            >
              Upload New Dataset
            </button>
            <button
              onClick={() => window.location.reload()}
              className="rounded border border-border px-4 py-2 text-sm font-medium text-text-secondary transition-colors hover:border-border-hover hover:text-text-primary"
            >
              Retry
            </button>
          </div>
          <pre className="rounded bg-surface px-4 py-2 font-mono text-[11px] text-text-muted">
            {sessionId}
          </pre>
        </div>
      </div>
    )
  }

  return (
    <>
      {showInsights && (
        <AutoInsightModal
          insights={AUTO_INSIGHTS}
          onDigDeeper={() => setShowInsights(false)}
          onShowVisualizations={() => setShowInsights(false)}
          onDismiss={() => setShowInsights(false)}
        />
      )}
      <ThreePanelLayout
        sessionId={sessionId}
        sessionInfo={{
          filename: sessionInfo.filename,
          shape: sessionInfo.shape as [number, number],
          columns: sessionInfo.columns,
          dtypes: sessionInfo.dtypes,
        }}
      />
    </>
  )
}
