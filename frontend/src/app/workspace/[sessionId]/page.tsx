"use client"

import { useParams, useRouter } from "next/navigation"
import { ThreePanelLayout } from "@/components/layout/ThreePanelLayout"
import { useEffect, useState } from "react"
import { api, type SessionInfo } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { AlertCircle, WifiOff } from "lucide-react"
import { BackendStatusIndicator } from "@/components/ui/BackendStatusIndicator"

export default function WorkspacePage() {
  const params = useParams()
  const router = useRouter()
  const sessionId = params.sessionId as string
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    console.log('🔵 Workspace loading with session ID:', sessionId)
    
    async function loadSession() {
      try {
        setIsLoading(true)
        console.log('🔵 Fetching session info from backend...')
        const info = await api.getSessionInfo(sessionId)
        console.log('✅ Session info loaded:', info)
        setSessionInfo(info)
      } catch (err) {
        console.error('❌ Failed to load session:', err)
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

  return (
    <ThreePanelLayout
      sessionId={sessionId}
      sessionInfo={{
        filename: sessionInfo.filename,
        shape: sessionInfo.shape as [number, number],
        columns: sessionInfo.columns,
        dtypes: sessionInfo.dtypes,
      }}
    />
  )
}
