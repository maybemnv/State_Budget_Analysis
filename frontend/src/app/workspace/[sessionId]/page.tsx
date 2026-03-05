"use client"

import { useParams, useRouter } from "next/navigation"
import { ThreePanelLayout } from "@/components/layout/ThreePanelLayout"
import { useEffect, useState } from "react"
import { api, type SessionInfo } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { AlertCircle } from "lucide-react"

export default function WorkspacePage() {
  const params = useParams()
  const router = useRouter()
  const sessionId = params.sessionId as string
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    console.log('Workspace loading with session ID:', sessionId)
    
    async function loadSession() {
      try {
        setIsLoading(true)
        console.log('Fetching session info from backend...')
        const info = await api.getSessionInfo(sessionId)
        console.log('Session info loaded:', info)
        setSessionInfo(info)
      } catch (err) {
        console.error('Failed to load session:', err)
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
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-error/10">
            <AlertCircle className="h-8 w-8 text-error" />
          </div>
          <h2 className="text-xl font-semibold text-text-primary">
            {error || 'Session not found'}
          </h2>
          <p className="text-sm text-text-muted max-w-md">
            The session you're looking for doesn't exist or has expired. 
            Please upload a new dataset.
          </p>
          <Button
            onClick={() => router.push('/')}
            className="bg-[#FF6B35] hover:bg-[#FF8555]"
          >
            Upload New Dataset
          </Button>
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
