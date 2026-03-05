"use client"

import { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Upload, FileSpreadsheet, CheckCircle2, Loader2, Sparkles, AlertCircle } from "lucide-react"
import { api } from "@/lib/api"

type UploadStateType = "idle" | "uploading" | "parsing" | "done" | "error"

export default function HomePage() {
  const router = useRouter()
  const [uploadState, setUploadState] = useState<UploadStateType>("idle")
  const [progress, setProgress] = useState(0)
  const [fileName, setFileName] = useState("")
  const [error, setError] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = async (file: File) => {
    setFileName(file.name)
    setUploadState("uploading" as UploadStateType)
    setProgress(0)
    setError("")

    try {
      // Simulate progress during upload
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 200)

      // Upload to backend
      const response = await api.upload(file)
      
      clearInterval(progressInterval)
      setProgress(100)
      setUploadState("parsing" as UploadStateType)

      // Brief pause to show completion
      await new Promise(resolve => setTimeout(resolve, 800))
      
      setUploadState("done" as UploadStateType)
      
      // Redirect to workspace with real session ID
      await new Promise(resolve => setTimeout(resolve, 1000))
      router.push(`/workspace/${response.session_id}`)
    } catch (err) {
      console.error('Upload error:', err)
      setError(err instanceof Error ? err.message : 'Upload failed')
      setUploadState("error" as UploadStateType)
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.name.match(/\.(csv|xlsx|xls|parquet)$/)) {
      handleFileSelect(file)
    } else {
      setError('Unsupported file type. Please upload CSV, XLSX, XLS, or Parquet.')
    }
  }

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background p-4">
      {/* Hero */}
      <div className="mb-12 text-center">
        <div className="mb-4 flex items-center justify-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#FF6B35] shadow-[0_0_30px_8px_rgba(255,133,85,0.3)]">
            <Sparkles className="h-6 w-6 text-white" />
          </div>
        </div>
        <h1 className="mb-3 text-4xl font-bold tracking-tight text-text-primary">
          DataLens AI
        </h1>
        <p className="max-w-md text-text-secondary">
          Autonomous data analysis powered by AI. Upload your dataset and start
          a conversation.
        </p>
      </div>

      {/* Upload Card */}
      <Card className="w-full max-w-xl border-border bg-surface/30">
        <CardContent className="p-8">
          {uploadState === "idle" && (
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              className="flex flex-col items-center justify-center border-2 border-dashed border-border rounded-lg p-12 text-center transition-colors hover:border-[#FF6B35]/50 hover:bg-surface/30"
            >
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-surface">
                <Upload className="h-8 w-8 text-[#FF6B35]" />
              </div>
              <h3 className="mb-2 text-lg font-medium text-text-primary">
                Drop your file here
              </h3>
              <p className="mb-4 text-sm text-text-muted">
                or click to browse
              </p>
              <p className="text-xs text-text-muted">
                Supported: CSV, XLSX, XLS, Parquet (max 100MB)
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xlsx,.xls,.parquet"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) handleFileSelect(file)
                }}
              />
              <Button
                onClick={() => fileInputRef.current?.click()}
                className="mt-6 bg-[#FF6B35] hover:bg-[#FF8555]"
              >
                Select File
              </Button>
            </div>
          )}

          {uploadState === "uploading" && (
            <div className="flex flex-col items-center py-12">
              <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-surface">
                <FileSpreadsheet className="h-8 w-8 text-[#FF6B35]" />
              </div>
              <h3 className="mb-2 text-lg font-medium text-text-primary">
                {fileName}
              </h3>
              <p className="mb-4 text-sm text-text-muted">Uploading to server...</p>
              <div className="w-full max-w-xs">
                <div className="h-2 w-full overflow-hidden rounded-full bg-border">
                  <div
                    className="h-full bg-[#FF6B35] transition-all duration-200"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="mt-2 text-center text-xs text-text-muted">
                  {progress}%
                </p>
              </div>
            </div>
          )}

          {uploadState === "parsing" && (
            <div className="flex flex-col items-center py-12">
              <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-surface">
                <Loader2 className="h-8 w-8 animate-spin text-[#FF6B35]" />
              </div>
              <h3 className="mb-2 text-lg font-medium text-text-primary">
                {fileName}
              </h3>
              <p className="mb-4 text-sm text-text-muted">Processing dataset...</p>
            </div>
          )}

          {uploadState === "done" && (
            <div className="flex flex-col items-center py-12">
              <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-success/10">
                <CheckCircle2 className="h-8 w-8 text-success" />
              </div>
              <h3 className="mb-2 text-lg font-medium text-text-primary">
                Dataset ready!
              </h3>
              <p className="text-sm text-text-muted">
                Redirecting to workspace...
              </p>
            </div>
          )}

          {uploadState === "error" && (
            <div className="flex flex-col items-center py-12">
              <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-error/10">
                <AlertCircle className="h-8 w-8 text-error" />
              </div>
              <h3 className="mb-2 text-lg font-medium text-text-primary">
                Upload failed
              </h3>
              <p className="text-sm text-text-muted mb-4">
                {error || 'Please try again or contact support.'}
              </p>
              <Button
                variant="outline"
                onClick={() => {
                  setUploadState("idle")
                  setFileName("")
                  setProgress(0)
                  setError("")
                }}
                className="border-border bg-surface/50"
              >
                Try Again
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Features */}
      <div className="mt-16 grid max-w-3xl grid-cols-3 gap-8">
        {[
          {
            icon: <Sparkles className="h-5 w-5" />,
            title: "AI-Powered",
            desc: "Autonomous agent that reasons step-by-step",
          },
          {
            icon: <FileSpreadsheet className="h-5 w-5" />,
            title: "Any Dataset",
            desc: "CSV, Excel, or Parquet up to 100MB",
          },
          {
            icon: <CheckCircle2 className="h-5 w-5" />,
            title: "Actionable Insights",
            desc: "Clear answers with visualizations",
          },
        ].map((feature) => (
          <div key={feature.title} className="text-center">
            <div className="mb-3 flex items-center justify-center gap-2 text-[#FF6B35]">
              {feature.icon}
            </div>
            <h3 className="mb-1 text-sm font-medium text-text-primary">
              {feature.title}
            </h3>
            <p className="text-xs text-text-muted">{feature.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
