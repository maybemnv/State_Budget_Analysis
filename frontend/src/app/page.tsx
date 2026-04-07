"use client"

import { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import { Upload, FileText, CheckCircle2, Loader2, AlertCircle, ArrowRight } from "lucide-react"
import { api } from "@/lib/api"
import { cn } from "@/lib/utils"

type UploadState = "idle" | "uploading" | "parsing" | "done" | "error"

const PREVIEW_ROWS = [
  ["2024-01", "AG-001", "Infrastructure", "$123,456"],
  ["2024-02", "AG-002", "Education", "$234,567"],
  ["2024-03", "AG-001", "Healthcare", "$345,678"],
  ["...", "...", "...", "..."],
]

export default function HomePage() {
  const router = useRouter()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [state, setState] = useState<UploadState>("idle")
  const [progress, setProgress] = useState(0)
  const [fileName, setFileName] = useState("")
  const [error, setError] = useState("")
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = async (file: File) => {
    if (!file.name.match(/\.(csv|xlsx|xls|parquet)$/)) {
      setError("Unsupported format. Upload CSV, XLSX, or Parquet.")
      return
    }

    setFileName(file.name)
    setState("uploading")
    setProgress(0)
    setError("")

    const interval = setInterval(() => {
      setProgress((p) => (p >= 88 ? (clearInterval(interval), 88) : p + 12))
    }, 180)

    try {
      const { session_id } = await api.upload(file)
      clearInterval(interval)
      setProgress(100)
      setState("parsing")
      await new Promise((r) => setTimeout(r, 900))
      setState("done")
      await new Promise((r) => setTimeout(r, 700))
      router.push(`/workspace/${session_id}`)
    } catch (err) {
      clearInterval(interval)
      setError(err instanceof Error ? err.message : "Upload failed")
      setState("error")
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Nav */}
      <nav className="flex items-center justify-between px-10 py-5">
        <span className="text-sm font-semibold tracking-tight text-text-primary">
          DataLens<span className="text-primary"> AI</span>
        </span>
        <div className="flex gap-1">
          {["Terminal", "Archive", "Agents", "Settings"].map((item) => (
            <button
              key={item}
              className="rounded-full px-3.5 py-1.5 text-xs font-medium text-text-muted transition-colors hover:bg-surface hover:text-text-primary"
            >
              {item}
            </button>
          ))}
        </div>
      </nav>

      {/* Hero */}
      <div className="flex flex-1 flex-col items-center justify-center px-6 pb-20 pt-12">
        {/* Orange radial — the ONE accent region */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 flex items-start justify-center"
          style={{
            background:
              "radial-gradient(ellipse 60% 40% at 50% 20%, rgba(228,77,10,0.09) 0%, transparent 70%)",
          }}
        />

        {/* Ghost code texture — upper right, decorative only */}
        <pre
          aria-hidden
          className="pointer-events-none absolute right-10 top-24 select-none font-mono text-[11px] leading-5 text-text-primary opacity-[0.04]"
        >
          {`const df = pd.read_csv(file)
schema = df.dtypes.to_dict()
agent.run(query, context=schema)
insights = llm.analyze(df.head(100))`}
        </pre>

        <div className="relative w-full max-w-xl">
          {/* Headline */}
          <div className="mb-10 text-center">
            <h1 className="mb-3 text-5xl font-bold tracking-tight text-text-primary">
              Ingest. Analyze. Act.
            </h1>
            <p className="text-base text-text-muted">
              Drop your dataset. The agent does the rest.
            </p>
          </div>

          {/* Upload card */}
          <div className="rounded-lg border border-border bg-surface">
            {state === "idle" && (
              <div
                onDrop={onDrop}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
                onDragLeave={() => setIsDragging(false)}
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  "flex cursor-pointer flex-col items-center px-10 py-14 text-center transition-colors",
                  isDragging ? "bg-elevated" : "hover:bg-elevated/50"
                )}
              >
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
                  <Upload className="h-5 w-5 text-primary" />
                </div>
                <p className="mb-1 text-sm font-medium text-text-primary">
                  Drop your file here
                </p>
                <p className="text-xs text-text-muted">CSV · XLSX · Parquet · up to 100 MB</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls,.parquet"
                  className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
                />
              </div>
            )}

            {(state === "uploading" || state === "parsing") && (
              <div className="px-10 py-10">
                <div className="mb-5 flex items-center gap-3">
                  {state === "parsing" ? (
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  ) : (
                    <FileText className="h-4 w-4 text-text-muted" />
                  )}
                  <span className="text-sm font-medium text-text-primary">{fileName}</span>
                </div>

                {/* Progress bar */}
                <div className="mb-1 h-1 w-full overflow-hidden rounded-full bg-elevated">
                  <div
                    className="h-full bg-primary transition-all duration-200"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="mb-6 text-right text-xs text-text-muted">{progress}%</p>

                {/* Cascade table — parsing state */}
                {state === "parsing" && (
                  <div className="overflow-hidden rounded border border-border font-mono text-[11px]">
                    <div className="grid grid-cols-4 border-b border-border bg-elevated px-3 py-2 font-semibold text-text-muted animate-fade-in-up">
                      {["date", "agency", "category", "amount"].map((h) => <span key={h}>{h}</span>)}
                    </div>
                    {PREVIEW_ROWS.map((row, i) => (
                      <div
                        key={i}
                        className="grid grid-cols-4 border-b border-border/50 px-3 py-1.5 text-text-secondary opacity-0 last:border-0 animate-fade-in-up"
                        style={{ animationDelay: `${(i + 1) * 120}ms`, animationFillMode: "forwards" }}
                      >
                        {row.map((cell, j) => <span key={j}>{cell}</span>)}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {state === "done" && (
              <div className="flex flex-col items-center px-10 py-14">
                <CheckCircle2 className="mb-3 h-10 w-10 text-success animate-scale-in" />
                <p className="text-sm font-medium text-text-primary">Dataset ready — redirecting</p>
              </div>
            )}

            {state === "error" && (
              <div className="px-10 py-10 text-center">
                <AlertCircle className="mx-auto mb-3 h-8 w-8 text-error" />
                <p className="mb-4 text-sm text-text-primary">{error}</p>
                <button
                  onClick={() => { setState("idle"); setError("") }}
                  className="rounded-full border border-border px-5 py-2 text-sm text-text-secondary transition-colors hover:border-border-hover hover:text-text-primary"
                >
                  Try again
                </button>
              </div>
            )}
          </div>

          {/* Inline CTA row */}
          {state === "idle" && (
            <div className="mt-6 flex items-center justify-center gap-6 text-xs text-text-muted">
              <span>CSV, XLSX, Parquet</span>
              <span className="h-3 w-px bg-border" />
              <span>Up to 100 MB</span>
              <span className="h-3 w-px bg-border" />
              <span>No account needed</span>
            </div>
          )}
        </div>

        {/* Typographic counterweight */}
        <p
          aria-hidden
          className="pointer-events-none absolute bottom-8 right-10 select-none font-mono text-[80px] font-bold leading-none tracking-tighter text-text-primary opacity-[0.03]"
        >
          DL.AI
        </p>
      </div>
    </div>
  )
}
