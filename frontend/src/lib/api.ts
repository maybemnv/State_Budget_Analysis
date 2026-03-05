// API client for DataLens backend
// Base URLs - configure via environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'

// Types
export interface UploadResponse {
  session_id: string
  filename: string
  rows: number
  columns: number
  column_names: string[]
}

export interface SessionInfo {
  session_id: string
  filename: string
  shape: [number, number]
  columns: string[]
  dtypes: Record<string, string>
  numeric_columns?: string[]
  categorical_columns?: string[]
  missing_values?: number
}

export interface ChatResponse {
  answer: string
  chart_spec?: Record<string, unknown>
  has_error: boolean
  steps: Array<{
    tool: string
    args: Record<string, unknown>
    result: unknown
  }>
}

// API Client
export const api = {
  /**
   * Upload a dataset file
   */
  async upload(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    const res = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    })

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Upload failed' }))
      throw new Error(error.detail || 'Upload failed')
    }

    return res.json()
  },

  /**
   * Get session information
   */
  async getSessionInfo(sessionId: string): Promise<SessionInfo> {
    const res = await fetch(`${API_BASE_URL}/sessions/${sessionId}`)

    if (!res.ok) {
      if (res.status === 404) {
        throw new Error('Session not found')
      }
      throw new Error('Failed to fetch session info')
    }

    return res.json()
  },

  /**
   * Delete a session
   */
  async deleteSession(sessionId: string): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
      method: 'DELETE',
    })

    if (!res.ok && res.status !== 404) {
      throw new Error('Failed to delete session')
    }
  },

  /**
   * Chat via HTTP (non-streaming)
   */
  async chat(sessionId: string, message: string): Promise<ChatResponse> {
    const res = await fetch(`${API_BASE_URL}/chat/${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    })

    if (!res.ok) {
      if (res.status === 404) {
        throw new Error('Session not found')
      }
      throw new Error('Chat request failed')
    }

    return res.json()
  },

  /**
   * Health check
   */
  async health(): Promise<{ status: string; version: string }> {
    const res = await fetch(`${API_BASE_URL}/health`)
    if (!res.ok) throw new Error('Health check failed')
    return res.json()
  },
}

// WebSocket client
export interface WSMessage {
  type: 'thought' | 'tool_call' | 'tool_result' | 'chart' | 'answer' | 'error' | 'done'
  content?: string
  tool?: string
  args?: Record<string, unknown>
  result?: unknown
  spec?: Record<string, unknown>
  message?: string
}

export interface UseWebSocketOptions {
  sessionId: string
  onMessage?: (message: WSMessage) => void
  onThought?: (content: string) => void
  onToolCall?: (tool: string, args: Record<string, unknown>) => void
  onToolResult?: (tool: string, result: unknown) => void
  onChart?: (spec: Record<string, unknown>) => void
  onAnswer?: (content: string) => void
  onError?: (message: string) => void
  onDone?: () => void
  onConnect?: () => void
  onDisconnect?: () => void
}

export function createWebSocketClient({
  sessionId,
  onMessage,
  onThought,
  onToolCall,
  onToolResult,
  onChart,
  onAnswer,
  onError,
  onDone,
  onConnect,
  onDisconnect,
}: UseWebSocketOptions) {
  const wsUrl = `${WS_BASE_URL}/ws/${sessionId}`
  console.log('Connecting to WebSocket:', wsUrl)
  
  const ws = new WebSocket(wsUrl)

  ws.onopen = () => {
    console.log('WebSocket connected')
    onConnect?.()
  }

  ws.onclose = (event) => {
    console.log('WebSocket closed:', event.code, event.reason)
    onDisconnect?.()
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
    // Don't call onError here - it will be called by onclose with proper info
  }

  ws.onmessage = (event) => {
    try {
      const message: WSMessage = JSON.parse(event.data)
      console.log('WS message received:', message.type)
      onMessage?.(message)

      // Dispatch to specific handlers
      switch (message.type) {
        case 'thought':
          onThought?.(message.content || '')
          break
        case 'tool_call':
          onToolCall?.(message.tool || '', message.args || {})
          break
        case 'tool_result':
          onToolResult?.(message.tool || '', message.result)
          break
        case 'chart':
          onChart?.(message.spec || {})
          break
        case 'answer':
          onAnswer?.(message.content || '')
          break
        case 'error':
          onError?.(message.message || 'Unknown error')
          break
        case 'done':
          onDone?.()
          break
      }
    } catch (e) {
      console.error('Failed to parse WebSocket message:', e)
    }
  }

  return {
    ws,
    send: (message: string | Record<string, unknown>) => {
      const payload = typeof message === 'string' 
        ? JSON.stringify({ message }) 
        : JSON.stringify(message)
      
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(payload)
      } else {
        console.warn('WebSocket not connected, readyState:', ws.readyState)
      }
    },
    close: () => {
      ws.close()
    },
  }
}

// Export constants
export { API_BASE_URL, WS_BASE_URL }
