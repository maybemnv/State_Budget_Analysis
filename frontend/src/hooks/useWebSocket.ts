"use client"

import { useEffect, useRef, useCallback, useState } from "react"
import type { WSMessageType, ThoughtMessage, ToolCallMessage, ToolResultMessage, ChartSpecMessage, AnswerMessage, ErrorMessage } from "@/lib/types"

export interface UseWebSocketOptions {
  sessionId: string
  baseUrl?: string
  onMessage?: (message: WSMessageType) => void
  onThought?: (message: ThoughtMessage) => void
  onToolCall?: (message: ToolCallMessage) => void
  onToolResult?: (message: ToolResultMessage) => void
  onChart?: (message: ChartSpecMessage) => void
  onAnswer?: (message: AnswerMessage) => void
  onError?: (message: ErrorMessage) => void
  onDone?: () => void
  onConnect?: () => void
  onDisconnect?: () => void
}

export interface UseWebSocketReturn {
  sendMessage: (message: string | Record<string, unknown>) => void
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  messages: WSMessageType[]
}

/**
 * useWebSocket — Hook for real-time communication with the DataLens backend.
 * Handles reconnection, message queuing, and typed message dispatch.
 */
export function useWebSocket({
  sessionId,
  baseUrl = "",
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
}: UseWebSocketOptions): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [messages, setMessages] = useState<WSMessageType[]>([])
  const messageQueue = useRef<string[]>([])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setIsConnecting(true)
    setError(null)

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const wsUrl = `${protocol}//${window.location.host}${baseUrl}/ws/${sessionId}`

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      setIsConnected(true)
      setIsConnecting(false)
      onConnect?.()

      // Flush message queue
      messageQueue.current.forEach((msg) => ws.send(msg))
      messageQueue.current = []
    }

    ws.onclose = () => {
      setIsConnected(false)
      setIsConnecting(false)
      onDisconnect?.()
    }

    ws.onerror = () => {
      setError("WebSocket connection failed")
      setIsConnecting(false)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WSMessageType
        setMessages((prev) => [...prev, data])
        onMessage?.(data)

        // Dispatch to specific handlers
        switch (data.type) {
          case "thought":
            onThought?.(data)
            break
          case "tool_call":
            onToolCall?.(data)
            break
          case "tool_result":
            onToolResult?.(data)
            break
          case "chart":
            onChart?.(data)
            break
          case "answer":
            onAnswer?.(data)
            break
          case "error":
            onError?.(data)
            break
          case "done":
            onDone?.()
            break
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e)
      }
    }

    wsRef.current = ws
  }, [sessionId, baseUrl, onMessage, onThought, onToolCall, onToolResult, onChart, onAnswer, onError, onDone, onConnect, onDisconnect])

  useEffect(() => {
    connect()

    return () => {
      wsRef.current?.close()
    }
  }, [connect])

  const sendMessage = useCallback(
    (message: string | Record<string, unknown>) => {
      const payload = typeof message === "string" ? message : JSON.stringify(message)

      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(payload)
      } else {
        // Queue for later
        messageQueue.current.push(payload)
      }
    },
    []
  )

  return {
    sendMessage,
    isConnected,
    isConnecting,
    error,
    messages,
  }
}
