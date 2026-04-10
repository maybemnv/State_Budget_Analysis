"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import {
  createWebSocketClient,
  type WSMessage,
} from "@/lib/api"

export interface UseWebSocketOptions {
  sessionId: string
  baseUrl?: string
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
  /** Maximum reconnection attempts. Default: 10 */
  maxReconnectAttempts?: number
}

export interface UseWebSocketReturn {
  sendMessage: (message: string | Record<string, unknown>) => void
  isConnected: boolean
  isConnecting: boolean
  isReconnecting: boolean
  reconnectAttempts: number
  error: string | null
  messages: WSMessage[]
  disconnect: () => void
}

/**
 * useWebSocket — Hook for real-time communication with the DataLens backend.
 * Uses WebSocket streaming for agent responses.
 *
 * Key design: callbacks are stored in refs so the connect function is stable.
 * This prevents the infinite connect/disconnect loop caused by React re-renders
 * creating new callback references on every render.
 */
export function useWebSocket({
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
  maxReconnectAttempts = 10,
}: UseWebSocketOptions): UseWebSocketReturn {
  const clientRef = useRef<ReturnType<typeof createWebSocketClient> | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(true)
  const [isReconnecting, setIsReconnecting] = useState(false)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [messages, setMessages] = useState<WSMessage[]>([])

  // ─── Stable callback refs ────────────────────────────────────────────
  // Store callbacks in refs so the connect function below is stable
  // (only depends on sessionId). Without this, every re-render creates
  // new callback references → connect changes → useEffect cleanup →
  // WS close → reconnect → infinite loop.
  const callbacksRef = useRef({
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
  })

  // Keep refs up to date on every render
  useEffect(() => {
    callbacksRef.current = {
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
    }
  })

  // ─── Stable connect function ─────────────────────────────────────────
  // Only re-creates when sessionId or maxReconnectAttempts changes.
  const connect = useRef(false) // dummy dep to avoid recreating; we use sessionId directly

  useEffect(() => {
    // Cleanup previous connection
    if (clientRef.current) {
      clientRef.current.close()
    }

    setIsConnecting(true)
    setIsReconnecting(false)
    setError(null)

    clientRef.current = createWebSocketClient({
      sessionId,
      maxReconnectAttempts,
      onMessage: (msg) => {
        setMessages((prev) => [...prev, msg])
        callbacksRef.current.onMessage?.(msg)
      },
      onThought: (content) => callbacksRef.current.onThought?.(content),
      onToolCall: (tool, args) => callbacksRef.current.onToolCall?.(tool, args),
      onToolResult: (tool, result) => callbacksRef.current.onToolResult?.(tool, result),
      onChart: (spec) => callbacksRef.current.onChart?.(spec),
      onAnswer: (content) => callbacksRef.current.onAnswer?.(content),
      onError: (msg) => {
        setError(msg)
        callbacksRef.current.onError?.(msg)
      },
      onDone: () => callbacksRef.current.onDone?.(),
      onConnect: () => {
        setIsConnected(true)
        setIsConnecting(false)
        setIsReconnecting(false)
        setReconnectAttempts(0)
        callbacksRef.current.onConnect?.()
      },
      onDisconnect: () => {
        setIsConnected(false)
        setIsReconnecting(true)
        callbacksRef.current.onDisconnect?.()
      },
    })

    return () => {
      clientRef.current?.close()
    }
    // Only re-run when sessionId or maxReconnectAttempts change
  }, [sessionId, maxReconnectAttempts])

  const sendMessage = useCallback(
    (message: string | Record<string, unknown>) => {
      clientRef.current?.send(message)
    },
    []
  )

  const disconnect = useCallback(() => {
    clientRef.current?.close()
  }, [])

  return {
    sendMessage,
    isConnected,
    isConnecting,
    isReconnecting,
    reconnectAttempts,
    error,
    messages,
    disconnect,
  }
}
