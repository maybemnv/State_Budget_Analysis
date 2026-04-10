"use client"

import { useEffect, useRef, useCallback, useState } from "react"
import {
  createWebSocketClient,
  type WSMessage,
  type CreateWebSocketClientOptions as ApiOptions,
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
 * Features: auto-reconnection with backoff, connection state tracking.
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

  const connect = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.close()
    }

    setIsConnecting(true)
    setError(null)

    clientRef.current = createWebSocketClient({
      sessionId,
      maxReconnectAttempts,
      onMessage: (msg) => {
        setMessages((prev) => [...prev, msg])
        onMessage?.(msg)
      },
      onThought,
      onToolCall,
      onToolResult,
      onChart,
      onAnswer,
      onError: (msg) => {
        setError(msg)
        onError?.(msg)
      },
      onDone: () => {
        onDone?.()
      },
      onConnect: () => {
        setIsConnected(true)
        setIsConnecting(false)
        setIsReconnecting(false)
        setReconnectAttempts(0)
        onConnect?.()
      },
      onDisconnect: () => {
        setIsConnected(false)
        setIsReconnecting(true)
        onDisconnect?.()
      },
    })
  }, [
    sessionId,
    maxReconnectAttempts,
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
  ])

  useEffect(() => {
    connect()

    return () => {
      clientRef.current?.close()
    }
  }, [connect])

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
