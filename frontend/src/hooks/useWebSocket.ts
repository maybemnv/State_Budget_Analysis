"use client"

import { useEffect, useRef, useCallback, useState } from "react"
import { createWebSocketClient, type WSMessage, type UseWebSocketOptions as ApiOptions } from "@/lib/api"

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
}

export interface UseWebSocketReturn {
  sendMessage: (message: string | Record<string, unknown>) => void
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  messages: WSMessage[]
}

/**
 * useWebSocket — Hook for real-time communication with the DataLens backend.
 * Uses WebSocket streaming for agent responses.
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
}: UseWebSocketOptions): UseWebSocketReturn {
  const clientRef = useRef<ReturnType<typeof createWebSocketClient> | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(true)
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
        onConnect?.()
      },
      onDisconnect: () => {
        setIsConnected(false)
        setIsConnecting(false)
        onDisconnect?.()
      },
    })
  }, [sessionId, onMessage, onThought, onToolCall, onToolResult, onChart, onAnswer, onError, onDone, onConnect, onDisconnect])

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

  return {
    sendMessage,
    isConnected,
    isConnecting,
    error,
    messages,
  }
}
