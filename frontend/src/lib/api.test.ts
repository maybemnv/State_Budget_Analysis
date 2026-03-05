import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock global fetch
const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

import { api, createWebSocketClient, API_BASE_URL, WS_BASE_URL } from '@/lib/api'

describe('api', () => {
  beforeEach(() => {
    mockFetch.mockClear()
  })

  describe('upload', () => {
    it('uploads a file successfully', async () => {
      const mockResponse = {
        session_id: 'test-123',
        filename: 'test.csv',
        rows: 100,
        columns: 5,
        column_names: ['a', 'b', 'c', 'd', 'e'],
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })

      const file = new File(['test'], 'test.csv', { type: 'text/csv' })
      const result = await api.upload(file)

      expect(result).toEqual(mockResponse)
      expect(mockFetch).toHaveBeenCalledWith(
        `${API_BASE_URL}/upload`,
        expect.objectContaining({
          method: 'POST',
        })
      )
    })

    it('throws error on failed upload', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ detail: 'Invalid file' }),
      })

      const file = new File(['test'], 'test.csv', { type: 'text/csv' })
      await expect(api.upload(file)).rejects.toThrow('Invalid file')
    })
  })

  describe('getSessionInfo', () => {
    it('fetches session info successfully', async () => {
      const mockResponse = {
        session_id: 'test-123',
        filename: 'test.csv',
        shape: [100, 5],
        columns: ['a', 'b', 'c'],
        dtypes: { a: 'float64', b: 'object', c: 'int64' },
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })

      const result = await api.getSessionInfo('test-123')
      expect(result).toEqual(mockResponse)
    })

    it('throws error when session not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      })

      await expect(api.getSessionInfo('nonexistent')).rejects.toThrow('Session not found')
    })
  })

  describe('deleteSession', () => {
    it('deletes session successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
      })

      await expect(api.deleteSession('test-123')).resolves.not.toThrow()
    })

    it('handles 404 gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      })

      await expect(api.deleteSession('nonexistent')).resolves.not.toThrow()
    })
  })

  describe('chat', () => {
    it('sends chat message successfully', async () => {
      const mockResponse = {
        answer: 'Test answer',
        has_error: false,
        chart_spec: null,
        steps: [],
      }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })

      const result = await api.chat('test-123', 'Hello')
      expect(result).toEqual(mockResponse)
    })

    it('throws error on chat failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      })

      await expect(api.chat('nonexistent', 'Hello')).rejects.toThrow('Session not found')
    })
  })

  describe('health', () => {
    it('returns health status', async () => {
      const mockResponse = { status: 'ok', version: '1.0.0' }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      })

      const result = await api.health()
      expect(result).toEqual(mockResponse)
    })

    it('throws error on health check failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
      })

      await expect(api.health()).rejects.toThrow('Health check failed')
    })
  })
})

describe('WebSocket client', () => {
  let mockWsInstance: {
    send: ReturnType<typeof vi.fn>
    close: ReturnType<typeof vi.fn>
    onopen: (() => void) | null
    onclose: (() => void) | null
    onmessage: ((event: { data: string }) => void) | null
    onerror: ((error: Event) => void) | null
    readyState: number
  }

  beforeEach(() => {
    mockWsInstance = {
      send: vi.fn(),
      close: vi.fn(),
      onopen: null,
      onclose: null,
      onmessage: null,
      onerror: null,
      readyState: 1, // WebSocket.OPEN
    }

    const MockWs = vi.fn(() => mockWsInstance)
    MockWs.OPEN = 1
    MockWs.CONNECTING = 0
    MockWs.CLOSING = 2
    MockWs.CLOSED = 3
    vi.stubGlobal('WebSocket', MockWs)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('creates WebSocket connection with correct URL', () => {
    createWebSocketClient({
      sessionId: 'test-123',
      onConnect: vi.fn(),
    })

    expect(WebSocket).toHaveBeenCalledWith(`${WS_BASE_URL}/ws/test-123`)
  })

  it('calls onConnect callback when connection opens', () => {
    const onConnect = vi.fn()
    createWebSocketClient({
      sessionId: 'test-123',
      onConnect,
    })

    mockWsInstance.onopen?.()

    expect(onConnect).toHaveBeenCalled()
  })

  it('sends message correctly when WebSocket is open', () => {
    // Create client and set readyState to OPEN
    const client = createWebSocketClient({
      sessionId: 'test-123',
    })
    
    // Verify WebSocket is created in OPEN state
    expect(mockWsInstance.readyState).toBe(1) // WebSocket.OPEN
    
    client.send('Hello world')

    expect(mockWsInstance.send).toHaveBeenCalledWith(
      JSON.stringify({ message: 'Hello world' })
    )
  })

  it('parses and dispatches message by type', () => {
    const onThought = vi.fn()
    const onChart = vi.fn()
    const onAnswer = vi.fn()

    createWebSocketClient({
      sessionId: 'test-123',
      onThought,
      onChart,
      onAnswer,
    })

    mockWsInstance.onmessage?.({
      data: JSON.stringify({ type: 'thought', content: 'Thinking...' }),
    })

    mockWsInstance.onmessage?.({
      data: JSON.stringify({ type: 'chart', spec: { mark: 'bar' } }),
    })

    mockWsInstance.onmessage?.({
      data: JSON.stringify({ type: 'answer', content: 'Final answer' }),
    })

    expect(onThought).toHaveBeenCalledWith('Thinking...')
    expect(onChart).toHaveBeenCalledWith({ mark: 'bar' })
    expect(onAnswer).toHaveBeenCalledWith('Final answer')
  })

  it('handles error messages', () => {
    const onError = vi.fn()

    createWebSocketClient({
      sessionId: 'test-123',
      onError,
    })

    mockWsInstance.onmessage?.({
      data: JSON.stringify({ type: 'error', message: 'Something went wrong' }),
    })

    expect(onError).toHaveBeenCalledWith('Something went wrong')
  })

  it('calls onDone when done message received', () => {
    const onDone = vi.fn()

    createWebSocketClient({
      sessionId: 'test-123',
      onDone,
    })

    mockWsInstance.onmessage?.({
      data: JSON.stringify({ type: 'done' }),
    })

    expect(onDone).toHaveBeenCalled()
  })

  it('closes connection on client.close()', () => {
    const client = createWebSocketClient({
      sessionId: 'test-123',
    })

    client.close()

    expect(mockWsInstance.close).toHaveBeenCalled()
  })
})
