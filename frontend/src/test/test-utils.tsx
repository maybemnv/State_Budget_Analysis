import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// Mock WebSocket for testing
class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  
  readyState = MockWebSocket.OPEN
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: ((error: Event) => void) | null = null
  
  constructor(public url: string) {
    setTimeout(() => this.onopen?.(), 0)
  }
  
  send(data: string) {}
  close() {}
}

vi.stubGlobal('WebSocket', MockWebSocket)

// Test utilities
export * from '@testing-library/react'
export { render, screen, waitFor, userEvent }

// Test data factories
export const createMockSessionInfo = (overrides = {}) => ({
  session_id: 'test-session-123',
  filename: 'test.csv',
  shape: [100, 5] as [number, number],
  columns: ['date', 'revenue', 'category', 'amount', 'status'],
  dtypes: {
    date: 'object',
    revenue: 'float64',
    category: 'object',
    amount: 'float64',
    status: 'object',
  },
  numeric_columns: ['revenue', 'amount'],
  categorical_columns: ['date', 'category', 'status'],
  missing_values: 0,
  ...overrides,
})

export const createMockUploadResponse = (overrides = {}) => ({
  session_id: 'test-session-123',
  filename: 'test.csv',
  rows: 100,
  columns: 5,
  column_names: ['date', 'revenue', 'category', 'amount', 'status'],
  ...overrides,
})

export const createMockChatResponse = (overrides = {}) => ({
  answer: 'The average revenue is $1,000.50',
  chart_spec: null,
  has_error: false,
  steps: [
    {
      tool: 'describe_dataset',
      args: { session_id: 'test-session-123' },
      result: { columns: ['date', 'revenue'] },
    },
  ],
  ...overrides,
})

export const createMockWSMessage = (type: string, data = {}) => ({
  type,
  ...data,
})
