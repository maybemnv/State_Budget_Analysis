import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { AgentAvatar } from '@/components/agent/AgentAvatar'
import { AgentChat } from '@/components/agent/AgentChat'
import { ThoughtStep } from '@/components/agent/ThoughtStep'
import { ToolCallCard } from '@/components/agent/ToolCallCard'

describe('AgentAvatar', () => {
  it('renders idle state correctly', () => {
    render(<AgentAvatar state="idle" />)
    expect(screen.getByTestId('agent-avatar')).toBeInTheDocument()
  })

  it('renders thinking state correctly', () => {
    render(<AgentAvatar state="thinking" />)
    expect(screen.getByTestId('agent-avatar')).toBeInTheDocument()
  })

  it('renders executing state correctly', () => {
    render(<AgentAvatar state="executing" />)
    expect(screen.getByTestId('agent-avatar')).toBeInTheDocument()
  })

  it('renders error state correctly', () => {
    render(<AgentAvatar state="error" />)
    expect(screen.getByTestId('agent-avatar')).toBeInTheDocument()
  })
})

describe('ThoughtStep', () => {
  it('renders thought content', () => {
    render(<ThoughtStep content="Analyzing the data..." />)
    expect(screen.getByText('Analyzing the data...')).toBeInTheDocument()
  })

  it('renders with different statuses', () => {
    const { rerender } = render(<ThoughtStep status="pending" content="Test" />)
    expect(screen.getByTestId('thought-step')).toBeInTheDocument()

    rerender(<ThoughtStep status="active" content="Test" />)
    expect(screen.getByTestId('thought-step')).toBeInTheDocument()

    rerender(<ThoughtStep status="completed" content="Test" />)
    expect(screen.getByTestId('thought-step')).toBeInTheDocument()

    rerender(<ThoughtStep status="error" content="Test" />)
    expect(screen.getByTestId('thought-step')).toBeInTheDocument()
  })
})

describe('ToolCallCard', () => {
  const mockToolCall = {
    tool: 'describe_dataset',
    args: { session_id: 'test-123' },
    result: { columns: ['a', 'b', 'c'] },
  }

  it('renders tool name and args', () => {
    render(<ToolCallCard tool={mockToolCall.tool} args={mockToolCall.args} />)
    expect(screen.getByText('describe_dataset')).toBeInTheDocument()
  })

  it('renders result when provided', () => {
    render(
      <ToolCallCard
        tool={mockToolCall.tool}
        args={mockToolCall.args}
        result={mockToolCall.result}
      />
    )
    expect(screen.getByText(/columns:/)).toBeInTheDocument()
  })

  it('handles missing result gracefully', () => {
    render(<ToolCallCard tool="test_tool" args={{}} />)
    expect(screen.getByTestId('tool-call-card')).toBeInTheDocument()
  })
})

describe('AgentChat', () => {
  const mockProps = {
    sessionId: 'test-session-123',
    onMessage: vi.fn(),
  }

  beforeEach(() => {
    vi.stubGlobal('WebSocket', vi.fn(() => ({
      readyState: 1,
      send: vi.fn(),
      close: vi.fn(),
      onopen: null,
      onclose: null,
      onmessage: null,
      onerror: null,
    })))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders chat interface', () => {
    render(<AgentChat {...mockProps} />)
    expect(screen.getByTestId('agent-chat')).toBeInTheDocument()
  })

  it('renders message input', () => {
    render(<AgentChat {...mockProps} />)
    expect(screen.getByTestId('message-input')).toBeInTheDocument()
  })

  it('renders send button', () => {
    render(<AgentChat {...mockProps} />)
    expect(screen.getByTestId('send-button')).toBeInTheDocument()
  })
})
