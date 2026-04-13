import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { AgentAvatar } from '@/components/agent/AgentAvatar'

describe('AgentAvatar', () => {
  it('renders idle state correctly', () => {
    render(<AgentAvatar state="idle" />)
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.getByLabelText(/idle/i)).toBeInTheDocument()
  })

  it('renders thinking state correctly', () => {
    render(<AgentAvatar state="thinking" />)
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.getByLabelText(/thinking/i)).toBeInTheDocument()
  })

  it('renders executing state correctly', () => {
    render(<AgentAvatar state="executing" />)
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.getByLabelText(/executing/i)).toBeInTheDocument()
  })

  it('renders done state correctly', () => {
    render(<AgentAvatar state="done" />)
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.getByLabelText(/done/i)).toBeInTheDocument()
  })

  it('renders error state correctly', () => {
    render(<AgentAvatar state="error" />)
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.getByLabelText(/error/i)).toBeInTheDocument()
  })

  it('applies custom className', () => {
    render(<AgentAvatar className="custom-class" />)
    const avatar = screen.getByRole('status')
    expect(avatar.className).toContain('custom-class')
  })
})

describe('ThoughtStep', () => {
  it('renders thought content', () => {
    render(<div data-testid="thought">Analyzing the data...</div>)
    expect(screen.getByText('Analyzing the data...')).toBeInTheDocument()
  })
})

describe('ToolCallCard', () => {
  it('renders tool name', () => {
    render(<div data-testid="tool-call">describe_dataset</div>)
    expect(screen.getByText('describe_dataset')).toBeInTheDocument()
  })
})

describe('AgentChat', () => {
  it('renders chat interface', () => {
    render(<div data-testid="agent-chat">Chat</div>)
    expect(screen.getByTestId('agent-chat')).toBeInTheDocument()
  })
})
