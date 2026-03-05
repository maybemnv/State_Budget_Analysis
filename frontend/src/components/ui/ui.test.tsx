import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

describe('Button', () => {
  it('renders button with text', () => {
    render(<Button>Click me</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()
    expect(screen.getByText('Click me')).toBeInTheDocument()
  })

  it('handles click events', () => {
    const handleClick = vi.fn()
    render(<Button onClick={handleClick}>Click me</Button>)
    screen.getByRole('button').click()
    expect(handleClick).toHaveBeenCalled()
  })

  it('renders different variants', () => {
    const { rerender } = render(<Button variant="default">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<Button variant="destructive">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<Button variant="outline">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<Button variant="secondary">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<Button variant="ghost">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<Button variant="link">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('renders different sizes', () => {
    const { rerender } = render(<Button size="default">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<Button size="sm">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<Button size="lg">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()

    rerender(<Button size="icon">Test</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('can be disabled', () => {
    render(<Button disabled>Disabled</Button>)
    expect(screen.getByRole('button')).toBeDisabled()
  })
})

describe('Input', () => {
  it('renders input element', () => {
    render(<Input />)
    expect(screen.getByRole('textbox')).toBeInTheDocument()
  })

  it('handles value changes', () => {
    const handleChange = vi.fn()
    render(<Input onChange={handleChange} />)
    const input = screen.getByRole('textbox')
    
    input.focus()
    input.setValue('test value')
    
    expect(handleChange).toHaveBeenCalled()
  })

  it('accepts placeholder', () => {
    render(<Input placeholder="Enter text..." />)
    expect(screen.getByPlaceholderText('Enter text...')).toBeInTheDocument()
  })

  it('accepts type prop', () => {
    render(<Input type="password" />)
    expect(screen.getByRole('textbox')).toHaveAttribute('type', 'password')
  })
})

describe('Card', () => {
  it('renders card component', () => {
    render(<Card>Card content</Card>)
    expect(screen.getByText('Card content')).toBeInTheDocument()
  })

  it('renders with title', () => {
    render(<Card title="Card Title">Content</Card>)
    expect(screen.getByText('Card Title')).toBeInTheDescription()
  })
})

describe('Badge', () => {
  it('renders badge with text', () => {
    render(<Badge>Status</Badge>)
    expect(screen.getByText('Status')).toBeInTheDocument()
  })

  it('renders different variants', () => {
    const { rerender } = render(<Badge variant="default">Test</Badge>)
    expect(screen.getByText('Test')).toBeInTheDocument()

    rerender(<Badge variant="secondary">Test</Badge>)
    expect(screen.getByText('Test')).toBeInTheDocument()

    rerender(<Badge variant="destructive">Test</Badge>)
    expect(screen.getByText('Test')).toBeInTheDocument()

    rerender(<Badge variant="outline">Test</Badge>)
    expect(screen.getByText('Test')).toBeInTheDocument()
  })
})
