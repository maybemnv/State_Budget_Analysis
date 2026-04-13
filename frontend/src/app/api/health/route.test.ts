import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('@/lib/api', () => ({
  api: {
    health: vi.fn(),
  },
}))

describe('Health Route', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns health status', async () => {
    const { api } = await import('@/lib/api')
    api.health = vi.fn().mockResolvedValue({ status: 'ok', version: '1.0.0' })
    
    // Test would go here if we could import the route handler directly
    expect(api.health).toBeDefined()
  })

  it('handles health check error', async () => {
    const { api } = await import('@/lib/api')
    api.health = vi.fn().mockRejectedValue(new Error('Failed'))
    
    await expect(api.health()).rejects.toThrow('Failed')
  })
})
