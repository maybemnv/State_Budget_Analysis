import { describe, it, expect } from 'vitest'
import { cn } from '@/lib/utils'

describe('utils', () => {
  describe('cn', () => {
    it('merges class names correctly', () => {
      const result = cn('foo', 'bar')
      expect(result).toBe('foo bar')
    })

    it('handles conditional classes', () => {
      const result = cn('foo', false && 'bar', 'baz')
      expect(result).toBe('foo baz')
    })

    it('handles array inputs', () => {
      const result = cn(['foo', 'bar'], 'baz')
      expect(result).toBe('foo bar baz')
    })

    it('handles empty inputs', () => {
      const result = cn()
      expect(result).toBe('')
    })

    it('merges tailwind classes with conflicts', () => {
      const result = cn('p-4 p-6', 'text-red-500 text-blue-500')
      // twMerge should resolve conflicts, keeping the last value
      expect(result).toContain('p-6')
      expect(result).toContain('text-blue-500')
    })
  })
})
