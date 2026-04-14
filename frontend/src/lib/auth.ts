// Auth API client + token helpers

const getOrigin = () => {
  if (typeof window === 'undefined') return 'http://localhost:8000'
  return `${window.location.protocol}//${window.location.hostname}:8000`
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || getOrigin()
const TOKEN_KEY = 'datalens_token'

export interface User {
  id: number
  email: string
  created_at: string | null
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export function getToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(TOKEN_KEY)
}

function authHeaders(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = { 'Content-Type': 'application/json', ...authHeaders(), ...init?.headers }
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export const authApi = {
  async register(email: string, password: string): Promise<AuthResponse> {
    return request('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })
  },

  async login(email: string, password: string): Promise<AuthResponse> {
    return request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })
  },

  async me(): Promise<User> {
    return request('/auth/me')
  },

  async getSessions(): Promise<Array<{
    session_id: string
    filename: string
    shape: [number, number]
    created_at: string
    expires_at: string | null
  }>> {
    return request('/auth/sessions')
  },

  async deleteSession(sessionId: string): Promise<void> {
    await request(`/sessions/${sessionId}`, { method: 'DELETE' })
  },
}
