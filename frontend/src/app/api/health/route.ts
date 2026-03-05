// Health check API route for frontend
import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, {
      cache: 'no-store',
    })
    
    if (!res.ok) {
      return NextResponse.json(
        { status: 'error', message: 'Backend health check failed' },
        { status: 503 }
      )
    }
    
    const data = await res.json()
    return NextResponse.json({
      status: 'ok',
      backend: data,
    })
  } catch (error) {
    return NextResponse.json(
      { 
        status: 'error', 
        message: 'Backend unreachable',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 503 }
    )
  }
}
