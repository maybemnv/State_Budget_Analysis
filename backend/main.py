from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.upload import router as upload_router
from .routes.chat import router as chat_router
from .session import list_sessions

_VERSION = "2.0.0"

app = FastAPI(
    title="DataLens AI",
    version=_VERSION,
    description="Autonomous Data Analysis Platform — FastAPI backend",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for frontend integration
# In development, allow localhost:3000
# In production, configure with your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "*",  # Allow all for development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "version": _VERSION}


@app.get("/sessions")
def get_all_sessions() -> dict:
    """List all active sessions (for debugging)."""
    session_ids = list_sessions()
    return {
        "count": len(session_ids),
        "sessions": session_ids,
    }


@app.get("/")
def root() -> dict:
    """Root endpoint with API info."""
    return {
        "name": "DataLens AI API",
        "version": _VERSION,
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload",
            "session_info": "GET /sessions/{session_id}",
            "delete_session": "DELETE /sessions/{session_id}",
            "chat_http": "POST /chat/{session_id}",
            "chat_ws": "WS /ws/{session_id}",
            "health": "GET /health",
        }
    }
