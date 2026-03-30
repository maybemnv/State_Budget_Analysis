import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .db import close_redis, init_db
from .logger import get_logger, log_request, setup_logging
from .routes.chat import router as chat_router
from .routes.upload import router as upload_router
from .session import list_sessions
from .tasks.cleanup import cleanup_expired_sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    setup_logging()
    logger = get_logger(__name__)
    logger.info("=" * 50)
    logger.info("DataLens AI Backend Starting")
    logger.info(f"Version: {settings.model_name}")
    logger.info(f"Max upload size: {settings.max_upload_mb}MB")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info("=" * 50)

    await init_db()
    logger.info("Database initialized")


    try:
        deleted = await cleanup_expired_sessions()
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired sessions on startup")
    except Exception as e:
        logger.warning(f"Startup cleanup failed (non-fatal): {e}")

    yield

    await close_redis()
    logger.info("DataLens AI Backend Shutting Down")


app = FastAPI(
    title="DataLens AI",
    version="2.0.0",
    description="Autonomous Data Analysis Platform — FastAPI backend",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

logger = get_logger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing."""
    import time

    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    log_request(request.method, request.url.path, response.status_code, duration)
    return response


app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


@app.get("/sessions")
async def get_all_sessions() -> dict:
    """List all active sessions (for debugging)."""
    from .db import get_db

    async with get_db() as db:
        session_ids = await list_sessions(db)
        return {
            "count": len(session_ids),
            "sessions": session_ids,
        }


@app.get("/")
async def root() -> dict:
    """Root endpoint with API info."""
    return {
        "name": "DataLens AI API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload",
            "session_info": "GET /sessions/{session_id}",
            "delete_session": "DELETE /sessions/{session_id}",
            "chat_http": "POST /chat/{session_id}",
            "chat_ws": "WS /ws/{session_id}",
            "health": "GET /health",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
