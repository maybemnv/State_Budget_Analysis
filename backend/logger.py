import logging
import sys
from typing import Any

from .config import settings


def setup_logging() -> None:
    """Configure application-wide logging."""
    log_level = logging.DEBUG if settings.debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> logging.Logger:
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return logging.getLogger(name)


def log_request(method: str, path: str, status: int, duration_ms: float) -> None:
    """Log HTTP request details."""
    get_logger("api").debug(
        "%s %s - %d (%.2fms)",
        method,
        path,
        status,
        duration_ms,
    )


def log_websocket_event(
    event_type: str, session_id: str, details: dict[str, Any] | None = None
) -> None:
    """Log WebSocket event."""
    msg = f"WebSocket [{event_type}] session={session_id}"
    if details:
        msg += f" | {details}"
    get_logger("websocket").debug(msg)


def log_tool_execution(
    tool_name: str, session_id: str, duration_ms: float, success: bool
) -> None:
    """Log tool execution."""
    status = "OK" if success else "FAIL"
    get_logger("tools").debug(
        "Tool '%s' | session=%s | %s | %.2fms",
        tool_name,
        session_id,
        status,
        duration_ms,
    )


def log_agent_error(session_id: str, error: Exception) -> None:
    """Log agent error with context."""
    get_logger("agent").error(
        "Agent error | session=%s | %s: %s",
        session_id,
        type(error).__name__,
        str(error),
        exc_info=True,
    )
