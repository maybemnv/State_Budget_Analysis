import json
import uuid
from typing import Optional

import pandas as pd

_sessions: dict[str, dict] = {}


def _resolve_id(raw: Optional[str]) -> str:
    """Unwrap a session_id that LangChain may deliver as a JSON-wrapped string.

    Example: '{"session_id": "abc-123"}' → 'abc-123'
    """
    if not raw:
        return ""
    s = raw.strip()
    if s.startswith("{"):
        try:
            return str(json.loads(s).get("session_id", s))
        except (json.JSONDecodeError, AttributeError):
            pass
    return s


def create_session(df: pd.DataFrame, filename: str) -> str:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "df": df,
        "filename": filename,
        "metadata": _build_metadata(df, filename),
    }
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    return _sessions.get(_resolve_id(session_id))


def get_df(session_id: str) -> Optional[pd.DataFrame]:
    session = _sessions.get(_resolve_id(session_id))
    return session["df"] if session else None


def delete_session(session_id: str) -> bool:
    """Delete a session and free memory."""
    resolved_id = _resolve_id(session_id)
    if resolved_id in _sessions:
        del _sessions[resolved_id]
        return True
    return False


def list_sessions() -> list[str]:
    """List all active session IDs."""
    return list(_sessions.keys())


def _build_metadata(df: pd.DataFrame, filename: str) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    return {
        "filename": filename,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": int(df.isnull().sum().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
