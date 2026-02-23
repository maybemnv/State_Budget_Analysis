from typing import Optional
import pandas as pd
from ..session import get_df


def require_df(session_id: Optional[str]) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Return (df, None) on success or (None, error_dict) when session is missing."""
    if not session_id:
        return None, {"error": "session_id is required"}
    df = get_df(session_id)
    if df is None:
        return None, {"error": f"Session {session_id!r} not found"}
    return df, None
