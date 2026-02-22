from typing import Optional
import pandas as pd
from ..session import get_df


def require_df(session_id: str) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Return (df, None) on success or (None, error_dict) when session is missing."""
    df = get_df(session_id)
    if df is None:
        return None, {"error": f"Session {session_id!r} not found"}
    return df, None
