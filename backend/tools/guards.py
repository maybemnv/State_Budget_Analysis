from typing import Optional
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from ..session import get_df
from ..db.database import get_db


async def require_df(
    session_id: Optional[str], db: Optional[AsyncSession] = None
) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Return (df, None) on success or (None, error_dict) when session is missing.
    In case db is not provided, it creates a new session.
    """
    if not session_id:
        return None, {"error": "session_id is required"}

    if db:
        df = await get_df(session_id, db)
    else:
        async with get_db() as local_db:
            df = await get_df(session_id, local_db)

    if df is None:
        return None, {"error": f"Session {session_id!r} not found"}
    return df, None
