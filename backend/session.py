import uuid
from typing import Optional
import pandas as pd


# In-memory session store: session_id → {df, filename, metadata}
_sessions: dict[str, dict] = {}


def create_session(df: pd.DataFrame, filename: str) -> str:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "df": df,
        "filename": filename,
        "metadata": _build_metadata(df, filename),
    }
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    return _sessions.get(session_id)


def get_df(session_id: str) -> Optional[pd.DataFrame]:
    session = _sessions.get(session_id)
    return session["df"] if session else None


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
