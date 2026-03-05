import json
import uuid
import pickle
import base64
from pathlib import Path
from typing import Optional
import pandas as pd

# Session storage file
SESSIONS_FILE = Path(__file__).parent / ".sessions.json"
_sessions: dict[str, dict] = {}

# Load sessions from disk on startup
def _load_sessions():
    """Load sessions from disk if file exists."""
    global _sessions
    if SESSIONS_FILE.exists():
        try:
            with open(SESSIONS_FILE, 'r') as f:
                data = json.load(f)
                # Convert base64 encoded DataFrames back to pandas
                for session_id, session_data in data.items():
                    if 'df_base64' in session_data:
                        # Decode DataFrame from base64
                        df_bytes = base64.b64decode(session_data['df_base64'])
                        session_data['df'] = pd.read_pickle(pd.io.common.BytesIO(df_bytes))
                        del session_data['df_base64']
                _sessions = data
                print(f"Loaded {len(_sessions)} sessions from disk")
        except Exception as e:
            print(f"Failed to load sessions: {e}")
            _sessions = {}
    else:
        _sessions = {}

# Save sessions to disk
def _save_sessions():
    """Save sessions to disk."""
    try:
        data = {}
        for session_id, session_data in _sessions.items():
            # Create a copy without the DataFrame
            session_copy = session_data.copy()
            if 'df' in session_data:
                # Encode DataFrame as base64
                df_bytes = session_data['df'].to_pickle()
                session_copy['df_base64'] = base64.b64encode(df_bytes).decode('utf-8')
                del session_copy['df']
            data[session_id] = session_copy
        
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Failed to save sessions: {e}")

# Load sessions on module import
_load_sessions()


def _resolve_id(raw: Optional[str]) -> str:
    """Unwrap a session_id that LangChain may deliver as a JSON-wrapped string."""
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
        "created_at": pd.Timestamp.now().isoformat(),
    }
    _save_sessions()  # Persist to disk
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
        _save_sessions()  # Persist to disk
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
