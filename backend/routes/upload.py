from pathlib import Path
import io
import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException
from ..session import create_session, get_session
from ..schemas import UploadResponse
from ..config import settings

router = APIRouter(tags=["upload"])

_ALLOWED = {".csv", ".xlsx", ".xls", ".parquet"}
_MAX_BYTES = settings.max_upload_mb * 1024 * 1024


def _parse_file(filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(content))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(content))
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(content))
    raise ValueError(f"Unsupported file type: {suffix}")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes("object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip().replace(r"^\s*$", pd.NA, regex=True))
    df.dropna(how="all", axis=0, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED:
        raise HTTPException(status_code=422, detail=f"File type {suffix!r} not supported. Use: {_ALLOWED}")

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB limit")

    try:
        df = _parse_file(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    df = _clean(df)
    session_id = create_session(df, file.filename)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        rows=df.shape[0],
        columns=df.shape[1],
        column_names=df.columns.tolist(),
    )


@router.get("/sessions/{session_id}")
def get_session_info(session_id: str) -> dict:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    meta = session["metadata"]
    return {
        "session_id": session_id,
        "filename": meta["filename"],
        "shape": meta["shape"],
        "columns": meta["columns"],
        "dtypes": meta["dtypes"],
    }
