import logging
from pathlib import Path
import io

import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..session import create_session, get_session, delete_session
from ..schemas import UploadResponse, SessionInfo
from ..config import settings


logger = logging.getLogger(__name__)

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
    df[obj_cols] = df[obj_cols].apply(
        lambda s: s.str.strip().replace(r"^\s*$", pd.NA, regex=True)
    )
    df.dropna(how="all", axis=0, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    logger.info(f"Upload request: filename={filename}, size={file.size or 'unknown'}")

    if suffix not in _ALLOWED:
        raise HTTPException(
            status_code=422,
            detail=f"File type {suffix!r} not supported. Allowed: {_ALLOWED}",
        )

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB limit"
        )

    try:
        df = _parse_file(filename, content)
    except Exception as e:
        logger.error(f"Failed to parse file {filename}: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    df = _clean(df)

    session_id = await create_session(df, filename, db)

    logger.info(f"Upload success: session_id={session_id}, rows={df.shape[0]}, cols={df.shape[1]}")

    return UploadResponse(
        session_id=session_id,
        filename=filename,
        rows=df.shape[0],
        columns=df.shape[1],
        column_names=df.columns.tolist(),
    )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    session = await get_session(session_id, db)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    schema = session["schema"]
    return {
        "session_id": session_id,
        "filename": schema["filename"],
        "shape": schema["shape"],
        "columns": schema["columns"],
        "dtypes": schema["dtypes"],
        "numeric_columns": schema.get("numeric_columns", []),
        "categorical_columns": schema.get("categorical_columns", []),
        "missing_values": schema.get("missing_values", 0),
    }


@router.delete("/sessions/{session_id}")
async def delete_session_endpoint(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    session = await get_session(session_id, db)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    await delete_session(session_id, db)
    return {"status": "deleted", "session_id": session_id}
