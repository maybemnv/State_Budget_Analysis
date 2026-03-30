import json
import uuid
import io
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from cachetools import LRUCache
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import get_db as get_db_session, get_cache, get_minio, get_redis
from .db.models import Session as SessionModel
from .logger import get_logger

logger = get_logger(__name__)

_memory_cache: LRUCache[str, pd.DataFrame] = LRUCache(maxsize=10)


def _resolve_id(raw: Optional[str]) -> str:
    if not raw:
        return ""
    s = raw.strip()
    if s.startswith("{"):
        try:
            import json
            return str(json.loads(s).get("session_id", s))
        except (json.JSONDecodeError, AttributeError):
            pass
    return s


def _build_schema(df: pd.DataFrame, filename: str) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    return {
        "filename": filename,
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": int(df.isnull().sum().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


class SessionManager:
    def __init__(self):
        self._cache = get_cache()
        self._minio = get_minio()
        self._memory = _memory_cache

    async def get_session(self, session_id: str, db: AsyncSession) -> Optional[dict]:
        resolved_id = _resolve_id(session_id)

        cache_key = f"session:{resolved_id}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Session cache hit: {resolved_id}")
            return cached

        result = await db.execute(
            select(SessionModel).where(SessionModel.session_id == resolved_id)
        )
        session_record = result.scalar_one_or_none()

        if session_record is None:
            return None

        if session_record.expires_at is not None and session_record.expires_at < datetime.utcnow():
            logger.warning(f"Session expired: {resolved_id}")
            return None

        session_data = {
            "session_id": session_record.session_id,
            "filename": session_record.filename,
            "file_path": session_record.file_path,
            "schema": session_record.schema,
            "row_count": session_record.schema.get("shape", [0, 0])[0] if session_record.schema else 0,
        }

        self._cache.set(cache_key, session_data)
        
        redis_client = await get_redis()
        await redis_client.cache_set(f"session:{resolved_id}", session_data, ttl=3600)

        return session_data

    async def load_dataframe(self, session_id: str, db: AsyncSession) -> Optional[pd.DataFrame]:
        resolved_id = _resolve_id(session_id)

        if resolved_id in self._memory:
            logger.debug(f"DataFrame memory cache hit: {resolved_id}")
            return self._memory[resolved_id]

        session = await self.get_session(resolved_id, db)
        if not session:
            return None

        parquet_buffer = await self._minio.download_parquet(resolved_id)
        if parquet_buffer is None:
            return None

        df = pd.read_parquet(parquet_buffer)
        self._memory[resolved_id] = df
        
        logger.debug(f"Loaded DataFrame into memory: {resolved_id}, shape: {df.shape}")
        return df

    async def cache_session(self, session_id: str, data: dict, ttl: int = 3600) -> None:
        redis_client = await get_redis()
        await redis_client.cache_set(f"session:{session_id}", data, ttl)


_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager


async def create_session(df: pd.DataFrame, filename: str, db: AsyncSession) -> str:
    session_id = str(uuid.uuid4())

    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    parquet_size = parquet_buffer.getbuffer().nbytes

    minio_client = get_minio()
    object_path = await minio_client.upload_parquet(session_id, parquet_buffer, parquet_size)

    expires_at = datetime.utcnow() + timedelta(seconds=settings.session_ttl_seconds)

    session_record = SessionModel(
        session_id=session_id,
        filename=filename,
        file_path=object_path,
        schema=_build_schema(df, filename),
        expires_at=expires_at,
    )
    db.add(session_record)
    await db.commit()

    cache = get_cache()
    cache_key = f"session:{session_id}"
    cache.set(cache_key, {"filename": filename, "schema": _build_schema(df, filename), "file_path": object_path})

    redis_client = await get_redis()
    await redis_client.cache_set(
        f"session:{session_id}",
        {"session_id": session_id, "filename": filename, "file_path": object_path, "row_count": len(df)},
        ttl=settings.session_ttl_seconds
    )

    logger.info(f"Created session: {session_id}, file: {object_path}")
    return session_id


async def get_session(session_id: str, db: AsyncSession) -> Optional[dict]:
    return await _session_manager.get_session(session_id, db)


async def get_df(session_id: str, db: AsyncSession) -> Optional[pd.DataFrame]:
    return await _session_manager.load_dataframe(session_id, db)


async def delete_session(session_id: str, db: AsyncSession) -> bool:
    resolved_id = _resolve_id(session_id)

    result = await db.execute(
        select(SessionModel).where(SessionModel.session_id == resolved_id)
    )
    session_record = result.scalar_one_or_none()

    if session_record is None:
        return False

    minio_client = get_minio()
    await minio_client.delete_parquet(resolved_id)

    await db.delete(session_record)
    await db.commit()

    cache = get_cache()
    cache.delete(f"session:{resolved_id}")

    if resolved_id in _memory_cache:
        del _memory_cache[resolved_id]

    redis_client = await get_redis()
    await redis_client.cache_delete(f"session:{resolved_id}")

    logger.info(f"Deleted session: {resolved_id}")
    return True


async def list_sessions(db: AsyncSession) -> list[str]:
    result = await db.execute(
        select(SessionModel.session_id).where(
            SessionModel.expires_at > datetime.utcnow()
        )
    )
    return [row[0] for row in result.all()]


async def refresh_session_ttl(session_id: str, db: AsyncSession) -> bool:
    resolved_id = _resolve_id(session_id)

    result = await db.execute(
        select(SessionModel).where(SessionModel.session_id == resolved_id)
    )
    session_record = result.scalar_one_or_none()

    if session_record is None:
        return False

    session_record.expires_at = datetime.utcnow() + timedelta(seconds=settings.session_ttl_seconds)
    await db.commit()

    return True
