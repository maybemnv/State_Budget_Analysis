import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from backend.session import (
    create_session,
    get_session,
    get_df,
    delete_session,
    list_sessions,
    _resolve_id,
    _build_schema,
)
from backend import session as session_module


@pytest.fixture(autouse=True)
def clean_sessions():
    session_module._memory_cache.clear()
    yield
    session_module._memory_cache.clear()


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=50, freq="ME").astype(str),
            "revenue": rng.normal(1000, 100, 50).round(2),
            "category": rng.choice(["A", "B", "C"], 50),
        }
    )


class TestResolveId:
    def test_resolve_id_plain_string(self):
        assert _resolve_id("abc-123") == "abc-123"

    def test_resolve_id_json_string(self):
        assert _resolve_id('{"session_id": "abc-123"}') == "abc-123"

    def test_resolve_id_none(self):
        assert _resolve_id(None) == ""

    def test_resolve_id_empty_string(self):
        assert _resolve_id("") == ""

    def test_resolve_id_invalid_json(self):
        assert _resolve_id("{invalid") == "{invalid"


class TestBuildSchema:
    def test_build_schema_basic(self, sample_df):
        schema = _build_schema(sample_df, "test.csv")
        assert schema["filename"] == "test.csv"
        assert schema["shape"] == [50, 3]
        assert schema["columns"] == ["date", "revenue", "category"]
        assert "revenue" in schema["numeric_columns"]
        assert "category" in schema["categorical_columns"]

    def test_build_schema_no_numeric_columns(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
        schema = _build_schema(df, "test.csv")
        assert schema["numeric_columns"] == []
        assert schema["categorical_columns"] == ["a", "b"]

    def test_build_schema_missing_values(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        schema = _build_schema(df, "test.csv")
        assert schema["missing_values"] == 2

    def test_build_schema_dtypes(self, sample_df):
        schema = _build_schema(sample_df, "test.csv")
        assert "revenue" in schema["dtypes"]
        assert "date" in schema["dtypes"]
        assert "category" in schema["dtypes"]


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_create_session(self, sample_df):
        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()

        with patch("backend.session._store_df_in_redis", new_callable=AsyncMock), \
             patch("backend.session.get_cache") as mock_cache, \
             patch("backend.session.get_redis") as mock_redis:
            mock_cache.return_value.set = MagicMock()
            mock_redis_instance = AsyncMock()
            mock_redis_instance.cache_set = AsyncMock()
            mock_redis.return_value = mock_redis_instance

            session_id = await create_session(sample_df, "test.csv", mock_db)
            assert session_id is not None
            assert len(session_id) == 36

    @pytest.mark.asyncio
    async def test_get_session_not_exists(self):
        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("backend.session.get_cache") as mock_cache:
            mock_cache.return_value.get = MagicMock(return_value=None)
            session = await get_session("nonexistent-id", mock_db)
            assert session is None

    @pytest.mark.asyncio
    async def test_delete_session_not_exists(self):
        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await delete_session("nonexistent-id", mock_db)
        assert result is False

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self):
        mock_db = AsyncMock()
        mock_result = AsyncMock()
        mock_result.all = MagicMock(return_value=[])
        mock_db.execute = AsyncMock(return_value=mock_result)

        sessions = await list_sessions(mock_db)
        assert sessions == []
