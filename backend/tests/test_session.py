import pytest
import pandas as pd
import numpy as np
from backend.session import (
    create_session,
    get_session,
    get_df,
    delete_session,
    list_sessions,
    _resolve_id,
    _build_metadata,
)
from backend import session as session_module


@pytest.fixture(autouse=True)
def clean_sessions():
    """Clean up sessions before and after each test."""
    session_module._sessions.clear()
    yield
    session_module._sessions.clear()


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
    """Test suite for session ID resolution"""

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


class TestSessionManagement:
    """Test suite for session management functions"""

    def test_create_session(self, sample_df):
        session_id = create_session(sample_df, "test.csv")
        assert session_id is not None
        assert len(session_id) == 36

    def test_get_session_exists(self, sample_df):
        session_id = create_session(sample_df, "test.csv")
        session = get_session(session_id)
        assert session is not None
        assert "df" in session
        assert "metadata" in session

    def test_get_session_not_exists(self):
        session = get_session("nonexistent-id")
        assert session is None

    def test_get_df_exists(self, sample_df):
        session_id = create_session(sample_df, "test.csv")
        df = get_df(session_id)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    def test_get_df_not_exists(self):
        df = get_df("nonexistent-id")
        assert df is None

    def test_delete_session_success(self, sample_df):
        session_id = create_session(sample_df, "test.csv")
        result = delete_session(session_id)
        assert result is True
        assert get_session(session_id) is None

    def test_delete_session_not_exists(self):
        result = delete_session("nonexistent-id")
        assert result is False

    def test_list_sessions_empty(self):
        sessions = list_sessions()
        assert sessions == []

    def test_list_sessions_with_data(self, sample_df):
        id1 = create_session(sample_df, "test1.csv")
        id2 = create_session(sample_df, "test2.csv")
        sessions = list_sessions()
        assert len(sessions) == 2
        assert id1 in sessions
        assert id2 in sessions


class TestBuildMetadata:
    """Test suite for metadata building"""

    def test_build_metadata_basic(self, sample_df):
        meta = _build_metadata(sample_df, "test.csv")
        assert meta["filename"] == "test.csv"
        assert meta["shape"] == (50, 3)
        assert meta["columns"] == ["date", "revenue", "category"]
        assert "revenue" in meta["numeric_columns"]
        assert "category" in meta["categorical_columns"]

    def test_build_metadata_no_numeric_columns(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
        meta = _build_metadata(df, "test.csv")
        assert meta["numeric_columns"] == []
        assert meta["categorical_columns"] == ["a", "b"]

    def test_build_metadata_missing_values(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
        meta = _build_metadata(df, "test.csv")
        assert meta["missing_values"] == 2

    def test_build_metadata_dtypes(self, sample_df):
        meta = _build_metadata(sample_df, "test.csv")
        assert "revenue" in meta["dtypes"]
        assert "date" in meta["dtypes"]
        assert "category" in meta["dtypes"]


class TestSessionIntegration:
    """Integration tests for full session lifecycle"""

    def test_full_session_lifecycle(self, sample_df):
        session_id = create_session(sample_df, "data.csv")

        session = get_session(session_id)
        assert session["filename"] == "data.csv"
        assert len(session["df"]) == 50

        df = get_df(session_id)
        assert df is not None

        sessions_before = list_sessions()
        assert len(sessions_before) == 1

        delete_session(session_id)

        sessions_after = list_sessions()
        assert len(sessions_after) == 0

    def test_multiple_sessions(self, sample_df):
        id1 = create_session(sample_df, "file1.csv")
        id2 = create_session(sample_df, "file2.csv")

        assert get_session(id1)["filename"] == "file1.csv"
        assert get_session(id2)["filename"] == "file2.csv"

        delete_session(id1)

        assert get_session(id1) is None
        assert get_session(id2) is not None
