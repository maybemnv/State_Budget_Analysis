import io
import pytest
import pandas as pd
import numpy as np
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from backend.main import app
from backend.session import create_session
from backend import session as session_module


@pytest.fixture(autouse=True)
def clean_sessions():
    """Clean up sessions before and after each test."""
    session_module._sessions.clear()
    yield
    session_module._sessions.clear()


@pytest.fixture
def csv_bytes() -> bytes:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=50, freq="ME").astype(str),
            "revenue": rng.normal(1000, 100, 50).round(2),
            "category": rng.choice(["A", "B", "C"], 50),
        }
    )
    return df.to_csv(index=False).encode()


@pytest.fixture
def excel_bytes() -> bytes:
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "value": rng.normal(50, 20, 20),
            "label": rng.choice(["X", "Y"], 20),
        }
    )
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
def sample_session(csv_bytes):
    session_id = create_session(pd.read_csv(io.BytesIO(csv_bytes)), "test.csv")
    return session_id


class TestUploadEndpoint:
    """Test suite for file upload endpoint"""

    async def test_upload_csv(self, client, csv_bytes):
        r = await client.post(
            "/upload",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "session_id" in body
        assert body["rows"] == 50
        assert "revenue" in body["column_names"]
        assert "category" in body["column_names"]

    async def test_upload_excel(self, client, excel_bytes):
        r = await client.post(
            "/upload",
            files={
                "file": (
                    "test.xlsx",
                    io.BytesIO(excel_bytes),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "session_id" in body
        assert body["rows"] == 20

    async def test_upload_unsupported_type(self, client):
        r = await client.post(
            "/upload",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert r.status_code == 422

    async def test_upload_file_too_large(self, client):
        big = b"a,b\n" + b"1,2\n" * 10_000_000
        r = await client.post(
            "/upload",
            files={"file": ("big.csv", io.BytesIO(big), "text/csv")},
        )
        assert r.status_code in {200, 413}

    async def test_upload_empty_file(self, client):
        r = await client.post(
            "/upload",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        assert r.status_code == 422


class TestSessionEndpoint:
    """Test suite for session info endpoints"""

    async def test_get_session_info(self, client, csv_bytes):
        upload = await client.post(
            "/upload",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        session_id = upload.json()["session_id"]

        r = await client.get(f"/sessions/{session_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == session_id
        assert "columns" in body
        assert "dtypes" in body
        assert "categorical_columns" in body

    async def test_get_session_not_found(self, client):
        r = await client.get("/sessions/nonexistent-id")
        assert r.status_code == 404

    async def test_delete_session(self, client, sample_session):
        r = await client.delete(f"/sessions/{sample_session}")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"

    async def test_delete_session_not_found(self, client):
        r = await client.delete("/sessions/nonexistent-id")
        assert r.status_code == 404


class TestChatEndpoint:
    """Test suite for chat endpoints"""

    async def test_chat_session_not_found(self, client):
        r = await client.post(
            "/chat/nonexistent-session",
            json={"message": "Hello"},
        )
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_chat_with_mock_agent(self, client, sample_session):
        with patch("backend.routes.chat.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "output": "The average revenue is 1000.50",
                "intermediate_steps": [],
            }
            r = await client.post(
                f"/chat/{sample_session}",
                json={"message": "What is the average revenue?"},
            )
            assert r.status_code == 200
            body = r.json()
            assert "answer" in body

    @pytest.mark.asyncio
    async def test_chat_with_chart(self, client, sample_session):
        chart_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": "bar",
        }
        with patch("backend.routes.chat.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {
                "output": f"```json\n{chart_spec}\n```\nHere is the chart",
                "intermediate_steps": [],
            }
            r = await client.post(
                f"/chat/{sample_session}",
                json={"message": "Show me a chart"},
            )
            assert r.status_code == 200
            body = r.json()
            assert "chart_spec" in body


class TestHealthEndpoint:
    """Test suite for health check endpoint"""

    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    async def test_health_includes_version(self, client):
        r = await client.get("/health")
        body = r.json()
        assert "version" in body or "status" in body
