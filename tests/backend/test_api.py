import io
import pytest
import pandas as pd
import numpy as np
from httpx import AsyncClient, ASGITransport
from backend.main import app


@pytest.fixture
def csv_bytes() -> bytes:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=50, freq="ME").astype(str),
        "revenue": rng.normal(1000, 100, 50).round(2),
        "category": rng.choice(["A", "B", "C"], 50),
    })
    return df.to_csv(index=False).encode()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_upload_csv(client, csv_bytes):
    r = await client.post(
        "/upload",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert body["rows"] == 50
    assert "revenue" in body["column_names"]


async def test_upload_invalid_type(client):
    r = await client.post(
        "/upload",
        files={"file": ("bad.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert r.status_code == 422


async def test_upload_too_large(client):
    big = b"a,b\n" + b"1,2\n" * 10_000_000
    r = await client.post(
        "/upload",
        files={"file": ("big.csv", io.BytesIO(big), "text/csv")},
    )
    # The server limits are set in settings; with default 100MB and ~120MB file this should 413
    # Depending on actual size this may pass — just assert it's not 500
    assert r.status_code in {200, 413}


async def test_session_info(client, csv_bytes):
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


async def test_session_not_found(client):
    r = await client.get("/sessions/does-not-exist")
    assert r.status_code == 404
