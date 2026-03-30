import json
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from ..config import settings
from ..logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    def __init__(self, client: Redis):
        self.client = client

    async def ping(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        await self.client.close()

    async def register_ws(self, session_id: str, ws_id: str, ttl: int = 3600) -> None:
        await self.client.setex(f"ws:{session_id}", ttl, ws_id)

    async def unregister_ws(self, session_id: str) -> None:
        await self.client.delete(f"ws:{session_id}")

    async def get_ws(self, session_id: str) -> Optional[str]:
        return await self.client.get(f"ws:{session_id}")

    async def set_streaming_state(
        self, session_id: str, state: dict[str, Any], ttl: int = 300
    ) -> None:
        key = f"agent_stream:{session_id}"
        await self.client.hset(key, mapping=state)
        await self.client.expire(key, ttl)

    async def get_streaming_state(
        self, session_id: str
    ) -> Optional[dict[str, Any]]:
        key = f"agent_stream:{session_id}"
        data = await self.client.hgetall(key)
        if data:
            return {k.decode(): v.decode() if isinstance(v, bytes) else v for k, v in data.items()}
        return None

    async def clear_streaming_state(self, session_id: str) -> None:
        await self.client.delete(f"agent_stream:{session_id}")

    async def cache_set(self, key: str, value: Any, ttl: int = 3600) -> None:
        serialized = json.dumps(value)
        await self.client.setex(f"cache:{key}", ttl, serialized)

    async def cache_get(self, key: str) -> Optional[Any]:
        data = await self.client.get(f"cache:{key}")
        if data:
            return json.loads(data)
        return None

    async def cache_delete(self, key: str) -> None:
        await self.client.delete(f"cache:{key}")

    async def rate_limit(
        self, key: str, max_calls: int, window_seconds: int
    ) -> tuple[bool, int]:
        current = await self.client.incr(f"ratelimit:{key}")
        if current == 1:
            await self.client.expire(f"ratelimit:{key}", window_seconds)
        remaining = max(0, max_calls - current)
        return current <= max_calls, remaining


_redis_client: Optional[RedisClient] = None


async def get_redis() -> RedisClient:
    global _redis_client
    if _redis_client is None:
        client = redis.from_url(settings.redis_url)
        _redis_client = RedisClient(client)
    return _redis_client


async def close_redis() -> None:
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
