import json
from typing import Any, Optional, TYPE_CHECKING

# Type checking imports - these help the linter understand the types
if TYPE_CHECKING:
    try:
        from upstash_redis.asyncio import Redis as UpstashRedis
    except ImportError:
        import redis.asyncio as redis
        Redis = redis.Redis
else:
    # Runtime imports with fallback
    try:
        from upstash_redis.asyncio import Redis as UpstashRedis
        Redis = UpstashRedis
        _USING_UPSTASH = True
    except ImportError:
        # Fallback to standard redis if upstash-redis is not available
        import redis.asyncio as redis
        Redis = redis.Redis
        _USING_UPSTASH = False
        import warnings
        warnings.warn("upstash-redis not found, falling back to standard redis", ImportWarning)

from ..config import settings
from ..logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    def __init__(self, client: Redis):
        self.client = client

    async def ping(self) -> bool:
        try:
            # Upstash Redis doesn't have ping, use a simple get/set test instead
            await self.client.set("ping_test", "pong", ex=10)
            result = await self.client.get("ping_test")
            await self.client.delete("ping_test")
            return result == "pong"
        except Exception:
            return False

    async def close(self) -> None:
        # Upstash Redis doesn't need explicit closing
        pass

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
            # Upstash Redis returns strings directly, no need to decode
            return data
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
        # Parse Upstash Redis URL
        redis_url = settings.redis_url
        
        # Extract token from URL if using Upstash REST API format
        if redis_url.startswith("https://"):
            # Upstash REST API format: https://token@host
            # For Upstash, we need to parse URL and token separately
            from urllib.parse import urlparse
            parsed = urlparse(redis_url)
            token = parsed.username or ""
            url = f"https://{parsed.hostname}"
            if parsed.port:
                url += f":{parsed.port}"
            
            client = Redis(url=url, token=token)
            logger.info(f"Connected to Upstash Redis REST API at {url}")
        elif "upstash.io" in redis_url:
            # Upstash Redis format: redis://token@host:port
            from urllib.parse import urlparse
            parsed = urlparse(redis_url)
            token = parsed.username or ""
            url = f"https://{parsed.hostname}"
            if parsed.port:
                url += f":{parsed.port}"
            
            client = Redis(url=url, token=token)
            logger.info(f"Connected to Upstash Redis at {url}")
        else:
            # Standard Redis format: redis://username:password@host:port
            client = Redis.from_url(redis_url)
            logger.info(f"Connected to standard Redis at {redis_url}")
            
        _redis_client = RedisClient(client)
    return _redis_client


async def close_redis() -> None:
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Upstash Redis connection closed")
