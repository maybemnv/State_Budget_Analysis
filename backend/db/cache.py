from functools import lru_cache
from typing import Any, Optional

from cachetools import LRUCache

from ..logger import get_logger

logger = get_logger(__name__)


class CacheClient:
    def __init__(self, maxsize: int = 100):
        self._cache: LRUCache[str, Any] = LRUCache(maxsize=maxsize)

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._cache[key] = value

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()

    def has(self, key: str) -> bool:
        return key in self._cache


_cache_client: Optional[CacheClient] = None


def get_cache() -> CacheClient:
    global _cache_client
    if _cache_client is None:
        _cache_client = CacheClient(maxsize=100)
    return _cache_client
