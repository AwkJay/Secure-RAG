"""Semantic query cache — Redis-backed.

Regular caches key on exact strings: "SQL injection CVEs" != "CVEs about SQL injection".
This cache keys on *embeddings* — if two queries are semantically similar enough
(cosine similarity >= threshold), the second one gets the cached result.

Storage:
  Each cache entry is a Redis hash with two fields:
    - "embedding": raw float32 bytes (384 dims * 4 bytes = 1536 bytes per entry)
    - "result": JSON-serialised answer dict

Lookup:
  1. SCAN all keys under the prefix
  2. Load their embeddings, stack into a matrix
  3. One numpy matmul → cosine similarities (L2-normalized, so dot product == cosine)
  4. Return the result if best similarity >= threshold

Graceful degradation:
  If Redis is unavailable, every operation logs a warning and returns None/noop.
  The app continues working — it just doesn't cache.
"""

import json
import logging
import uuid

import numpy as np
import redis

logger = logging.getLogger(__name__)

_PREFIX = "securerag:sem:"
_EMBEDDING_FIELD = b"embedding"
_RESULT_FIELD = b"result"


class SemanticCache:
    def __init__(self, redis_client: redis.Redis, threshold: float = 0.92, ttl: int = 3600):
        self._r = redis_client
        self.threshold = threshold
        self.ttl = ttl   # seconds each entry lives; refreshed on hit

    def _scan_keys(self) -> list[bytes]:
        keys = []
        cursor = 0
        while True:
            cursor, batch = self._r.scan(cursor, match=f"{_PREFIX}*", count=100)
            keys.extend(batch)
            if cursor == 0:
                break
        return keys

    def get(self, query_embedding: list[float]) -> dict | None:
        try:
            keys = self._scan_keys()
            if not keys:
                return None

            q = np.array(query_embedding, dtype=np.float32)

            embeddings = []
            valid_keys = []
            for key in keys:
                raw = self._r.hget(key, _EMBEDDING_FIELD)
                if raw:
                    emb = np.frombuffer(raw, dtype=np.float32).copy()
                    embeddings.append(emb)
                    valid_keys.append(key)

            if not embeddings:
                return None

            matrix = np.stack(embeddings, axis=0)
            sims = matrix @ q
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            if best_score >= self.threshold:
                raw_result = self._r.hget(valid_keys[best_idx], _RESULT_FIELD)
                if raw_result:
                    # refresh TTL so hot entries don't expire
                    self._r.expire(valid_keys[best_idx], self.ttl)
                    logger.info("Semantic cache HIT (similarity=%.4f)", best_score)
                    return json.loads(raw_result)

            logger.debug("Semantic cache miss (best=%.4f)", best_score)
            return None

        except redis.RedisError as e:
            logger.warning("Redis cache get error: %s", e)
            return None

    def set(self, query_embedding: list[float], result: dict) -> None:
        try:
            key = f"{_PREFIX}{uuid.uuid4().hex}"
            emb_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            self._r.hset(key, mapping={
                _EMBEDDING_FIELD: emb_bytes,
                _RESULT_FIELD: json.dumps(result),
            })
            self._r.expire(key, self.ttl)
        except redis.RedisError as e:
            logger.warning("Redis cache set error: %s", e)

    def invalidate(self) -> None:
        try:
            keys = self._scan_keys()
            if keys:
                self._r.delete(*keys)
            logger.info("Semantic cache cleared (%d entries)", len(keys))
        except redis.RedisError as e:
            logger.warning("Redis cache invalidate error: %s", e)

    @property
    def size(self) -> int:
        try:
            return len(self._scan_keys())
        except redis.RedisError:
            return 0


# module-level singleton — lazy init so Redis isn't required at import time
_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        from core.config import settings
        r = redis.Redis.from_url(settings.redis_url, decode_responses=False)
        _cache = SemanticCache(r, threshold=0.92, ttl=3600)
    return _cache
