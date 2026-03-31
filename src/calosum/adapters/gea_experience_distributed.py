from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RedisExperienceConfig:
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "calosum:gea:experience"
    ttl_seconds: int = 3600
    max_stored: int = 500


class RedisDistributedExperienceStore:
    """Inter-agent experience sharing via Redis pub/sub + sorted sets.

    When Redis is unavailable, degrades to in-memory local storage
    so the GEA reflection loop is never blocked by infrastructure.
    """

    def __init__(self, config: RedisExperienceConfig | None = None) -> None:
        self.config = config or RedisExperienceConfig()
        self._redis: Any | None = None
        self._local_cache: list[dict[str, Any]] = []
        self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis
            self._redis = redis.from_url(self.config.redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("Redis distributed experience store connected: %s", self.config.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable, using local fallback: %s", exc)
            self._redis = None

    def broadcast_experience(
        self,
        *,
        agent_id: str,
        context_type: str,
        variant_id: str,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "agent_id": agent_id,
            "context_type": context_type,
            "variant_id": variant_id,
            "score": score,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        if self._redis is not None:
            try:
                key = f"{self.config.key_prefix}:{context_type}"
                self._redis.zadd(key, {json.dumps(entry): score})
                self._redis.expire(key, self.config.ttl_seconds)
                self._redis.zremrangebyrank(key, 0, -self.config.max_stored - 1)
                return
            except Exception as exc:
                logger.warning("Redis broadcast failed, caching locally: %s", exc)

        self._local_cache.append(entry)
        if len(self._local_cache) > self.config.max_stored:
            self._local_cache = self._local_cache[-self.config.max_stored:]

    def collect_peer_experiences(
        self,
        *,
        context_type: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if self._redis is not None:
            try:
                key = f"{self.config.key_prefix}:{context_type}"
                raw = self._redis.zrevrange(key, 0, limit - 1)
                return [json.loads(entry) for entry in raw if entry]
            except Exception as exc:
                logger.warning("Redis collect failed: %s", exc)

        return sorted(
            [e for e in self._local_cache if e.get("context_type") == context_type],
            key=lambda e: e.get("score", 0),
            reverse=True,
        )[:limit]

    # ExperienceStorePort compatibility
    def record_experience(
        self,
        *,
        context_type: str,
        variant_id: str,
        score: float,
        reward: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.broadcast_experience(
            agent_id="local",
            context_type=context_type,
            variant_id=variant_id,
            score=score,
            metadata={**(metadata or {}), "reward": reward},
        )

    def variant_prior(
        self,
        *,
        context_type: str,
        variant_id: str,
        limit: int = 100,
    ) -> float:
        experiences = self.collect_peer_experiences(context_type=context_type, limit=limit)
        matching = [e for e in experiences if e.get("variant_id") == variant_id]
        if not matching:
            return 0.5
        return sum(e.get("score", 0.5) for e in matching) / len(matching)
