"""
services/semantic_cache.py

A Redis-backed semantic cache for the Financial RAG pipeline.

How it works:
  - Each cached entry stores the vector embedding of the resolved query + the answer.
  - On lookup, we compute cosine similarity between the incoming query embedding
    and all stored embeddings. If similarity >= THRESHOLD, we return the cached answer.
  - Short/ambiguous queries (< MIN_QUERY_WORDS words) are never cached.
  - Entries expire after CACHE_TTL_DAYS days automatically (Redis TTL).

Fail-safe: All Redis operations are wrapped in try/except. If Redis is down
or not configured, the cache silently returns None and the pipeline runs normally.
"""

import json
import logging
import uuid
from typing import Optional

import numpy as np
import redis

logger = logging.getLogger(__name__)

CACHE_PREFIX     = "fin_rag:"       # Namespace prefix for all cache keys
SIMILARITY_THRESHOLD = 0.92         # Cosine similarity threshold for a cache hit
MIN_QUERY_WORDS  = 6                # Shorter queries are too ambiguous to cache reliably
CACHE_TTL_DAYS   = 7                # Cache entries expire after 7 days


class SemanticCache:
    """
    Redis-backed semantic cache.
    Pass embedding (list[float]) + resolved_query (str) to get/set.
    """

    def __init__(self, host: str, port: int, username: str, password: str):
        self._enabled = bool(host and password)
        if not self._enabled:
            logger.info("SemanticCache: Redis not configured — cache disabled.")
            self._r = None
            return

        try:
            self._r = redis.Redis(
                host=host,
                port=port,
                username=username,
                password=password,
                decode_responses=True,
                socket_timeout=3,           # Never block the API for more than 3s
                socket_connect_timeout=3,
            )
            # Ping to verify connection at startup
            self._r.ping()
            logger.info(f"SemanticCache: Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.warning(f"SemanticCache: Failed to connect to Redis — cache disabled. Error: {e}")
            self._enabled = False
            self._r = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, embedding: list[float], resolved_query: str) -> Optional[str]:
        """
        Look up a cached answer by vector similarity.
        Returns the cached answer string, or None on a cache miss.
        """
        if not self._enabled:
            return None
        if len(resolved_query.split()) < MIN_QUERY_WORDS:
            logger.debug(f"Cache SKIP (too short): '{resolved_query}'")
            return None

        try:
            best_sim   = 0.0
            best_answer = None

            for key in self._r.scan_iter(f"{CACHE_PREFIX}*"):
                raw = self._r.get(key)
                if not raw:
                    continue
                entry = json.loads(raw)
                sim = self._cosine_similarity(embedding, entry["embedding"])
                if sim > best_sim:
                    best_sim    = sim
                    best_answer = entry["answer"]

            if best_sim >= SIMILARITY_THRESHOLD:
                logger.info(f"Cache HIT  | sim={best_sim:.4f} | query='{resolved_query[:60]}'")
                return best_answer

            logger.info(f"Cache MISS | best_sim={best_sim:.4f} | query='{resolved_query[:60]}'")
            return None

        except Exception as e:
            logger.warning(f"Cache lookup error (returning None): {e}")
            return None

    def set(self, embedding: list[float], resolved_query: str, answer: str) -> None:
        """
        Store a new cache entry with a 7-day TTL.
        Short queries are silently skipped.
        """
        if not self._enabled:
            return
        if len(resolved_query.split()) < MIN_QUERY_WORDS:
            return

        key   = f"{CACHE_PREFIX}{uuid.uuid4()}"
        entry = json.dumps({
            "embedding": embedding,
            "question":  resolved_query,
            "answer":    answer,
        })

        try:
            self._r.set(key, entry, ex=CACHE_TTL_DAYS * 86_400)
            logger.info(f"Cache SET  | key={key} | query='{resolved_query[:60]}'")
        except Exception as e:
            logger.warning(f"Cache write error (skipping): {e}")

    def clear(self) -> int:
        """Delete all cache entries. Returns number of keys deleted."""
        if not self._enabled:
            return 0
        try:
            keys = list(self._r.scan_iter(f"{CACHE_PREFIX}*"))
            if keys:
                self._r.delete(*keys)
            logger.info(f"Cache CLEAR | deleted {len(keys)} entries")
            return len(keys)
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))
