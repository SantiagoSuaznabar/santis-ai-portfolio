"""
Semantic cache backed by Redis.

Key layout:
  semantic_cache:<sha256_of_query[:16]>  →  JSON {
      query, embedding, cached_at,
      payload: {
          response, results,
          original_retrieval_latency_ms,
          original_llm_latency_ms,
          original_total_latency_ms,
      }
  }
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from app.services.logger_service import logger
import numpy as np
from dotenv import load_dotenv
from redis import Redis
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

SEMANTIC_THRESHOLD = 0.92
CACHE_TTL_SECONDS  = 60 * 60 * 24
KEY_PREFIX         = "semantic_cache:"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


class RedisService:
    def __init__(self):
        self.KEY_PREFIX = KEY_PREFIX
        self.redis = Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    def _make_key(self, query: str) -> str:
        digest = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
        return f"{KEY_PREFIX}{digest}"

    async def _embed(self, text: str) -> list[float]:
        return await self.embeddings.aembed_query(text)

    async def get(self, query: str) -> dict:
        """
        Scan all cache entries and return a rich result dict regardless of hit/miss.

        Always returns:
          hit            : bool
          threshold      : float
          entries_scanned: int

        On hit, also returns:
          response, results, original_*_latency_ms,
          matched_query, similarity_score, cache_key, cached_at, ttl_seconds

        On miss, also returns:
          best_candidate: { matched_query, cache_key, similarity_score, cached_at } | None
        """
        query_embedding = await self._embed(query)

        cursor       = 0
        best_score   = 0.0
        best_entry   = None
        best_key     = None
        scanned      = 0

        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{KEY_PREFIX}*", count=100)
            for key in keys:
                raw = self.redis.get(key)
                if raw is None:
                    continue
                scanned += 1
                entry = json.loads(raw)
                score = _cosine_similarity(query_embedding, entry["embedding"])
                logger.debug(f"Vs [{key[-8:]}] '{entry['query']}' -> Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_entry = entry
                    best_key   = key
            if cursor == 0:
                break
        
        logger.info(f"Cache scan complete. Checked {scanned} entries. Best score: {best_score:.4f}")

        base = {
            "hit":             False,
            "threshold":       SEMANTIC_THRESHOLD,
            "entries_scanned": scanned,
        }

        if best_entry is None:
            return {**base, "best_candidate": None}

        if best_score >= SEMANTIC_THRESHOLD:
            ttl = self.redis.ttl(best_key)
            return {
                **base,
                "hit":      True,
                "response": best_entry["payload"]["response"],
                "results":  best_entry["payload"]["results"],
                "original_retrieval_latency_ms": best_entry["payload"].get("original_retrieval_latency_ms"),
                "original_llm_latency_ms":       best_entry["payload"].get("original_llm_latency_ms"),
                "original_total_latency_ms":     best_entry["payload"].get("original_total_latency_ms"),
                "matched_query":    best_entry["query"],
                "similarity_score": round(best_score, 4),
                "cache_key":        best_key,
                "cached_at":        best_entry.get("cached_at"),
                "ttl_seconds":      ttl if ttl >= 0 else None,
                "best_candidate":   None,   
            }

        return {
            **base,
            "best_candidate": {
                "matched_query":    best_entry["query"],
                "cache_key":        best_key,
                "similarity_score": round(best_score, 4),
                "cached_at":        best_entry.get("cached_at"),
            },
        }

    async def set(
        self,
        query: str,
        response: str,
        results: list[dict],
        retrieval_latency_ms: float,
        llm_latency_ms: float,
        total_latency_ms: float,
    ) -> str:
        """
        Store a cache entry. Returns the Redis key on success.
        Raises on any error (caller decides how to handle/surface it).
        """
        logger.debug(f"Attempting to cache response for: '{query}'")
        embedding = await self._embed(query)
        key = self._make_key(query)
        entry = {
            "query":     query,
            "embedding": embedding,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "response": response,
                "results":  results,
                "original_retrieval_latency_ms": retrieval_latency_ms,
                "original_llm_latency_ms":       llm_latency_ms,
                "original_total_latency_ms":     total_latency_ms,
            },
        }
        self.redis.setex(key, CACHE_TTL_SECONDS, json.dumps(entry))
        logger.info(f"Successfully cached under key: {key}")
        return key

    def list_all(self) -> list[dict]:
        """
        Return every cache entry without embeddings (too large to expose).
        Used by the inspection endpoint.
        """
        cursor  = 0
        entries = []
        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{KEY_PREFIX}*", count=100)
            for key in keys:
                raw = self.redis.get(key)
                if raw is None:
                    continue
                entry = json.loads(raw)
                ttl = self.redis.ttl(key)
                entries.append({
                    "cache_key":  key,
                    "query":      entry.get("query", ""),
                    "cached_at":  entry.get("cached_at"),
                    "ttl_seconds": ttl if ttl >= 0 else None,
                    "original_retrieval_latency_ms": entry["payload"].get("original_retrieval_latency_ms"),
                    "original_llm_latency_ms":       entry["payload"].get("original_llm_latency_ms"),
                    "original_total_latency_ms":     entry["payload"].get("original_total_latency_ms"),
                    "result_count":     len(entry["payload"].get("results", [])),
                    "response_preview": entry["payload"].get("response", "")[:120],
                })
            if cursor == 0:
                break
        return entries

    def delete(self, keys: list[str] | None = None) -> list[str]:
        """
        Delete specific keys, or ALL cache entries if keys is None.
        Returns the list of keys actually deleted.
        """
        if keys is None:
            cursor   = 0
            all_keys = []
            while True:
                cursor, found = self.redis.scan(cursor, match=f"{KEY_PREFIX}*", count=100)
                all_keys.extend(found)
                if cursor == 0:
                    break
            keys = all_keys

        deleted = []
        for key in keys:
            if self.redis.delete(key):
                deleted.append(key)
        return deleted

    def ping(self) -> bool:
        try:
            return self.redis.ping()
        except Exception:
            return False


redis_service = RedisService()