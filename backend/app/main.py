import time
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from app.models.schemas import (
    ChatRequest, ChatResponse,
    RAGRequest, RAGChatResponse, RAGDebugInfo, RetrievedMovie,
    CacheDebugInfo, CacheMissCandidate,
    SessionCreateResponse, SessionDeleteResponse,
    SessionHistoryResponse, HistoryMessage,
    CacheEntry, CacheListResponse, CacheDeleteResponse,
)
from app.services.llm_service import llm_service
from app.services.qdrant_service import qdrant_service, SCORE_THRESHOLD
from app.services.redis_service import redis_service, _cosine_similarity
from app.services.session_store import session_store

load_dotenv()

app = FastAPI(
    title="AI Portfolio Backend",
    description="RAG chatbot — hybrid search + semantic cache",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TOP_K = 5


@app.get("/")
async def root():
    return {
        "status": "online",
        "environment": os.getenv("ENV_MODE", "development"),
        "redis": "connected" if redis_service.ping() else "unavailable",
    }


@app.post("/api/chat/test", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    return ChatResponse(response=await llm_service.get_simple_response(request.message))


@app.post("/api/session", response_model=SessionCreateResponse, status_code=201)
async def create_session():
    return SessionCreateResponse(**session_store.create_session())


@app.delete("/api/session/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str):
    if not session_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDeleteResponse(session_id=session_id, deleted=True)


@app.get("/api/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_history(session_id: str):
    if not session_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    history = session_store.get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        message_count=len(history),
        messages=[HistoryMessage(**m) for m in history],
    )


@app.get("/api/cache", response_model=CacheListResponse)
async def list_cache():
    """
    List every entry currently in the semantic cache.
    Useful for debugging hits/misses — shows the exact query text stored,
    when it was cached, TTL remaining, and a preview of the response.
    Embeddings are omitted (too large).
    """
    entries = redis_service.list_all()
    return CacheListResponse(
        total_entries=len(entries),
        threshold=0.92,
        entries=[CacheEntry(**e) for e in entries],
    )


@app.delete("/api/cache", response_model=CacheDeleteResponse)
async def clear_cache():
    """Delete ALL cache entries."""
    deleted = redis_service.delete()
    return CacheDeleteResponse(deleted_keys=deleted, deleted_count=len(deleted))


@app.delete("/api/cache/{cache_key:path}", response_model=CacheDeleteResponse)
async def delete_cache_entry(cache_key: str):
    """Delete a single cache entry by its key (from GET /api/cache)."""
    deleted = redis_service.delete([cache_key])
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Cache key not found: {cache_key}")
    return CacheDeleteResponse(deleted_keys=deleted, deleted_count=len(deleted))


@app.post("/api/chat/rag", response_model=RAGChatResponse)
async def rag_chat(request: RAGRequest):
    total_start = time.perf_counter()

    history: list[dict] = []
    if request.session_id:
        if not session_store.session_exists(request.session_id):
            raise HTTPException(status_code=404, detail="Session not found. Create via POST /api/session")
        history = session_store.get_history_for_llm(request.session_id)

    cache_lookup: dict | None = None
    cache_skip_reason: str | None = None

    if history:
        cache_skip_reason = "has_history"
    else:
        try:
            cache_lookup = await redis_service.get(request.message)
        except Exception as e:
            cache_skip_reason = f"redis_unavailable: {e}"

    if cache_lookup and cache_lookup["hit"]:
        total_ms      = (time.perf_counter() - total_start) * 1000
        cached_movies = [RetrievedMovie(**m) for m in cache_lookup["results"]]

        if request.session_id:
            session_store.add_message(request.session_id, "user", request.message)
            session_store.add_message(request.session_id, "assistant", cache_lookup["response"])

        return RAGChatResponse(
            response=cache_lookup["response"],
            session_id=request.session_id,
            debug=RAGDebugInfo(
                query=request.message,
                top_k=TOP_K,
                retrieved_count=len(cached_movies),
                score_threshold=SCORE_THRESHOLD,
                retrieval_mode="hybrid",
                retrieval_latency_ms=0.0,
                llm_latency_ms=0.0,
                total_latency_ms=round(total_ms, 2),
                cache=CacheDebugInfo(
                    hit=True,
                    matched_query=cache_lookup["matched_query"],
                    similarity_score=cache_lookup["similarity_score"],
                    cache_key=cache_lookup["cache_key"],
                    cached_at=cache_lookup["cached_at"],
                    ttl_seconds=cache_lookup["ttl_seconds"],
                    threshold=cache_lookup["threshold"],
                    entries_scanned=cache_lookup["entries_scanned"],
                    original_retrieval_latency_ms=cache_lookup["original_retrieval_latency_ms"],
                    original_llm_latency_ms=cache_lookup["original_llm_latency_ms"],
                    original_total_latency_ms=cache_lookup["original_total_latency_ms"],
                ),
            ),
            results=cached_movies,
        )

    t0 = time.perf_counter()
    try:
        docs = await qdrant_service.search(request.message, k=TOP_K)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector store error: {e}")
    retrieval_ms = (time.perf_counter() - t0) * 1000

    def _miss_cache_info(write_status: str) -> CacheDebugInfo:
        best = None
        if cache_lookup and cache_lookup.get("best_candidate"):
            best = CacheMissCandidate(**cache_lookup["best_candidate"])
        return CacheDebugInfo(
            hit=False,
            threshold=cache_lookup["threshold"] if cache_lookup else None,
            entries_scanned=cache_lookup["entries_scanned"] if cache_lookup else None,
            best_candidate=best,
            skipped_reason=cache_skip_reason,
            write_status=write_status,
        )

    if not docs:
        total_ms = (time.perf_counter() - total_start) * 1000
        return RAGChatResponse(
            response="I couldn't find any relevant movies for your query.",
            session_id=request.session_id,
            debug=RAGDebugInfo(
                query=request.message,
                top_k=TOP_K,
                retrieved_count=0,
                score_threshold=SCORE_THRESHOLD,
                retrieval_mode="hybrid",
                retrieval_latency_ms=round(retrieval_ms, 2),
                llm_latency_ms=0.0,
                total_latency_ms=round(total_ms, 2),
                cache=_miss_cache_info("skipped_no_results"),
            ),
            results=[],
        )

    t0 = time.perf_counter()
    try:
        ai_response = await llm_service.get_rag_response(
            question=request.message,
            context_docs=docs,
            history=history or None,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")
    llm_ms    = (time.perf_counter() - t0) * 1000
    total_ms  = (time.perf_counter() - total_start) * 1000

    retrieved_movies = [
        RetrievedMovie(
            title=doc["metadata"].get("title", "Unknown"),
            overview=doc["overview"],
            genres=doc["metadata"].get("genres", []),
            vote_average=doc["metadata"].get("vote_average"),
            release_date=str(doc["metadata"].get("release_date", "")) or None,
            poster_url=doc["metadata"].get("poster_url"),
            relevance_rank=doc["relevance_rank"],
            relevance_score=doc["relevance_score"],
        )
        for doc in docs
    ]

    write_status = "skipped_has_history" if history else None
    if not history:
        try:
            stored_key = await redis_service.set(
                query=request.message,
                response=ai_response,
                results=[m.model_dump() for m in retrieved_movies],
                retrieval_latency_ms=round(retrieval_ms, 2),
                llm_latency_ms=round(llm_ms, 2),
                total_latency_ms=round(total_ms, 2),
            )
            write_status = f"stored:{stored_key}"
        except Exception as e:
            write_status = f"error:{e}"

    if request.session_id:
        session_store.add_message(request.session_id, "user", request.message)
        session_store.add_message(request.session_id, "assistant", ai_response)

    return RAGChatResponse(
        response=ai_response,
        session_id=request.session_id,
        debug=RAGDebugInfo(
            query=request.message,
            top_k=TOP_K,
            retrieved_count=len(retrieved_movies),
            score_threshold=SCORE_THRESHOLD,
            retrieval_mode="hybrid",
            retrieval_latency_ms=round(retrieval_ms, 2),
            llm_latency_ms=round(llm_ms, 2),
            total_latency_ms=round(total_ms, 2),
            cache=_miss_cache_info(write_status),
        ),
        results=retrieved_movies,
    )


@app.get("/api/cache/debug")
async def debug_cache(q: str):
    """
    Low-level cache probe. Pass the exact query string as ?q=...
    Returns for every stored entry:
      - The stored query text
      - The raw cosine similarity vs your query
      - Whether the embedding field exists and its length
      - The cache key
    No threshold applied — you see all scores as-is.
    Lets you verify the similarity engine is working and what score
    your query actually gets against each stored entry.
    """
    try:
        query_embedding = await redis_service._embed(q)
    except Exception as e:
        return {"error": f"Failed to embed query: {e}"}

    cursor  = 0
    results = []
    while True:
        cursor, keys = redis_service.redis.scan(cursor, match=f"{redis_service.KEY_PREFIX}*", count=100)
        for key in keys:
            raw = redis_service.redis.get(key)
            if raw is None:
                continue
            try:
                entry   = json.loads(raw)
                emb     = entry.get("embedding")
                has_emb = isinstance(emb, list) and len(emb) > 0
                score   = round(_cosine_similarity(query_embedding, emb), 6) if has_emb else None
                results.append({
                    "cache_key":        key,
                    "stored_query":     entry.get("query", "<missing>"),
                    "cached_at":        entry.get("cached_at", "<missing>"),
                    "embedding_present": has_emb,
                    "embedding_length": len(emb) if has_emb else 0,
                    "similarity_score": score,
                    "would_hit":        score is not None and score >= 0.92,
                })
            except Exception as e:
                results.append({"cache_key": key, "error": str(e)})
        if cursor == 0:
            break

    return {
        "query": q,
        "query_embedding_length": len(query_embedding),
        "threshold": 0.92,
        "entries_checked": len(results),
        "results": sorted(results, key=lambda x: x.get("similarity_score") or 0, reverse=True),
    }