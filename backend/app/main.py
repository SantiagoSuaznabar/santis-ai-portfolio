import asyncio
import time
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import os
from app.services.logger_service import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


from app.models.schemas import (
    ChatRequest, ChatResponse,
    RAGRequest, RAGChatResponse, RAGDebugInfo, RetrievedChunk,
    CacheDebugInfo, CacheMissCandidate,
    SessionCreateResponse, SessionDeleteResponse,
    SessionHistoryResponse, HistoryMessage,
    CacheEntry, CacheListResponse, CacheDeleteResponse,
)
from app.services.llm_service import llm_service
from app.services.qdrant_service import qdrant_service, SCORE_THRESHOLD
from app.services.redis_service import redis_service, _cosine_similarity
from app.services.session_store import session_store
from app.services.log_capture import capture_logs

load_dotenv()
redis_url = os.getenv("REDIS_URL", "memory://")
limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url)
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS")
origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]

app = FastAPI(
    title="AI Portfolio Backend",
    description="RAG chatbot — hybrid search + semantic cache",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TOP_K = 5

# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(event: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(event)}\n\n"


# ── Shared pipeline helpers ───────────────────────────────────────────────────

async def _resolve_query(message: str, session_id: str | None) -> tuple[list[dict], str]:
    """
    Returns (history, search_query).
    Raises HTTPException 404 if session_id is provided but not found.
    """
    history: list[dict] = []
    if session_id:
        if not session_store.session_exists(session_id):
            raise HTTPException(
                status_code=404,
                detail="Session not found. Create via POST /api/session",
            )
        history = session_store.get_history_for_llm(session_id)

    search_query = message
    if history:
        search_query = await llm_service.reformulate_query(message, history)
        logger.info(f"Reformulated Query: '{message}' -> '{search_query}'")

    return history, search_query


def _build_miss_cache_info(
    cache_lookup: dict | None,
    cache_skip_reason: str | None,
    write_status: str,
) -> CacheDebugInfo:
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


def _build_hit_cache_info(cache_lookup: dict) -> CacheDebugInfo:
    return CacheDebugInfo(
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
    )


def _map_chunks(docs: list[dict]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            content=doc["page_content"],
            relevance_rank=doc["relevance_rank"],
            relevance_score=doc["relevance_score"],
        )
        for doc in docs
    ]


# ── App startup ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up AI Portfolio Backend...")
    qdrant_service.verify_hybrid_setup()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "online",
        "environment": os.getenv("ENV_MODE", "development"),
        "redis": "connected" if redis_service.ping() else "unavailable",
    }


# ── Test chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat/test", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    return ChatResponse(response=await llm_service.get_simple_response(request.message))


# ── Session management ────────────────────────────────────────────────────────

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


# ── Cache management ──────────────────────────────────────────────────────────

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


# ── RAG chat (non-streaming) ──────────────────────────────────────────────────

@app.post("/api/chat/rag", response_model=RAGChatResponse)
@limiter.limit("5/minute")
async def rag_chat(request: Request, body: RAGRequest):
    logger.info(f"--- New Request | Session: {body.session_id} ---")
    logger.info(f"User Message: '{body.message}'")

    total_start = time.perf_counter()

    # 1 & 2 — History + query reformulation
    history, search_query = await _resolve_query(body.message, body.session_id)

    # 3 — Check cache
    cache_lookup: dict | None = None
    cache_skip_reason: str | None = None
    try:
        cache_lookup = await redis_service.get(search_query)
    except Exception as e:
        cache_skip_reason = f"redis_unavailable: {e}"
        logger.error(f"Redis cache error: {e}")

    # 4 — Cache hit
    if cache_lookup and cache_lookup.get("hit"):
        logger.info(f"Cache HIT! Score: {cache_lookup['similarity_score']}. Returning cached response.")
        total_ms = (time.perf_counter() - total_start) * 1000
        cached_chunks = [RetrievedChunk(**m) for m in cache_lookup["results"]]

        if body.session_id:
            session_store.add_message(body.session_id, "user", body.message)
            session_store.add_message(body.session_id, "assistant", cache_lookup["response"])

        return RAGChatResponse(
            response=cache_lookup["response"],
            session_id=body.session_id,
            debug=RAGDebugInfo(
                query=search_query,
                top_k=TOP_K,
                retrieved_count=len(cached_chunks),
                score_threshold=SCORE_THRESHOLD,
                retrieval_mode="hybrid",
                retrieval_latency_ms=0.0,
                llm_latency_ms=0.0,
                total_latency_ms=round(total_ms, 2),
                cache=_build_hit_cache_info(cache_lookup),
            ),
            results=cached_chunks,
        )

    logger.info("Cache MISS. Proceeding to Vector Store.")

    # 5 — Qdrant search
    t0 = time.perf_counter()
    try:
        docs = await qdrant_service.search(search_query, k=TOP_K)
    except Exception as e:
        logger.error(f"Vector store error: {e}")
        raise HTTPException(status_code=503, detail=f"Vector store error: {e}")
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not docs:
        logger.info("No relevant documents found in vector store.")
        total_ms = (time.perf_counter() - total_start) * 1000
        return RAGChatResponse(
            response="I couldn't find any relevant passages from the book for your query.",
            session_id=body.session_id,
            debug=RAGDebugInfo(
                query=search_query,
                top_k=TOP_K,
                retrieved_count=0,
                score_threshold=SCORE_THRESHOLD,
                retrieval_mode="hybrid",
                retrieval_latency_ms=round(retrieval_ms, 2),
                llm_latency_ms=0.0,
                total_latency_ms=round(total_ms, 2),
                cache=_build_miss_cache_info(cache_lookup, cache_skip_reason, "skipped_no_results"),
            ),
            results=[],
        )

    # 6 — LLM generation
    t0 = time.perf_counter()
    try:
        logger.info("Generating response with LLM...")
        ai_response = await llm_service.get_rag_response(
            question=body.message,
            context_docs=docs,
            history=history or None,
        )
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")
    llm_ms   = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    retrieved_chunks = _map_chunks(docs)

    # 7 — Write to cache
    write_status = None
    try:
        stored_key = await redis_service.set(
            query=search_query,
            response=ai_response,
            results=[m.model_dump() for m in retrieved_chunks],
            retrieval_latency_ms=round(retrieval_ms, 2),
            llm_latency_ms=round(llm_ms, 2),
            total_latency_ms=round(total_ms, 2),
        )
        write_status = f"stored:{stored_key}"
    except Exception as e:
        logger.error(f"Failed to write to cache: {e}")
        write_status = f"error:{e}"

    # 8 — Update session
    if body.session_id:
        session_store.add_message(body.session_id, "user", body.message)
        session_store.add_message(body.session_id, "assistant", ai_response)

    return RAGChatResponse(
        response=ai_response,
        session_id=body.session_id,
        debug=RAGDebugInfo(
            query=search_query,
            top_k=TOP_K,
            retrieved_count=len(retrieved_chunks),
            score_threshold=SCORE_THRESHOLD,
            retrieval_mode="hybrid",
            retrieval_latency_ms=round(retrieval_ms, 2),
            llm_latency_ms=round(llm_ms, 2),
            total_latency_ms=round(total_ms, 2),
            cache=_build_miss_cache_info(cache_lookup, cache_skip_reason, write_status),
        ),
        results=retrieved_chunks,
    )


# ── RAG chat (streaming) ──────────────────────────────────────────────────────

@app.post("/api/chat/rag/stream")
@limiter.limit("5/minute")
async def rag_chat_stream(request: Request, body: RAGRequest):
    """
    Streaming version of POST /api/chat/rag.

    Returns Server-Sent Events (SSE) with Content-Type: text/event-stream.
    Each event is a JSON object on a `data:` line.

    Event sequence
    ──────────────
    1. {"type": "meta", "query": str, "session_id": str|null,
        "results": [RetrievedChunk, ...], "from_cache": bool}
       Sent as soon as chunks are retrieved (or immediately on cache hit).
       The frontend can render source cards before the first token arrives.

    2. {"type": "token", "content": str}
       One event per LLM token (or word, on a cache hit replay).

    3. {"type": "done", "debug": RAGDebugInfo}
       Sent once after the last token. Contains the full debug payload.

    Error path
    ──────────
    {"type": "error", "message": str}  — stream ends after this event.

    Notes
    ─────
    - On a cache hit the cached response is replayed word-by-word so the
      frontend behaviour is identical regardless of cache state.
    - The full response is buffered server-side before caching so the cache
      write happens after streaming completes, not before.
    """
    logger.info(f"--- New Stream Request | Session: {body.session_id} ---")
    logger.info(f"User Message: '{body.message}'")

    async def generate():
        total_start = time.perf_counter()

        with capture_logs(logger) as log_handler:

            # 1 & 2 — History + query reformulation
            try:
                history, search_query = await _resolve_query(body.message, body.session_id)
            except HTTPException as exc:
                yield _sse({"type": "error", "message": exc.detail})
                return
            for event in log_handler.drain():
                yield _sse(event)

            # 3 — Check cache
            cache_lookup: dict | None = None
            cache_skip_reason: str | None = None
            try:
                cache_lookup = await redis_service.get(search_query)
            except Exception as e:
                cache_skip_reason = f"redis_unavailable: {e}"
                logger.error(f"Redis cache error: {e}")
            for event in log_handler.drain():
                yield _sse(event)

            # 4 — Cache hit: replay word-by-word
            if cache_lookup and cache_lookup.get("hit"):
                logger.info(f"Cache HIT. Score: {cache_lookup['similarity_score']}")
                for event in log_handler.drain():
                    yield _sse(event)

                cached_chunks = [RetrievedChunk(**m) for m in cache_lookup["results"]]
                yield _sse({
                    "type":       "meta",
                    "query":      search_query,
                    "session_id": body.session_id,
                    "results":    [c.model_dump() for c in cached_chunks],
                    "from_cache": True,
                })

                # Replay cached response word-by-word
                words = cache_lookup["response"].split(" ")
                for i, word in enumerate(words):
                    token = word if i == 0 else f" {word}"
                    yield _sse({"type": "token", "content": token})
                    await asyncio.sleep(0)  # yield control to the event loop

                if body.session_id:
                    session_store.add_message(body.session_id, "user", body.message)
                    session_store.add_message(body.session_id, "assistant", cache_lookup["response"])

                total_ms = (time.perf_counter() - total_start) * 1000
                debug = RAGDebugInfo(
                    query=search_query,
                    top_k=TOP_K,
                    retrieved_count=len(cached_chunks),
                    score_threshold=SCORE_THRESHOLD,
                    retrieval_mode="hybrid",
                    retrieval_latency_ms=0.0,
                    llm_latency_ms=0.0,
                    total_latency_ms=round(total_ms, 2),
                    cache=_build_hit_cache_info(cache_lookup),
                )
                yield _sse({"type": "done", "debug": debug.model_dump()})
                return

            logger.info("Cache MISS. Proceeding to Vector Store.")
            for event in log_handler.drain():
                yield _sse(event)

            # 5 — Qdrant search
            t0 = time.perf_counter()
            try:
                docs = await qdrant_service.search(search_query, k=TOP_K)
            except Exception as e:
                logger.error(f"Vector store error: {e}")
                for event in log_handler.drain():
                    yield _sse(event)
                yield _sse({"type": "error", "message": f"Vector store error: {e}"})
                return
            retrieval_ms = (time.perf_counter() - t0) * 1000
            for event in log_handler.drain():
                yield _sse(event)

            retrieved_chunks = _map_chunks(docs)

            # Send metadata — frontend can render source cards immediately
            yield _sse({
                "type":       "meta",
                "query":      search_query,
                "session_id": body.session_id,
                "results":    [c.model_dump() for c in retrieved_chunks],
                "from_cache": False,
            })

            if not docs:
                logger.info("No relevant documents found.")
                for event in log_handler.drain():
                    yield _sse(event)
                no_result_msg = "I couldn't find any relevant passages from the book for your query."
                yield _sse({"type": "token", "content": no_result_msg})
                total_ms = (time.perf_counter() - total_start) * 1000
                debug = RAGDebugInfo(
                    query=search_query,
                    top_k=TOP_K,
                    retrieved_count=0,
                    score_threshold=SCORE_THRESHOLD,
                    retrieval_mode="hybrid",
                    retrieval_latency_ms=round(retrieval_ms, 2),
                    llm_latency_ms=0.0,
                    total_latency_ms=round(total_ms, 2),
                    cache=_build_miss_cache_info(cache_lookup, cache_skip_reason, "skipped_no_results"),
                )
                yield _sse({"type": "done", "debug": debug.model_dump()})
                return

            # 6 — Stream LLM tokens
            #     Drain logs before the first token, then again between tokens
            #     so service-level logs (e.g. "Assembling prompt...") appear
            #     before the response starts building.
            t0 = time.perf_counter()
            response_parts: list[str] = []
            try:
                async for token in llm_service.stream_rag_response(
                    question=body.message,
                    context_docs=docs,
                    history=history or None,
                ):
                    # Flush any logs that fired before this token
                    for event in log_handler.drain():
                        yield _sse(event)
                    response_parts.append(token)
                    yield _sse({"type": "token", "content": token})
            except Exception as e:
                logger.error(f"LLM streaming error: {e}")
                for event in log_handler.drain():
                    yield _sse(event)
                yield _sse({"type": "error", "message": f"LLM error: {e}"})
                return

            llm_ms      = (time.perf_counter() - t0) * 1000
            total_ms    = (time.perf_counter() - total_start) * 1000
            ai_response = "".join(response_parts)

            # 7 — Write to cache
            write_status = None
            try:
                stored_key = await redis_service.set(
                    query=search_query,
                    response=ai_response,
                    results=[m.model_dump() for m in retrieved_chunks],
                    retrieval_latency_ms=round(retrieval_ms, 2),
                    llm_latency_ms=round(llm_ms, 2),
                    total_latency_ms=round(total_ms, 2),
                )
                write_status = f"stored:{stored_key}"
            except Exception as e:
                logger.error(f"Failed to write to cache: {e}")
                write_status = f"error:{e}"
            for event in log_handler.drain():
                yield _sse(event)

            # 8 — Update session
            if body.session_id:
                session_store.add_message(body.session_id, "user", body.message)
                session_store.add_message(body.session_id, "assistant", ai_response)

            # 9 — Done event with full debug payload
            debug = RAGDebugInfo(
                query=search_query,
                top_k=TOP_K,
                retrieved_count=len(retrieved_chunks),
                score_threshold=SCORE_THRESHOLD,
                retrieval_mode="hybrid",
                retrieval_latency_ms=round(retrieval_ms, 2),
                llm_latency_ms=round(llm_ms, 2),
                total_latency_ms=round(total_ms, 2),
                cache=_build_miss_cache_info(cache_lookup, cache_skip_reason, write_status),
            )
            yield _sse({"type": "done", "debug": debug.model_dump()})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx response buffering
        },
    )


# ── Cache debug ───────────────────────────────────────────────────────────────

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
                    "cache_key":         key,
                    "stored_query":      entry.get("query", "<missing>"),
                    "cached_at":         entry.get("cached_at", "<missing>"),
                    "embedding_present": has_emb,
                    "embedding_length":  len(emb) if has_emb else 0,
                    "similarity_score":  score,
                    "would_hit":         score is not None and score >= 0.92,
                })
            except Exception as e:
                results.append({"cache_key": key, "error": str(e)})
        if cursor == 0:
            break

    return {
        "query":                  q,
        "query_embedding_length": len(query_embedding),
        "threshold":              0.92,
        "entries_checked":        len(results),
        "results":                sorted(results, key=lambda x: x.get("similarity_score") or 0, reverse=True),
    }