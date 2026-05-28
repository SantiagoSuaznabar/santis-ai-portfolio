from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message or question", min_length=1)


class ChatResponse(BaseModel):
    response: str


# ── Session schemas ───────────────────────────────────────────────────────────

class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool


# ── RAG schemas ───────────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    message: str = Field(..., description="User's message or question", min_length=1)
    session_id: str | None = Field(None, description="Session ID for memory. Omit for stateless.")


class RetrievedChunk(BaseModel):
    content: str
    relevance_rank: int = Field(..., description="1-based rank from the vector search")
    relevance_score: float = Field(..., description="Normalised similarity score [0-1]")


class CacheMissCandidate(BaseModel):
    """Best entry found in the cache that did NOT meet the threshold."""
    matched_query: str = Field(..., description="The query text of the closest cache entry found")
    cache_key: str
    similarity_score: float = Field(..., description="How close it was — below threshold so not used")
    cached_at: str | None


class CacheDebugInfo(BaseModel):
    hit: bool

    # ── Hit fields ────────────────────────────────────────────────────────────
    matched_query: str | None = Field(None, description="Original query that created this cache entry")
    similarity_score: float | None = Field(None, description="Cosine similarity between current and matched query")
    cache_key: str | None = Field(None, description="Redis key of the matched entry")
    cached_at: str | None = Field(None, description="ISO timestamp when entry was stored")
    ttl_seconds: int | None = Field(None, description="Seconds until this entry expires")
    original_retrieval_latency_ms: float | None = None
    original_llm_latency_ms: float | None = None
    original_total_latency_ms: float | None = None

    # ── Miss fields ───────────────────────────────────────────────────────────
    threshold: float | None = Field(None, description="Minimum cosine similarity required for a hit")
    entries_scanned: int | None = Field(None, description="How many cache entries were compared")
    best_candidate: CacheMissCandidate | None = Field(
        None,
        description="Closest entry found but below threshold. None if cache is empty."
    )
    skipped_reason: str | None = Field(
        None,
        description="Why cache was not checked at all (e.g. 'has_history', 'redis_unavailable', 'write_error')"
    )

    # ── Write status (on live responses only) ─────────────────────────────────
    write_status: str | None = Field(
        None,
        description="'stored' | 'skipped_has_history' | 'error:<msg>' — tells you if this response was cached"
    )


class RAGDebugInfo(BaseModel):
    query: str
    top_k: int
    retrieved_count: int
    score_threshold: float
    retrieval_mode: str = "hybrid"
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    cache: CacheDebugInfo


class RAGChatResponse(BaseModel):
    response: str
    session_id: str | None = None
    debug: RAGDebugInfo
    results: list[RetrievedChunk]


# ── History schemas ───────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    message_count: int
    messages: list[HistoryMessage]


# ── Cache inspection schemas ──────────────────────────────────────────────────

class CacheEntry(BaseModel):
    cache_key: str
    query: str
    cached_at: str | None
    ttl_seconds: int | None
    original_retrieval_latency_ms: float | None
    original_llm_latency_ms: float | None
    original_total_latency_ms: float | None
    result_count: int = Field(..., description="Number of chunks stored in this entry")
    response_preview: str = Field(..., description="First 120 chars of the cached response")


class CacheListResponse(BaseModel):
    total_entries: int
    threshold: float
    entries: list[CacheEntry]


class CacheDeleteResponse(BaseModel):
    deleted_keys: list[str]
    deleted_count: int