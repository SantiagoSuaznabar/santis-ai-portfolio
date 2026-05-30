"""
agent_schemas.py
────────────────
Pydantic models for the agent API surface.

Kept separate from schemas.py so the RAG and agent schemas evolve
independently without coupling.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    message: str = Field(..., description="User's message", min_length=1)
    session_id: str | None = Field(
        None,
        description="Optional session ID for conversation memory. "
                    "Create via POST /api/session — same endpoint as the RAG chatbot.",
    )


# ── Trace entry (one per node visited) ───────────────────────────────────────

class AgentTraceEntry(BaseModel):
    node: str   = Field(..., description="Node that produced this entry")
    action: str = Field(..., description="What the node did, e.g. 'cache_hit', 'evaluated'")
    detail: str | None = Field(None, description="Extra context, e.g. chunk count or expression")


# ── Per-agent debug payloads ──────────────────────────────────────────────────

class RAGAgentDebug(BaseModel):
    from_cache: bool
    cache_key: str | None
    chunk_count: int
    retrieval_latency_ms: float
    llm_latency_ms: float


class WebSearchDebug(BaseModel):
    source_count: int
    search_latency_ms: float
    llm_latency_ms: float
    sources: list[dict] = Field(default_factory=list)


class MathDebug(BaseModel):
    expression: str
    raw_result: str


# ── Top-level debug ───────────────────────────────────────────────────────────

class AgentDebugInfo(BaseModel):
    route: str = Field(..., description="Which agent handled the query")
    rag_sufficient: bool | None = Field(
        None,
        description="RAG sufficiency flag — None if RAG was not involved",
    )
    escalated_to_web: bool = Field(
        False,
        description="True if RAG was insufficient and the graph fell back to web search",
    )
    rag: RAGAgentDebug | None = None
    web_search: WebSearchDebug | None = None
    math: MathDebug | None = None
    trace: list[AgentTraceEntry] = Field(default_factory=list)


# ── Response ──────────────────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    response: str = Field(..., description="Final answer from the agent")
    session_id: str | None = None
    debug: AgentDebugInfo