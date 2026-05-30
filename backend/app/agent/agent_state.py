"""
agent_state.py
──────────────
Defines the shared state that LangGraph passes between every node.

LangGraph requires state to be a TypedDict (or dataclass). We use TypedDict
here and keep Pydantic for the API boundary (agent_schemas.py).

Field ownership per node:
  input / session_id / history   → set once by the caller before graph entry
  route                          → written by RouterNode, read by conditional edges
  rag_result                     → written by RAGNode
  rag_sufficient                 → written by RAGNode; edge uses this to decide
                                   whether to escalate to WebSearchNode
  web_result                     → written by WebSearchNode
  math_result                    → written by MathNode
  final_response                 → written by whichever node produces the answer;
                                   the API endpoint reads this field
  agent_trace                    → append-only log of what each node did;
                                   exposed in the debug payload
  error                          → set by any node on unrecoverable failure
"""

from __future__ import annotations
from typing import TypedDict, Literal


# ── Route literals ────────────────────────────────────────────────────────────

RouteTarget = Literal["rag", "web_search", "math", "unknown"]


# ── Per-node result payloads ──────────────────────────────────────────────────

class RAGResult(TypedDict, total=False):
    response: str
    retrieved_chunks: list[dict]     # same shape as qdrant_service returns
    from_cache: bool
    cache_key: str | None
    retrieval_latency_ms: float
    llm_latency_ms: float


class WebSearchResult(TypedDict, total=False):
    response: str
    sources: list[dict]              # [{title, url, snippet}]
    search_latency_ms: float
    llm_latency_ms: float


class MathResult(TypedDict, total=False):
    response: str
    expression: str                  # the expression that was evaluated
    raw_result: str                  # raw output from the evaluator tool


class TraceEntry(TypedDict):
    node: str
    action: str
    detail: str | None


# ── Main graph state ──────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # ── Inputs (set before graph entry) ──────────────────────────────────────
    input: str                       # raw user message
    session_id: str | None           # optional; same semantics as existing RAG
    history: list[dict]              # [{role, content}] from SessionStore

    # ── Routing ───────────────────────────────────────────────────────────────
    route: RouteTarget               # set by RouterNode

    # ── Node outputs ─────────────────────────────────────────────────────────
    rag_result: RAGResult
    rag_sufficient: bool             # False → graph escalates to web_search
    web_result: WebSearchResult
    math_result: MathResult

    # ── Final answer ──────────────────────────────────────────────────────────
    final_response: str

    # ── Debug / observability ─────────────────────────────────────────────────
    agent_trace: list[TraceEntry]    # append-only; reducer defined below
    error: str | None