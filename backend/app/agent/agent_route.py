"""
agent_route.py
──────────────
Drop-in FastAPI route for the multi-agent endpoint.
Mount this in main.py with:

    from app.agent.agent_route import router as agent_router
    app.include_router(agent_router)

Endpoint:
    POST /api/agent
    — Accepts an AgentRequest (message + optional session_id)
    — Runs the LangGraph pipeline
    — Returns an AgentResponse with the answer + full debug payload
"""

from __future__ import annotations
import time
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.agent.agent_graph import agent_graph
from app.agent.agent_schemas import (
    AgentRequest,
    AgentResponse,
    AgentDebugInfo,
    AgentTraceEntry,
    RAGAgentDebug,
    WebSearchDebug,
    MathDebug,
)
from app.agent.agent_state import AgentState
from app.services.session_store import session_store
from app.services.logger_service import logger

router  = APIRouter(prefix="/api/agent", tags=["agent"])
limiter = Limiter(key_func=get_remote_address)


@router.post("", response_model=AgentResponse)
@limiter.limit("20/minute")
async def run_agent(request: Request, body: AgentRequest) -> AgentResponse:
    """
    Multi-agent endpoint.

    Routes the user's message through:
      Router → RAG | WebSearch | Math | Unknown

    With optional ReAct fallback: if RAG finds no sufficient answer,
    the graph automatically escalates to web search.

    Session memory works the same as the RAG endpoint:
      - Pass a session_id to enable conversation history
      - Omit for stateless (single-turn) queries
    """

    # ── 1. Validate session (if provided) ────────────────────────────────────
    history: list[dict] = []
    if body.session_id:
        if not session_store.session_exists(body.session_id):
            raise HTTPException(
                status_code=404,
                detail="Session not found. Create via POST /api/session",
            )
        history = session_store.get_history_for_llm(body.session_id)

    # ── 2. Build initial graph state ──────────────────────────────────────────
    initial_state: AgentState = {
        "input":       body.message,
        "session_id":  body.session_id,
        "history":     history,
        "agent_trace": [],
    }

    # ── 3. Run the graph ──────────────────────────────────────────────────────
    logger.info(f"[AgentRoute] Invoking graph for: '{body.message[:80]}'")
    t0 = time.perf_counter()

    try:
        final_state: AgentState = await agent_graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"[AgentRoute] Graph execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[AgentRoute] Graph completed in {total_ms:.0f}ms. Route: {final_state.get('route')}")

    # ── 4. Update session memory ──────────────────────────────────────────────
    final_response = final_state.get("final_response", "I was unable to produce an answer.")
    if body.session_id:
        session_store.add_message(body.session_id, "user",      body.message)
        session_store.add_message(body.session_id, "assistant", final_response)

    # ── 5. Build debug payload ────────────────────────────────────────────────
    route         = final_state.get("route", "unknown")
    rag_result    = final_state.get("rag_result")
    web_result    = final_state.get("web_result")
    math_result   = final_state.get("math_result")
    rag_sufficient = final_state.get("rag_sufficient")

    rag_debug = None
    if rag_result:
        rag_debug = RAGAgentDebug(
            from_cache=rag_result.get("from_cache", False),
            cache_key=rag_result.get("cache_key"),
            chunk_count=len(rag_result.get("retrieved_chunks", [])),
            retrieval_latency_ms=rag_result.get("retrieval_latency_ms", 0.0),
            llm_latency_ms=rag_result.get("llm_latency_ms", 0.0),
        )

    web_debug = None
    if web_result:
        web_debug = WebSearchDebug(
            source_count=len(web_result.get("sources", [])),
            search_latency_ms=web_result.get("search_latency_ms", 0.0),
            llm_latency_ms=web_result.get("llm_latency_ms", 0.0),
            sources=web_result.get("sources", []),
        )

    math_debug = None
    if math_result:
        math_debug = MathDebug(
            expression=math_result.get("expression", ""),
            raw_result=math_result.get("raw_result", ""),
        )

    trace = [
        AgentTraceEntry(node=t["node"], action=t["action"], detail=t.get("detail"))
        for t in final_state.get("agent_trace", [])
    ]

    debug = AgentDebugInfo(
        route=route,
        rag_sufficient=rag_sufficient,
        escalated_to_web=(route == "rag" and web_result is not None),
        rag=rag_debug,
        web_search=web_debug,
        math=math_debug,
        trace=trace,
    )

    return AgentResponse(
        response=final_response,
        session_id=body.session_id,
        debug=debug,
    )