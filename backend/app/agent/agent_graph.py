"""
agent_graph.py
──────────────
Assembles the LangGraph StateGraph and exposes a single compiled graph
instance (`agent_graph`) that the FastAPI endpoint invokes.

Graph topology:

    START
      │
      ▼
  [router_node]          ← classifies intent
      │
      ├─ "rag"        ──► [rag_node] ──── sufficient? ──► END
      │                       │
      │                       └─ insufficient ──► [web_search_node] ──► END
      │
      ├─ "web_search" ──► [web_search_node] ──────────────────────────► END
      │
      ├─ "math"       ──► [math_node] ─────────────────────────────────► END
      │
      └─ "unknown"    ──► [unknown_node] ──────────────────────────────► END

The ReAct-style fallback lives on the rag → web_search edge: if RAGNode
sets `rag_sufficient = False`, the graph escalates to WebSearchNode
automatically — no extra router hop needed.
"""

from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from app.agent.agent_state import AgentState
from app.agent.agent_nodes import (
    router_node,
    rag_node,
    web_search_node,
    math_node,
    unknown_node,
)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# ─────────────────────────────────────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """Reads `state["route"]` and returns the next node name."""
    return state.get("route", "unknown")


def route_after_rag(state: AgentState) -> str:
    """
    If RAGNode found a sufficient answer, go to END.
    Otherwise escalate to the web search node.
    """
    if state.get("rag_sufficient", False):
        return END
    return "web_search_node"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("router_node",     router_node)
    graph.add_node("rag_node",        rag_node)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("math_node",       math_node)
    graph.add_node("unknown_node",    unknown_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "router_node")

    # ── Router → specialised agents (conditional) ────────────────────────────
    graph.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "rag":        "rag_node",
            "web_search": "web_search_node",
            "math":       "math_node",
            "unknown":    "unknown_node",
        },
    )

    # ── RAG → END or fallback to web search (ReAct escalation) ───────────────
    graph.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {
            END:              END,
            "web_search_node": "web_search_node",
        },
    )

    # ── Terminal nodes → END ──────────────────────────────────────────────────
    graph.add_edge("web_search_node", END)
    graph.add_edge("math_node",       END)
    graph.add_edge("unknown_node",    END)

    return graph


# ── Compile once at import time ───────────────────────────────────────────────
# The compiled graph is thread-safe and can be shared across requests.

agent_graph = build_agent_graph().compile()