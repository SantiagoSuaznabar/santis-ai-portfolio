"""
agent_tools.py
──────────────
LangChain tool definitions consumed by the agent nodes.

Tools defined here:
  rag_search_tool   — cache-aware hybrid RAG (reuses existing services)
  evaluate_math_tool — safe Python expression evaluator

Note on Brave Search:
  Brave is consumed via MCP in web_search_node (agent_nodes.py), not as a
  @tool here. The MCP client returns LangChain-compatible tools at runtime,
  so there is nothing to define statically in this file for web search.
"""

from __future__ import annotations
import os
import time
import ast
import operator
from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RAG search tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
async def rag_search_tool(query: str) -> dict:
    """
    Search the 'Three Days of Happiness' book knowledge base.

    Checks the semantic cache first (Redis). On a miss, runs a hybrid
    dense+sparse search against Qdrant and returns the top chunks.

    Args:
        query: A standalone, self-contained question or search phrase.

    Returns:
        {
            "from_cache": bool,
            "cache_key": str | None,
            "chunks": [{"page_content", "metadata", "relevance_rank", "relevance_score"}],
            "retrieval_latency_ms": float,
        }
    """
    from app.services.redis_service import redis_service
    from app.services.qdrant_service import qdrant_service

    # ── 1. Cache probe ────────────────────────────────────────────────────────
    try:
        cache_result = await redis_service.get(query)
        if cache_result.get("hit"):
            return {
                "from_cache": True,
                "cache_key": cache_result["cache_key"],
                "chunks": cache_result["results"],
                "cached_response": cache_result["response"],
                "retrieval_latency_ms": 0.0,
            }
    except Exception:
        pass  # Redis unavailable — fall through to vector store

    # ── 2. Vector store search ────────────────────────────────────────────────
    t0 = time.perf_counter()
    chunks = await qdrant_service.search(query)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    return {
        "from_cache": False,
        "cache_key": None,
        "chunks": chunks,
        "retrieval_latency_ms": round(retrieval_ms, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Math evaluator tool
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_OPERATORS: dict = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS: dict[str, object] = {
    "abs":   abs,
    "round": round,
    "sqrt":  __import__("math").sqrt,
    "log":   __import__("math").log,
    "log10": __import__("math").log10,
    "sin":   __import__("math").sin,
    "cos":   __import__("math").cos,
    "tan":   __import__("math").tan,
    "pi":    __import__("math").pi,
    "e":     __import__("math").e,
    "floor": __import__("math").floor,
    "ceil":  __import__("math").ceil,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Recursively evaluate a whitelisted AST node. Raises ValueError on anything unsafe."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _SAFE_FUNCTIONS:
        return _SAFE_FUNCTIONS[node.id]          # type: ignore[return-value]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        func = _safe_eval(node.func)
        args = [_safe_eval(a) for a in node.args]
        if callable(func):
            return func(*args)
    raise ValueError(f"Unsafe or unsupported expression: {ast.dump(node)}")


@tool
def evaluate_math_tool(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression string.

    Supports: +, -, *, /, **, %, unary minus/plus, and common math functions
    (abs, round, sqrt, log, log10, sin, cos, tan, floor, ceil) plus constants
    pi and e.

    Args:
        expression: A math expression string, e.g. "sqrt(2) * pi" or "2**10".

    Returns:
        {"expression": str, "result": float | int, "error": None}
        or
        {"expression": str, "result": None, "error": str}
    """
    try:
        tree   = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return {"expression": expression, "result": result, "error": None}
    except ZeroDivisionError:
        return {"expression": expression, "result": None, "error": "Division by zero"}
    except (ValueError, TypeError, SyntaxError) as exc:
        return {"expression": expression, "result": None, "error": str(exc)}


# ── Export list for node imports ──────────────────────────────────────────────

RAG_TOOLS  = [rag_search_tool]
MATH_TOOLS = [evaluate_math_tool]