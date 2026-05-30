"""
agent_nodes.py
──────────────
The four LangGraph nodes. Each is an async callable that receives the full
AgentState, does its work, and returns a *partial* state dict — LangGraph
merges it into the shared state automatically.

Node responsibilities:
  RouterNode       → classify the user's intent → sets `route`
  RAGNode          → book Q&A with cache-aware hybrid search → sets `rag_result`,
                     `rag_sufficient`, and `final_response` (if sufficient)
  WebSearchNode    → web search + synthesis → sets `web_result` + `final_response`
  MathNode         → expression extraction + evaluation → sets `math_result` + `final_response`

LLM assignments:
  Router         → Groq  / llama-3.3-70b-versatile   (fast, cheap classification)
  RAG Agent      → Gemini / gemini-2.5-flash          (same ecosystem as embeddings)
  Web Search     → Groq  / llama-3.3-70b-versatile   (good at summarisation)
  Math Agent     → Groq  / gemma2-9b-it               (overkill for tool-calling)
"""

from __future__ import annotations
import os
import time
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.agent.agent_state import AgentState, RouteTarget, TraceEntry
from app.agent.agent_tools import (
    rag_search_tool,
    brave_web_search_tool,
    evaluate_math_tool,
    RAG_TOOLS, WEB_SEARCH_TOOLS, MATH_TOOLS,
)
from app.services.logger_service import logger
from app.services.llm_service import llm_service   # reuse reformulate_query

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _groq(model: str = "llama-3.3-70b-versatile", temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(
        model=model,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        max_retries=2,
    )


def _gemini(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temperature,
        max_retries=2,
    )


def _trace(node: str, action: str, detail: str | None = None) -> TraceEntry:
    return TraceEntry(node=node, action=action, detail=detail)


def _append_trace(state: AgentState, entry: TraceEntry) -> list[TraceEntry]:
    """Return an updated trace list (never mutate state directly)."""
    return [*state.get("agent_trace", []), entry]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Router Node
# ─────────────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a routing assistant. Classify the user's message into exactly one of these categories:

  rag         — questions about the book "Three Days of Happiness" or its characters/plot
  web_search  — questions that require current or general world knowledge
  math        — requests to compute, calculate, or evaluate a mathematical expression
  unknown     — anything that doesn't fit the above

Respond with ONLY one word: rag, web_search, math, or unknown. No explanation."""


async def router_node(state: AgentState) -> dict:
    """
    Classifies `state["input"]` and sets `state["route"]`.
    No tools — pure LLM classification.
    """
    user_input = state["input"]
    logger.info(f"[RouterNode] Classifying: '{user_input[:80]}'")

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    llm = _groq()
    chain = llm | StrOutputParser()
    raw: str = await chain.ainvoke(messages)
    route_str = raw.strip().lower()

    # Normalise to a valid RouteTarget, fall back to "unknown"
    valid: set[RouteTarget] = {"rag", "web_search", "math", "unknown"}
    route: RouteTarget = route_str if route_str in valid else "unknown"  # type: ignore[assignment]

    logger.info(f"[RouterNode] Route decision: '{route}' (raw: '{raw.strip()}')")

    return {
        "route": route,
        "agent_trace": _append_trace(state, _trace("router", "classified", route)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RAG Agent Node
# ─────────────────────────────────────────────────────────────────────────────

RAG_AGENT_SYSTEM_PROMPT = """You are a helpful assistant and expert on the book 'Three Days of Happiness'.
You will be given retrieved passages from the book and the conversation history, followed by the user's question.
Use ONLY the provided context to answer. If the context doesn't contain enough information, say so honestly.
Keep answers concise but informative. Do not answer questions about other topics."""

RAG_SUFFICIENCY_PROMPT = """Based on the retrieved context and the answer you just produced, is the answer
sufficient and well-grounded in the source material?

Answer with ONLY one word: yes or no."""


async def rag_node(state: AgentState) -> dict:
    """
    Runs the cache-aware hybrid RAG pipeline.

    Sets:
      rag_result       — retrieval output + timing
      rag_sufficient   — True if the answer is grounded; False to trigger fallback
      final_response   — populated only when rag_sufficient is True
    """
    user_input = state["input"]
    history    = state.get("history") or []
    logger.info(f"[RAGNode] Handling question: '{user_input[:80]}'")

    # ── Query reformulation (reuse existing service) ──────────────────────────
    search_query = user_input
    if history:
        search_query = await llm_service.reformulate_query(user_input, history)
        logger.info(f"[RAGNode] Reformulated: '{search_query}'")

    # ── Tool call: retrieve chunks ────────────────────────────────────────────
    t0 = time.perf_counter()
    tool_output: dict = await rag_search_tool.ainvoke({"query": search_query})
    retrieval_ms = (time.perf_counter() - t0) * 1000

    chunks: list[dict] = tool_output.get("chunks", [])
    from_cache: bool   = tool_output.get("from_cache", False)

    # ── Early exit: cache already has a full response ─────────────────────────
    if from_cache and tool_output.get("cached_response"):
        cached_response = tool_output["cached_response"]
        logger.info("[RAGNode] Serving answer from semantic cache.")
        rag_result = {
            "response": cached_response,
            "retrieved_chunks": chunks,
            "from_cache": True,
            "cache_key": tool_output.get("cache_key"),
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "llm_latency_ms": 0.0,
        }
        return {
            "rag_result": rag_result,
            "rag_sufficient": True,
            "final_response": cached_response,
            "agent_trace": _append_trace(state, _trace("rag", "cache_hit", tool_output.get("cache_key"))),
        }

    # ── No chunks: immediately insufficient ───────────────────────────────────
    if not chunks:
        logger.info("[RAGNode] No chunks retrieved — marking insufficient.")
        return {
            "rag_result": {
                "response": "",
                "retrieved_chunks": [],
                "from_cache": False,
                "cache_key": None,
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "llm_latency_ms": 0.0,
            },
            "rag_sufficient": False,
            "agent_trace": _append_trace(state, _trace("rag", "no_chunks", None)),
        }

    # ── Build prompt and call Gemini ──────────────────────────────────────────
    context_parts = []
    for i, doc in enumerate(chunks, 1):
        meta = doc.get("metadata", {})
        idx  = meta.get("chunk_index", "?")
        context_parts.append(f"[{i}] [Chunk {idx}]\n{doc['page_content']}")
    context_text = "\n\n".join(context_parts)

    messages = [SystemMessage(content=RAG_AGENT_SYSTEM_PROMPT)]
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))
    messages.append(
        HumanMessage(content=f"Retrieved context:\n{context_text}\n\nQuestion: {user_input}")
    )

    llm = _gemini()
    t0 = time.perf_counter()
    response: str = await (llm | StrOutputParser()).ainvoke(messages)
    llm_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[RAGNode] LLM answered in {llm_ms:.0f}ms.")

    # ── Sufficiency check ─────────────────────────────────────────────────────
    # Ask the same LLM (cheap — tiny prompt) whether the answer is grounded.
    sufficiency_messages = [
        SystemMessage(content=RAG_SUFFICIENCY_PROMPT),
        HumanMessage(content=f"Context:\n{context_text}\n\nAnswer:\n{response}"),
    ]
    raw_suf: str = await (llm | StrOutputParser()).ainvoke(sufficiency_messages)
    rag_sufficient = raw_suf.strip().lower().startswith("yes")
    logger.info(f"[RAGNode] Sufficiency check: '{raw_suf.strip()}' → {rag_sufficient}")

    rag_result = {
        "response": response,
        "retrieved_chunks": chunks,
        "from_cache": False,
        "cache_key": None,
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "llm_latency_ms": round(llm_ms, 2),
    }

    result: dict = {
        "rag_result": rag_result,
        "rag_sufficient": rag_sufficient,
        "agent_trace": _append_trace(
            state,
            _trace("rag", "answered" if rag_sufficient else "insufficient", f"{len(chunks)} chunks")
        ),
    }
    if rag_sufficient:
        result["final_response"] = response

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Web Search Agent Node
# ─────────────────────────────────────────────────────────────────────────────

WEB_SEARCH_SYSTEM_PROMPT = """You are a helpful assistant. You will be given web search results and the user's question.
Synthesise the results into a clear, accurate answer. Cite sources where relevant.
If the results don't answer the question, say so honestly."""


async def web_search_node(state: AgentState) -> dict:
    """
    Calls the Brave Search tool, then synthesises the results with Groq.

    Sets:
      web_result     — search results + synthesis + timing
      final_response — the synthesised answer
    """
    user_input = state["input"]
    logger.info(f"[WebSearchNode] Searching for: '{user_input[:80]}'")

    # ── Tool call ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        search_output: dict = await brave_web_search_tool.ainvoke(
            {"query": user_input, "count": 5}
        )
        sources: list[dict] = search_output.get("results", [])
    except NotImplementedError:
        # Tool not yet implemented — return a placeholder so the graph doesn't crash
        logger.warning("[WebSearchNode] Brave Search not implemented yet — returning stub.")
        stub = "Web search is not yet configured. Please implement brave_web_search_tool."
        return {
            "web_result": {
                "response": stub,
                "sources": [],
                "search_latency_ms": 0.0,
                "llm_latency_ms": 0.0,
            },
            "final_response": stub,
            "agent_trace": _append_trace(state, _trace("web_search", "stub", None)),
        }
    search_ms = (time.perf_counter() - t0) * 1000

    # ── Synthesise with Groq ──────────────────────────────────────────────────
    results_text = "\n\n".join(
        f"[{i+1}] {r.get('title','')}\n{r.get('url','')}\n{r.get('description','')}"
        for i, r in enumerate(sources)
    )
    messages = [
        SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT),
        HumanMessage(content=f"Search results:\n{results_text}\n\nQuestion: {user_input}"),
    ]

    llm = _groq()
    t0 = time.perf_counter()
    response: str = await (llm | StrOutputParser()).ainvoke(messages)
    llm_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[WebSearchNode] Synthesised in {llm_ms:.0f}ms.")

    web_result = {
        "response": response,
        "sources": sources,
        "search_latency_ms": round(search_ms, 2),
        "llm_latency_ms": round(llm_ms, 2),
    }

    return {
        "web_result": web_result,
        "final_response": response,
        "agent_trace": _append_trace(state, _trace("web_search", "answered", f"{len(sources)} sources")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Math Agent Node
# ─────────────────────────────────────────────────────────────────────────────

MATH_EXTRACTION_PROMPT = """Extract ONLY the mathematical expression from the user's message.
Return ONLY the bare expression (e.g. "sqrt(2) * pi" or "2**10 + 5").
Do NOT add any explanation, units, or text. If no valid expression is present, return "NONE"."""

MATH_FORMAT_PROMPT = """The user asked a math question. The expression was evaluated and returned a result.
Present the result clearly and conversationally. Include the expression and its value."""


async def math_node(state: AgentState) -> dict:
    """
    Extracts the math expression from the input, evaluates it safely,
    then formats a natural-language response.

    Sets:
      math_result    — expression, raw result, timing
      final_response — formatted answer
    """
    user_input = state["input"]
    logger.info(f"[MathNode] Processing: '{user_input[:80]}'")

    llm = _groq(model="gemma2-9b-it")

    # ── Step 1: extract expression ────────────────────────────────────────────
    extract_messages = [
        SystemMessage(content=MATH_EXTRACTION_PROMPT),
        HumanMessage(content=user_input),
    ]
    expression: str = await (llm | StrOutputParser()).ainvoke(extract_messages)
    expression = expression.strip()
    logger.info(f"[MathNode] Extracted expression: '{expression}'")

    if expression.upper() == "NONE" or not expression:
        fallback = "I couldn't identify a mathematical expression in your message. Please provide a clear expression to evaluate."
        return {
            "math_result": {"response": fallback, "expression": "", "raw_result": ""},
            "final_response": fallback,
            "agent_trace": _append_trace(state, _trace("math", "no_expression", None)),
        }

    # ── Step 2: evaluate via tool ─────────────────────────────────────────────
    tool_output: dict = evaluate_math_tool.invoke({"expression": expression})
    raw_result = str(tool_output.get("result"))
    error      = tool_output.get("error")

    if error:
        logger.warning(f"[MathNode] Eval error: {error}")
        error_response = f"I couldn't evaluate '{expression}': {error}"
        return {
            "math_result": {"response": error_response, "expression": expression, "raw_result": ""},
            "final_response": error_response,
            "agent_trace": _append_trace(state, _trace("math", "eval_error", error)),
        }

    # ── Step 3: format the response ───────────────────────────────────────────
    format_messages = [
        SystemMessage(content=MATH_FORMAT_PROMPT),
        HumanMessage(content=f"Question: {user_input}\nExpression: {expression}\nResult: {raw_result}"),
    ]
    formatted: str = await (llm | StrOutputParser()).ainvoke(format_messages)
    logger.info(f"[MathNode] Result: {expression} = {raw_result}")

    math_result = {
        "response": formatted,
        "expression": expression,
        "raw_result": raw_result,
    }

    return {
        "math_result": math_result,
        "final_response": formatted,
        "agent_trace": _append_trace(state, _trace("math", "evaluated", f"{expression} = {raw_result}")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Unknown / Fallback Node
# ─────────────────────────────────────────────────────────────────────────────

async def unknown_node(state: AgentState) -> dict:
    """
    Graceful fallback for queries the router couldn't classify.
    """
    fallback = (
        "I'm not sure how to help with that. I can answer questions about the book "
        "'Three Days of Happiness', search the web for general topics, or evaluate "
        "mathematical expressions."
    )
    return {
        "final_response": fallback,
        "agent_trace": _append_trace(state, _trace("unknown", "fallback", None)),
    }