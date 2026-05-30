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
  WebSearchNode    → Brave MCP search + Groq synthesis → sets `web_result` + `final_response`
  MathNode         → expression extraction + evaluation → sets `math_result` + `final_response`

LLM assignments:
  Router         → Groq  / llama-3.3-70b-versatile   (fast classification)
  RAG Agent      → Gemini / gemini-2.5-flash          (same ecosystem as embeddings)
  Web Search     → Groq  / llama-3.3-70b-versatile   (synthesis)
  Math Agent     → Groq  / gemma2-9b-it               (extraction + formatting)
"""

from __future__ import annotations
import os
import time
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.agent.agent_state import AgentState, RouteTarget, TraceEntry
from app.agent.agent_tools import rag_search_tool, evaluate_math_tool
from app.services.logger_service import logger
from app.services.llm_service import llm_service

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

class RouterDecision(BaseModel):
    """Structured output model for the router LLM call."""
    route: Literal["rag", "web_search", "math", "unknown"] = Field(
        description="Which agent should handle this query"
    )
    reasoning: str = Field(
        description="One sentence explaining why this route was chosen"
    )


ROUTER_SYSTEM_PROMPT = """You are a routing assistant. Classify the user's message into exactly one of these categories:

  rag         — questions about the book "Three Days of Happiness", its characters, plot, or themes
  web_search  — questions that require current or general world knowledge not related to the book
  math        — requests to compute, calculate, or evaluate a mathematical expression
  unknown     — anything that doesn't clearly fit the above

Choose the single best category."""


async def router_node(state: AgentState) -> dict:
    """
    Classifies `state["input"]` and sets `state["route"]`.
    Uses structured output — no string parsing, no fallback validation needed.
    """
    user_input = state["input"]
    logger.info(f"[RouterNode] Classifying: '{user_input[:80]}'")

    llm = _groq().with_structured_output(RouterDecision)
    decision: RouterDecision = await llm.ainvoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ])

    logger.info(f"[RouterNode] Route: '{decision.route}' — {decision.reasoning}")

    return {
        "route": decision.route,
        "agent_trace": _append_trace(
            state,
            _trace("router", "classified", f"{decision.route} | {decision.reasoning}"),
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RAG Agent Node
# ─────────────────────────────────────────────────────────────────────────────

class SufficiencyCheck(BaseModel):
    """Structured output model for the RAG sufficiency check."""
    sufficient: bool = Field(
        description="True if the answer is well-grounded in the retrieved context"
    )
    reason: str = Field(
        description="One sentence explaining why the answer is or isn't sufficient"
    )


RAG_AGENT_SYSTEM_PROMPT = """You are a helpful assistant and expert on the book 'Three Days of Happiness'.
You will be given retrieved passages from the book and the conversation history, followed by the user's question.
Use ONLY the provided context to answer. If the context doesn't contain enough information, say so honestly.
Keep answers concise but informative. Do not answer questions about other topics."""

RAG_SUFFICIENCY_SYSTEM_PROMPT = """Evaluate whether the given answer is well-grounded in the retrieved context.
An answer is sufficient if it directly addresses the question using information from the context.
An answer is NOT sufficient if it says it cannot find the information, makes things up, or is vague."""


async def rag_node(state: AgentState) -> dict:
    """
    Runs the cache-aware hybrid RAG pipeline.

    Sets:
      rag_result       — retrieval output + timing
      rag_sufficient   — True if the answer is grounded; False triggers web search fallback
      final_response   — populated only when rag_sufficient is True
    """
    user_input = state["input"]
    history    = state.get("history") or []
    logger.info(f"[RAGNode] Handling question: '{user_input[:80]}'")

    # ── Query reformulation ───────────────────────────────────────────────────
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

    # ── Early exit: cache hit with full response ──────────────────────────────
    if from_cache and tool_output.get("cached_response"):
        cached_response = tool_output["cached_response"]
        logger.info("[RAGNode] Serving answer from semantic cache.")
        return {
            "rag_result": {
                "response": cached_response,
                "retrieved_chunks": chunks,
                "from_cache": True,
                "cache_key": tool_output.get("cache_key"),
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "llm_latency_ms": 0.0,
            },
            "rag_sufficient": True,
            "final_response": cached_response,
            "agent_trace": _append_trace(state, _trace("rag", "cache_hit", tool_output.get("cache_key"))),
        }

    # ── No chunks: immediately insufficient ──────────────────────────────────
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
        idx = doc.get("metadata", {}).get("chunk_index", "?")
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

    # ── Sufficiency check (structured output) ─────────────────────────────────
    sufficiency_llm = _gemini().with_structured_output(SufficiencyCheck)
    check: SufficiencyCheck = await sufficiency_llm.ainvoke([
        SystemMessage(content=RAG_SUFFICIENCY_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {user_input}\n\nAnswer:\n{response}"),
    ])
    logger.info(f"[RAGNode] Sufficiency: {check.sufficient} — {check.reason}")

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
        "rag_sufficient": check.sufficient,
        "agent_trace": _append_trace(
            state,
            _trace("rag", "answered" if check.sufficient else "insufficient", check.reason),
        ),
    }
    if check.sufficient:
        result["final_response"] = response

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Web Search Agent Node  (Brave MCP)
# ─────────────────────────────────────────────────────────────────────────────

WEB_SEARCH_SYSTEM_PROMPT = """You are a helpful assistant. You will be given web search results and the user's question.
Synthesise the results into a clear, accurate answer. Cite sources where relevant using [1], [2] notation.
If the results don't answer the question, say so honestly."""


def _build_mcp_client() -> MultiServerMCPClient:
    """
    Builds the MCP client that spawns the Brave Search MCP server as a
    local stdio process.

    Requirements:
      - Node.js installed on the server
      - BRAVE_API_KEY in environment
      - npm package @modelcontextprotocol/server-brave-search (auto-installed by npx)
    """
    return MultiServerMCPClient({
        "brave": {
            "transport":  "stdio",
            "command":    "npx",
            "args":       ["-y", "@modelcontextprotocol/server-brave-search"],
            "env":        {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")},
        }
    })


async def web_search_node(state: AgentState) -> dict:
    """
    Searches the web via the Brave MCP server, then synthesises with Groq.

    The MCP client spawns the Brave server as a stdio subprocess, fetches
    the LangChain-compatible tool list, binds them to the LLM, and lets the
    LLM decide how to call the tool and interpret the results.

    Sets:
      web_result     — sources + synthesis + timing
      final_response — the synthesised answer
    """
    user_input = state["input"]
    logger.info(f"[WebSearchNode] Searching for: '{user_input[:80]}'")

    # ── 1. Spin up MCP client and fetch tools ─────────────────────────────────
    t0 = time.perf_counter()
    try:
        async with _build_mcp_client() as mcp:
            brave_tools = await mcp.get_tools()
            logger.info(f"[WebSearchNode] MCP tools available: {[t.name for t in brave_tools]}")

            # ── 2. Bind tools to Groq and let the LLM run the search ─────────
            llm = _groq().bind_tools(brave_tools)
            tool_call_response = await llm.ainvoke([
                SystemMessage(content="Use the search tool to find relevant results for the user's query."),
                HumanMessage(content=user_input),
            ])

            # ── 3. Execute whatever tool call the LLM chose ───────────────────
            sources: list[dict] = []
            raw_results = ""

            if tool_call_response.tool_calls:
                for tool_call in tool_call_response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    logger.info(f"[WebSearchNode] LLM called tool '{tool_name}' with args: {tool_args}")

                    # Find the matching tool from MCP and invoke it
                    matching_tool = next((t for t in brave_tools if t.name == tool_name), None)
                    if matching_tool:
                        tool_result = await matching_tool.ainvoke(tool_args)
                        raw_results = str(tool_result)

                        # Parse the results into a consistent shape for the debug payload
                        # Brave returns a list of {title, url, description} dicts
                        if isinstance(tool_result, list):
                            sources = [
                                {
                                    "title":       r.get("title", ""),
                                    "url":         r.get("url", ""),
                                    "description": r.get("description", ""),
                                }
                                for r in tool_result
                                if isinstance(r, dict)
                            ]
                        elif isinstance(tool_result, dict):
                            # Some MCP servers wrap results in a top-level key
                            web_results = tool_result.get("web", {}).get("results", [])
                            sources = [
                                {
                                    "title":       r.get("title", ""),
                                    "url":         r.get("url", ""),
                                    "description": r.get("description", ""),
                                }
                                for r in web_results
                            ]

    except Exception as e:
        logger.error(f"[WebSearchNode] MCP error: {e}")
        error_response = f"Web search failed: {e}"
        return {
            "web_result": {
                "response": error_response,
                "sources": [],
                "search_latency_ms": 0.0,
                "llm_latency_ms": 0.0,
            },
            "final_response": error_response,
            "agent_trace": _append_trace(state, _trace("web_search", "mcp_error", str(e))),
        }

    search_ms = (time.perf_counter() - t0) * 1000

    # ── 4. Synthesise results with Groq ───────────────────────────────────────
    results_text = raw_results if raw_results else "No results found."
    t0 = time.perf_counter()
    synthesis_llm = _groq()
    response: str = await (synthesis_llm | StrOutputParser()).ainvoke([
        SystemMessage(content=WEB_SEARCH_SYSTEM_PROMPT),
        HumanMessage(content=f"Search results:\n{results_text}\n\nQuestion: {user_input}"),
    ])
    llm_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[WebSearchNode] Synthesised {len(sources)} sources in {llm_ms:.0f}ms.")

    return {
        "web_result": {
            "response": response,
            "sources": sources,
            "search_latency_ms": round(search_ms, 2),
            "llm_latency_ms": round(llm_ms, 2),
        },
        "final_response": response,
        "agent_trace": _append_trace(state, _trace("web_search", "answered", f"{len(sources)} sources")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Math Agent Node
# ─────────────────────────────────────────────────────────────────────────────

class MathExtraction(BaseModel):
    """Structured output model for extracting a math expression."""
    expression: str = Field(
        description="The bare mathematical expression to evaluate, e.g. 'sqrt(2) * pi'. "
                    "Return an empty string if no valid expression is present."
    )
    found: bool = Field(
        description="True if a valid mathematical expression was found in the message"
    )


MATH_FORMAT_PROMPT = """The user asked a math question. The expression was evaluated and returned a result.
Present the result clearly and conversationally. Include the expression and its value."""


async def math_node(state: AgentState) -> dict:
    """
    Extracts the math expression via structured output, evaluates it safely,
    then formats a natural-language response.
    """
    user_input = state["input"]
    logger.info(f"[MathNode] Processing: '{user_input[:80]}'")

    llm = _groq(model="gemma2-9b-it")

    # ── Step 1: extract expression (structured output) ────────────────────────
    extraction_llm = llm.with_structured_output(MathExtraction)
    extraction: MathExtraction = await extraction_llm.ainvoke([
        SystemMessage(content="Extract the mathematical expression from the user's message. "
                              "Return only the bare expression (e.g. 'sqrt(2) * pi' or '2**10 + 5'). "
                              "Set found=false if there is no valid math expression."),
        HumanMessage(content=user_input),
    ])
    logger.info(f"[MathNode] Extracted: '{extraction.expression}' (found={extraction.found})")

    if not extraction.found or not extraction.expression:
        fallback = "I couldn't identify a mathematical expression in your message. Please provide a clear expression to evaluate."
        return {
            "math_result": {"response": fallback, "expression": "", "raw_result": ""},
            "final_response": fallback,
            "agent_trace": _append_trace(state, _trace("math", "no_expression", None)),
        }

    # ── Step 2: evaluate via tool ─────────────────────────────────────────────
    tool_output: dict = evaluate_math_tool.invoke({"expression": extraction.expression})
    raw_result = str(tool_output.get("result"))
    error      = tool_output.get("error")

    if error:
        logger.warning(f"[MathNode] Eval error: {error}")
        error_response = f"I couldn't evaluate '{extraction.expression}': {error}"
        return {
            "math_result": {"response": error_response, "expression": extraction.expression, "raw_result": ""},
            "final_response": error_response,
            "agent_trace": _append_trace(state, _trace("math", "eval_error", error)),
        }

    # ── Step 3: format the response ───────────────────────────────────────────
    formatted: str = await (llm | StrOutputParser()).ainvoke([
        SystemMessage(content=MATH_FORMAT_PROMPT),
        HumanMessage(content=f"Question: {user_input}\nExpression: {extraction.expression}\nResult: {raw_result}"),
    ])
    logger.info(f"[MathNode] {extraction.expression} = {raw_result}")

    return {
        "math_result": {
            "response": formatted,
            "expression": extraction.expression,
            "raw_result": raw_result,
        },
        "final_response": formatted,
        "agent_trace": _append_trace(state, _trace("math", "evaluated", f"{extraction.expression} = {raw_result}")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Unknown / Fallback Node
# ─────────────────────────────────────────────────────────────────────────────

async def unknown_node(state: AgentState) -> dict:
    fallback = (
        "I'm not sure how to help with that. I can answer questions about the book "
        "'Three Days of Happiness', search the web for general topics, or evaluate "
        "mathematical expressions."
    )
    return {
        "final_response": fallback,
        "agent_trace": _append_trace(state, _trace("unknown", "fallback", None)),
    }