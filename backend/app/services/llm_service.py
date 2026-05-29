import os
from typing import AsyncGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from app.services.logger_service import logger

load_dotenv()

RAG_SYSTEM_PROMPT = """You are a helpful assistant and an expert on the book 'Three Days of Happiness'.
You will be given retrieved passages from the book and the conversation history, followed by the user's question.
Use ONLY the provided context to answer. If the context doesn't contain enough information, say so honestly.
Keep answers concise but informative. If asked anything about other topics, do not answer them and reject gracefully"""

REFORMULATION_SYSTEM_PROMPT = """Given the conversation history and the user's latest question, formulate a standalone question that can be understood entirely on its own.
Do NOT answer the question. Just rewrite it to resolve any pronouns (he, she, it) or implicit references into explicit names and subjects based on the history.
If the question is already standalone, return it exactly as is."""


class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.output_parser = StrOutputParser()

    def _build_history_messages(self, history: list[dict]) -> list:
        """Convert stored history dicts into LangChain message objects."""
        messages = []
        for m in history:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AIMessage(content=m["content"]))
        return messages

    def _build_rag_messages(
        self,
        question: str,
        context_docs: list[dict],
        history: list[dict] | None = None,
    ) -> list:
        """Shared message assembly for both streaming and non-streaming RAG calls."""
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            meta = doc.get("metadata", {})
            idx = meta.get("chunk_index", "Unknown")
            context_parts.append(f"[{i}] [Chunk Index: {idx}]\n{doc['page_content']}")
        context_text = "\n\n".join(context_parts)

        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]

        if history:
            logger.info(f"Injecting {len(history)} previous messages as conversation history.")
            messages += self._build_history_messages(history)

        messages.append(
            HumanMessage(
                content=f"Retrieved book context:\n{context_text}\n\nUser question: {question}"
            )
        )
        return messages

    async def get_simple_response(self, prompt: str) -> str:
        messages = [
            SystemMessage(content="You are an expert AI Engineer. Answer clearly and technically."),
            HumanMessage(content=prompt),
        ]
        chain = self.llm | self.output_parser
        return await chain.ainvoke(messages)

    async def get_rag_response(
        self,
        question: str,
        context_docs: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        """
        Answer `question` grounded in `context_docs`, optionally with conversation history.
        Returns the full response string (non-streaming).
        """
        logger.info(f"Assembling prompt for LLM with {len(context_docs)} context chunks.")
        messages = self._build_rag_messages(question, context_docs, history)

        chain = self.llm | self.output_parser
        logger.debug("Awaiting LLM generation...")
        result = await chain.ainvoke(messages)
        logger.info("LLM generation complete.")
        return result

    async def stream_rag_response(
        self,
        question: str,
        context_docs: list[dict],
        history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Answer `question` grounded in `context_docs` as an async token stream.
        Yields string chunks as the LLM produces them.
        """
        logger.info(f"Assembling streaming prompt with {len(context_docs)} context chunks.")
        messages = self._build_rag_messages(question, context_docs, history)

        chain = self.llm | self.output_parser
        logger.debug("Starting LLM stream...")
        async for chunk in chain.astream(messages):
            yield chunk
        logger.info("LLM stream complete.")

    async def reformulate_query(self, question: str, history: list[dict] | None = None) -> str:
        """Rewrites the query using chat history to resolve context/pronouns."""
        if not history:
            return question

        logger.debug("Reformulating query to resolve context...")
        messages = [SystemMessage(content=REFORMULATION_SYSTEM_PROMPT)]
        messages += self._build_history_messages(history)
        messages.append(HumanMessage(content=f"Latest question: {question}"))

        chain = self.llm | self.output_parser
        reformulated = await chain.ainvoke(messages)

        return reformulated.strip(' "\'')


llm_service = LLMService()