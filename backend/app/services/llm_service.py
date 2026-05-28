import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

RAG_SYSTEM_PROMPT = """You are a helpful movie recommendation assistant with access to a curated database of films.
You will be given retrieved movie context and the conversation history, then the user's latest question.
Use ONLY the provided context to answer. If the context doesn't contain enough information, say so honestly.
When you mention a movie title, format it in bold.
Keep answers concise but informative."""


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

        history items: {role: "user"|"assistant", content: str}
        """
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            meta = doc.get("metadata", {})
            genres = ", ".join(meta.get("genres", [])) or "N/A"
            context_parts.append(
                f"[{i}] {doc['page_content']}\n"
                f"    Genres: {genres} | Rating: {meta.get('vote_average', 'N/A')} "
                f"| Release: {meta.get('release_date', 'N/A')}"
            )
        context_text = "\n\n".join(context_parts)

        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]

        if history:
            messages += self._build_history_messages(history)

        messages.append(
            HumanMessage(
                content=f"Retrieved movie context:\n{context_text}\n\nUser question: {question}"
            )
        )

        chain = self.llm | self.output_parser
        return await chain.ainvoke(messages)


llm_service = LLMService()