"""
In-memory session store.
Holds conversation history per session_id.
All data lives in process memory — intentionally ephemeral (resets on server restart
or when the frontend starts a new session)
"""

import uuid
from datetime import datetime, timezone
from collections import defaultdict


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}
        self._created_at: dict[str, str] = {}

    def create_session(self) -> dict:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._sessions[session_id] = []
        self._created_at[session_id] = now
        return {"session_id": session_id, "created_at": now}

    def delete_session(self, session_id: str) -> bool:
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        self._created_at.pop(session_id, None)
        return existed

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found")
        self._sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_history(self, session_id: str) -> list[dict]:
        return self._sessions.get(session_id, [])

    def get_history_for_llm(self, session_id: str) -> list[dict]:
        """
        Returns history in the {role, content} format LangChain / the LLM service expects.
        Strips the timestamp field.
        """
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._sessions.get(session_id, [])
        ]


session_store = SessionStore()