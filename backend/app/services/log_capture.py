"""
Per-request log capture for SSE streaming.

Attaches a lightweight handler to the app logger for the duration of a
streaming request, buffers records in a thread-safe list, and lets the
generator drain them between awaits so they appear in the SSE stream as
  {"type": "log", "level": "INFO", "module": "qdrant_service", "message": "..."}

Thread safety: logging.Handler.emit() can be called from any thread
(LangChain / httpx use thread pools internally). We use a threading.Lock
so drain() never races with emit().
"""

import logging
import threading
from contextlib import contextmanager


class StreamLogHandler(logging.Handler):
    def __init__(self, min_level: int = logging.DEBUG):
        super().__init__(level=min_level)
        self._buffer: list[dict] = []
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        with self._lock:
            self._buffer.append({
                "type":    "log",
                "level":   record.levelname,       # "INFO", "DEBUG", "WARNING", "ERROR"
                "module":  record.module,           # e.g. "qdrant_service"
                "message": record.getMessage(),
                "ts":      record.created,          # Unix timestamp float
            })

    def drain(self) -> list[dict]:
        """Return and clear all buffered log events."""
        with self._lock:
            events, self._buffer = self._buffer, []
            return events


@contextmanager
def capture_logs(target_logger: logging.Logger, min_level: int = logging.DEBUG):
    """
    Context manager that attaches a StreamLogHandler for the duration of
    a streaming request and guarantees cleanup on exit.

    Usage:
        with capture_logs(logger) as log_handler:
            await do_something()
            for event in log_handler.drain():
                yield _sse(event)
    """
    handler = StreamLogHandler(min_level=min_level)
    target_logger.addHandler(handler)
    try:
        yield handler
    finally:
        target_logger.removeHandler(handler)