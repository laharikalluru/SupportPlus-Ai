"""
memory_service.py - User interaction memory (Memento pattern).

Current backend: SQLite (zero-dependency, file-based, production-ready for
moderate scale).

Upgrade path:
  - Set MEMORY_BACKEND=redis → Uses Redis for distributed, TTL-aware memory.
  - Set MEMORY_BACKEND=mongodb → Uses MongoDB for rich querying (future).

Each interaction is stored as:
  (user_id, query, response, timestamp)
"""

import logging
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from app.config import (
    MEMORY_BACKEND,
    MEMORY_DB_PATH,
    MEMORY_MAX_HISTORY,
    REDIS_URL,
    LLM_MAX_CHARS_PER_MEM,
)

logger = logging.getLogger("supportplus.memory")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# SQLite backend (default)
# ---------------------------------------------------------------------------

class SQLiteMemoryBackend:
    """
    Lightweight persistent memory using SQLite.
    Thread-safe via check_same_thread=False + WAL mode.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._init_schema()
        logger.info("SQLite memory backend initialised at %s", db_path)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self._db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT    NOT NULL,
                    query       TEXT    NOT NULL,
                    response    TEXT    NOT NULL,
                    metadata    TEXT    DEFAULT '{}',
                    created_at  TEXT    NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON interactions (user_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_created ON interactions (created_at)")

    def save(self, user_id: str, query: str, response: str, metadata: Dict[str, Any] = None) -> int:
        metadata = metadata or {}
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO interactions (user_id, query, response, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, query, response, json.dumps(metadata), _utc_now()),
            )
            return cur.lastrowid

    def get_history(self, user_id: str, limit: int = MEMORY_MAX_HISTORY) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, user_id, query, response, metadata, created_at "
                "FROM interactions WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        result = []
        for row in reversed(rows):             # chronological order
            result.append({
                "id": row["id"],
                "user_id": row["user_id"],
                "query": row["query"],
                "response": row["response"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "created_at": row["created_at"],
            })
        return result

    def delete_user(self, user_id: str) -> int:
        with self._conn() as con:
            cur = con.execute("DELETE FROM interactions WHERE user_id = ?", (user_id,))
            return cur.rowcount

    def stats(self) -> Dict[str, Any]:
        with self._conn() as con:
            total = con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            users = con.execute("SELECT COUNT(DISTINCT user_id) FROM interactions").fetchone()[0]
        return {"total_interactions": total, "unique_users": users}


# ---------------------------------------------------------------------------
# Redis backend (upgrade path – requires `redis` package)
# ---------------------------------------------------------------------------

class RedisMemoryBackend:
    """
    Redis-backed memory store.
    Stores a ZSET per user keyed by timestamp score.
    Requires: pip install redis
    """

    def __init__(self, redis_url: str) -> None:
        try:
            import redis as redis_lib
            self._redis = redis_lib.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("Redis memory backend connected at %s", redis_url)
        except Exception as exc:
            logger.error("Redis connection failed: %s – falling back to in-memory.", exc)
            raise

    def _key(self, user_id: str) -> str:
        return f"supportplus:history:{user_id}"

    def save(self, user_id: str, query: str, response: str, metadata: Dict[str, Any] = None) -> int:
        record = json.dumps({
            "query": query,
            "response": response,
            "metadata": metadata or {},
            "created_at": _utc_now(),
        })
        score = datetime.now(timezone.utc).timestamp()
        self._redis.zadd(self._key(user_id), {record: score})
        # Trim to max history size
        self._redis.zremrangebyrank(self._key(user_id), 0, -(MEMORY_MAX_HISTORY + 1))
        return -1  # Redis doesn't return auto-increment IDs

    def get_history(self, user_id: str, limit: int = MEMORY_MAX_HISTORY) -> List[Dict[str, Any]]:
        raw = self._redis.zrangebyscore(
            self._key(user_id), "-inf", "+inf",
            start=0, num=limit,
        )
        results = []
        for item in raw:
            entry = json.loads(item)
            entry["user_id"] = user_id
            results.append(entry)
        return results

    def delete_user(self, user_id: str) -> int:
        return self._redis.delete(self._key(user_id))

    def stats(self) -> Dict[str, Any]:
        keys = self._redis.keys("supportplus:history:*")
        return {"unique_users": len(keys)}


# ---------------------------------------------------------------------------
# MemoryService facade
# ---------------------------------------------------------------------------

class MemoryService:
    """
    Public facade for the memory system.
    Backend selection is controlled by the MEMORY_BACKEND env variable.
    """

    def __init__(self) -> None:
        self._backend = self._create_backend()

    def _create_backend(self):
        backend = MEMORY_BACKEND.lower()
        if backend == "redis":
            try:
                return RedisMemoryBackend(REDIS_URL)
            except Exception:
                logger.warning("Falling back to SQLite memory backend.")
                return SQLiteMemoryBackend(MEMORY_DB_PATH)
        else:
            return SQLiteMemoryBackend(MEMORY_DB_PATH)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def save_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a user interaction to the memory backend."""
        try:
            self._backend.save(user_id, query, response, metadata or {})
            logger.debug("Saved interaction for user '%s'.", user_id)
        except Exception as exc:
            logger.error("Failed to save interaction: %s", exc, exc_info=True)

    def get_history(self, user_id: str, limit: int = MEMORY_MAX_HISTORY) -> List[Dict[str, Any]]:
        """Retrieve the most recent interactions for a user."""
        try:
            return self._backend.get_history(user_id, limit)
        except Exception as exc:
            logger.error("Failed to retrieve history for '%s': %s", user_id, exc)
            return []

    def format_memory_context(self, user_id: str, limit: int = 5) -> str:
        """
        Format recent history as a concise context string to inject into
        the LLM prompt, so the model is aware of prior conversation.
        """
        history = self.get_history(user_id, limit)
        if not history:
            return ""

        lines = ["## Recent Conversation History"]
        for entry in history[-limit:]:
            lines.append(f"User: {entry['query']}")
            # Truncate long responses to keep the context window lean
            cap = max(120, LLM_MAX_CHARS_PER_MEM)
            response_preview = entry["response"][:cap]
            if len(entry["response"]) > cap:
                response_preview += "…"
            lines.append(f"Assistant: {response_preview}")
        return "\n".join(lines)

    def delete_user_data(self, user_id: str) -> int:
        """Remove all stored interactions for a user (GDPR support)."""
        try:
            count = self._backend.delete_user(user_id)
            logger.info("Deleted %d interactions for user '%s'.", count, user_id)
            return count
        except Exception as exc:
            logger.error("Failed to delete user data: %s", exc)
            return 0

    def health(self) -> Dict[str, Any]:
        """Return health stats."""
        try:
            stats = self._backend.stats()
            return {"status": "ok", "backend": MEMORY_BACKEND, **stats}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_memory_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
