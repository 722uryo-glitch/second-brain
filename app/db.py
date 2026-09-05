import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from .config import DB_PATH

_lock = threading.RLock()


def _connect():
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                embedding_json TEXT,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
            """
        )
        conn.commit()


def add_memory(kind, source, content, importance=0.5, embedding=None, metadata=None):
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO memories(created_at, kind, source, content, importance, embedding_json, metadata_json) VALUES(?,?,?,?,?,?,?)",
            (
                now,
                kind,
                source,
                content,
                float(importance),
                json.dumps(embedding) if embedding is not None else None,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return cur.lastrowid


def recent_memories(limit=30):
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


def all_memories_with_embeddings(limit=2000):
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM memories WHERE embedding_json IS NOT NULL ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["embedding"] = json.loads(d.pop("embedding_json"))
        except Exception:
            continue
        result.append(d)
    return result


def delete_memory(memory_id: int):
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
