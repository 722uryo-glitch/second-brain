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

            CREATE TABLE IF NOT EXISTS external_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                published_at TEXT,
                summary TEXT,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_external_items_collected_at ON external_items(collected_at DESC);
            CREATE INDEX IF NOT EXISTS idx_external_items_source ON external_items(source);

            CREATE TABLE IF NOT EXISTS intelligence_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_key TEXT NOT NULL UNIQUE,
                claim_text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'unverified',
                confidence REAL NOT NULL DEFAULT 0.0,
                first_seen TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                independent_sources INTEGER NOT NULL DEFAULT 0,
                contradictions INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_claims_updated ON intelligence_claims(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_claims_status ON intelligence_claims(status);

            CREATE TABLE IF NOT EXISTS claim_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                external_item_id INTEGER NOT NULL,
                source_domain TEXT,
                source_type TEXT,
                stance TEXT NOT NULL DEFAULT 'supports',
                credibility REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                UNIQUE(claim_id, external_item_id),
                FOREIGN KEY(claim_id) REFERENCES intelligence_claims(id),
                FOREIGN KEY(external_item_id) REFERENCES external_items(id)
            );
            CREATE INDEX IF NOT EXISTS idx_evidence_claim ON claim_evidence(claim_id);
            """
        )
        conn.commit()


def add_memory(kind, source, content, importance=0.5, embedding=None, metadata=None):
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO memories(created_at, kind, source, content, importance, embedding_json, metadata_json) VALUES(?,?,?,?,?,?,?)",
            (now, kind, source, content, float(importance), json.dumps(embedding) if embedding is not None else None,
             json.dumps(metadata or {}, ensure_ascii=False)),
        )
        conn.commit()
        return cur.lastrowid


def recent_memories(limit=30):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM memories ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def all_memories_with_embeddings(limit=2000):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM memories WHERE embedding_json IS NOT NULL ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
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


def add_external_item(source, title, url, published_at=None, summary=None, metadata=None):
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO external_items(collected_at,source,title,url,published_at,summary,metadata_json) VALUES(?,?,?,?,?,?,?)",
            (now, source, title, url, published_at, summary, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        conn.commit()
        if cur.rowcount <= 0:
            return None
        return cur.lastrowid


def recent_external_items(limit=100):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM external_items ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def unprocessed_external_items(limit=100):
    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT e.* FROM external_items e
            LEFT JOIN claim_evidence ce ON ce.external_item_id=e.id
            WHERE ce.id IS NULL
            ORDER BY e.id ASC LIMIT ?
            """, (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_claim(claim_key, claim_text, metadata=None):
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO intelligence_claims(claim_key,claim_text,first_seen,updated_at,metadata_json)
               VALUES(?,?,?,?,?) ON CONFLICT(claim_key) DO UPDATE SET updated_at=excluded.updated_at""",
            (claim_key, claim_text, now, now, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        row = conn.execute("SELECT id FROM intelligence_claims WHERE claim_key=?", (claim_key,)).fetchone()
        conn.commit()
        return int(row["id"])


def add_claim_evidence(claim_id, external_item_id, source_domain, source_type, stance="supports", credibility=0.5):
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO claim_evidence(claim_id,external_item_id,source_domain,source_type,stance,credibility,created_at) VALUES(?,?,?,?,?,?,?)",
            (claim_id, external_item_id, source_domain, source_type, stance, float(credibility), now),
        )
        conn.commit()
        recompute_claim(claim_id)


def recompute_claim(claim_id):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT source_domain,stance,credibility FROM claim_evidence WHERE claim_id=?", (claim_id,)).fetchall()
        if not rows:
            return
        domains = {r["source_domain"] for r in rows if r["source_domain"]}
        supports = [r for r in rows if r["stance"] == "supports"]
        contradicts = [r for r in rows if r["stance"] == "contradicts"]
        avg_cred = sum(float(r["credibility"]) for r in supports) / max(1, len(supports))
        independent = len(domains)
        if contradicts and supports:
            status = "disputed"
        elif independent >= 3 and avg_cred >= 0.6:
            status = "corroborated"
        elif independent >= 2:
            status = "partially_corroborated"
        else:
            status = "unverified"
        confidence = min(0.98, 0.18 * independent + 0.5 * avg_cred)
        conn.execute(
            "UPDATE intelligence_claims SET status=?,confidence=?,evidence_count=?,independent_sources=?,contradictions=?,updated_at=? WHERE id=?",
            (status, confidence, len(rows), independent, len(contradicts), datetime.now(timezone.utc).isoformat(), claim_id),
        )
        conn.commit()


def recent_claims(limit=100):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM intelligence_claims ORDER BY updated_at DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]
