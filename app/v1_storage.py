import hashlib
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


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_v1_storage():
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS external_documents (
                external_item_id INTEGER PRIMARY KEY,
                fetched_at TEXT NOT NULL,
                final_url TEXT,
                content_hash TEXT,
                body_text TEXT,
                fetch_status TEXT NOT NULL DEFAULT 'pending',
                http_status INTEGER,
                error TEXT,
                FOREIGN KEY(external_item_id) REFERENCES external_items(id)
            );
            CREATE INDEX IF NOT EXISTS idx_external_documents_status
                ON external_documents(fetch_status, fetched_at DESC);

            CREATE TABLE IF NOT EXISTS source_health (
                source TEXT PRIMARY KEY,
                source_type TEXT,
                last_attempt TEXT,
                last_success TEXT,
                last_error TEXT,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                fetched_total INTEGER NOT NULL DEFAULT 0,
                new_total INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                fetched INTEGER NOT NULL DEFAULT 0,
                new_items INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                skipped INTEGER NOT NULL DEFAULT 0,
                details_json TEXT
            );

            CREATE TABLE IF NOT EXISTS verification_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                rationale TEXT,
                evidence_json TEXT,
                FOREIGN KEY(claim_id) REFERENCES intelligence_claims(id)
            );
            CREATE INDEX IF NOT EXISTS idx_verification_audit_claim
                ON verification_audit(claim_id, created_at DESC);
            """
        )
        conn.commit()


def record_source_result(source, source_type, fetched=0, new_items=0, error=None):
    now = _now()
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM source_health WHERE source=?", (source,)).fetchone()
        failures = (int(row["consecutive_failures"]) if row else 0) + (1 if error else 0)
        if not error:
            failures = 0
        conn.execute(
            """
            INSERT INTO source_health(source,source_type,last_attempt,last_success,last_error,consecutive_failures,fetched_total,new_total)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(source) DO UPDATE SET
                source_type=excluded.source_type,
                last_attempt=excluded.last_attempt,
                last_success=CASE WHEN excluded.last_error IS NULL THEN excluded.last_attempt ELSE source_health.last_success END,
                last_error=excluded.last_error,
                consecutive_failures=excluded.consecutive_failures,
                fetched_total=source_health.fetched_total+excluded.fetched_total,
                new_total=source_health.new_total+excluded.new_total
            """,
            (source, source_type, now, None if error else now, error, failures, int(fetched), int(new_items)),
        )
        conn.commit()


def start_collection_run():
    with _lock, _connect() as conn:
        cur = conn.execute("INSERT INTO collection_runs(started_at) VALUES(?)", (_now(),))
        conn.commit()
        return int(cur.lastrowid)


def finish_collection_run(run_id, fetched, new_items, errors, skipped, details):
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE collection_runs SET completed_at=?,fetched=?,new_items=?,errors=?,skipped=?,details_json=? WHERE id=?",
            (_now(), int(fetched), int(new_items), int(errors), int(skipped), json.dumps(details, ensure_ascii=False), int(run_id)),
        )
        conn.commit()


def pending_document_items(limit=100):
    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT e.* FROM external_items e
            LEFT JOIN external_documents d ON d.external_item_id=e.id
            WHERE d.external_item_id IS NULL
            ORDER BY e.id DESC LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def save_document(external_item_id, final_url, body_text, status="ok", http_status=None, error=None):
    body_text = (body_text or "").strip()
    digest = hashlib.sha256(body_text.encode("utf-8", errors="ignore")).hexdigest() if body_text else None
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO external_documents(external_item_id,fetched_at,final_url,content_hash,body_text,fetch_status,http_status,error)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(external_item_id) DO UPDATE SET
                fetched_at=excluded.fetched_at,final_url=excluded.final_url,content_hash=excluded.content_hash,
                body_text=excluded.body_text,fetch_status=excluded.fetch_status,http_status=excluded.http_status,error=excluded.error
            """,
            (int(external_item_id), _now(), final_url, digest, body_text[:50000], status, http_status, error),
        )
        conn.commit()


def document_for_item(external_item_id):
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM external_documents WHERE external_item_id=?", (int(external_item_id),)).fetchone()
    return dict(row) if row else None


def queue_metrics():
    with _lock, _connect() as conn:
        ext = conn.execute("SELECT COUNT(*) c FROM external_items").fetchone()["c"]
        docs = conn.execute("SELECT COUNT(*) c FROM external_documents WHERE fetch_status='ok'").fetchone()["c"]
        claims = conn.execute("SELECT COUNT(*) c FROM intelligence_claims").fetchone()["c"]
        evidence = conn.execute("SELECT COUNT(*) c FROM claim_evidence").fetchone()["c"]
        unprocessed = conn.execute(
            "SELECT COUNT(*) c FROM external_items e LEFT JOIN claim_evidence ce ON ce.external_item_id=e.id WHERE ce.id IS NULL"
        ).fetchone()["c"]
        failed_sources = conn.execute("SELECT COUNT(*) c FROM source_health WHERE consecutive_failures>0").fetchone()["c"]
        run = conn.execute("SELECT * FROM collection_runs ORDER BY id DESC LIMIT 1").fetchone()
    return {
        "external_items": int(ext),
        "documents": int(docs),
        "claims": int(claims),
        "evidence": int(evidence),
        "factcheck_backlog": int(unprocessed),
        "failing_sources": int(failed_sources),
        "last_collection": dict(run) if run else None,
    }


def source_health(limit=200):
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM source_health ORDER BY consecutive_failures DESC,last_attempt DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


def record_verification_audit(claim_id, status, confidence, rationale, evidence):
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO verification_audit(claim_id,created_at,status,confidence,rationale,evidence_json) VALUES(?,?,?,?,?,?)",
            (int(claim_id), _now(), status, float(confidence), rationale, json.dumps(evidence, ensure_ascii=False)),
        )
        conn.commit()
