import json
import re
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

            CREATE TABLE IF NOT EXISTS agent_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                finished_at TEXT,
                user_request TEXT NOT NULL,
                goal TEXT,
                mode TEXT NOT NULL DEFAULT 'unknown',
                status TEXT NOT NULL DEFAULT 'running',
                plan_json TEXT,
                critique_json TEXT,
                final_response TEXT,
                error TEXT,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);

            CREATE TABLE IF NOT EXISTS agent_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                step_no INTEGER NOT NULL,
                step_type TEXT NOT NULL,
                label TEXT,
                status TEXT NOT NULL DEFAULT 'ok',
                input_json TEXT,
                output_json TEXT,
                duration_ms INTEGER,
                error TEXT,
                FOREIGN KEY(run_id) REFERENCES agent_runs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_agent_steps_run ON agent_steps(run_id, step_no);
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


def recent_conversation(limit=10):
    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT id,created_at,source,content
            FROM memories
            WHERE kind='conversation' AND source IN ('user','assistant')
            ORDER BY id DESC LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


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


def _clean_search_terms(terms, max_terms=10):
    out = []
    seen = set()
    for term in terms or []:
        t = str(term or "").strip().lower()
        if len(t) < 2 or t in seen:
            continue
        seen.add(t)
        out.append(t[:80])
        if len(out) >= max_terms:
            break
    return out


def search_external_items(terms, limit=30):
    terms = _clean_search_terms(terms)
    if not terms:
        return []
    clauses = []
    params = []
    for term in terms:
        clauses.append("(lower(title) LIKE ? OR lower(COALESCE(summary,'')) LIKE ? OR lower(source) LIKE ?)")
        pat = f"%{term}%"
        params.extend([pat, pat, pat])
    sql = f"""
        SELECT * FROM external_items
        WHERE {' OR '.join(clauses)}
        ORDER BY id DESC
        LIMIT ?
    """
    params.append(int(limit))
    with _lock, _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_claims(terms, limit=20):
    terms = _clean_search_terms(terms)
    if not terms:
        return []
    clauses = []
    params = []
    for term in terms:
        clauses.append("(lower(claim_text) LIKE ? OR lower(claim_key) LIKE ?)")
        pat = f"%{term}%"
        params.extend([pat, pat])
    sql = f"""
        SELECT * FROM intelligence_claims
        WHERE {' OR '.join(clauses)}
        ORDER BY confidence DESC, updated_at DESC
        LIMIT ?
    """
    params.append(int(limit))
    with _lock, _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def claim_evidence_sources(claim_id, limit=8):
    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT ce.stance,ce.credibility,ce.source_domain,ce.source_type,
                   e.source,e.title,e.url,e.published_at,e.collected_at,e.metadata_json
            FROM claim_evidence ce
            JOIN external_items e ON e.id=ce.external_item_id
            WHERE ce.claim_id=?
            ORDER BY ce.credibility DESC, e.id DESC
            LIMIT ?
            """,
            (int(claim_id), int(limit)),
        ).fetchall()
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


def _normalize_title_lineage(title: str):
    text = re.sub(r"\s+", " ", str(title or "").strip().lower())
    # Search/RSS titles often append the publisher after a dash. Remove a short suffix.
    parts = re.split(r"\s[-|–—]\s", text)
    if len(parts) > 1 and len(parts[-1]) <= 45:
        text = " ".join(parts[:-1])
    text = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:180]


def _lineage_key(row):
    try:
        md = json.loads(row["metadata_json"] or "{}")
    except Exception:
        md = {}
    for key in ("original_source", "wire_source", "canonical_source", "origin_domain"):
        value = str(md.get(key) or "").strip().lower()
        if value:
            return f"origin:{value}"

    source_type = str(row["source_type"] or "")
    domain = str(row["source_domain"] or "").lower()
    title_key = _normalize_title_lineage(row["title"])

    # Primary documents are independently authoritative by origin domain/document.
    if source_type == "primary":
        return f"primary:{domain}:{title_key[:100]}"

    # Identical/near-identical syndication titles across domains count once.
    if title_key:
        return f"story:{title_key}"
    return f"domain:{domain or row['external_item_id']}"


def recompute_claim(claim_id):
    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT ce.external_item_id,ce.source_domain,ce.source_type,ce.stance,ce.credibility,
                   e.title,e.metadata_json
            FROM claim_evidence ce
            JOIN external_items e ON e.id=ce.external_item_id
            WHERE ce.claim_id=?
            """,
            (claim_id,),
        ).fetchall()
        if not rows:
            return

        supports = [r for r in rows if r["stance"] == "supports"]
        contradicts = [r for r in rows if r["stance"] == "contradicts"]
        support_lineages = {_lineage_key(r) for r in supports}
        contradiction_lineages = {_lineage_key(r) for r in contradicts}
        support_domains = {r["source_domain"] for r in supports if r["source_domain"]}

        avg_cred = sum(float(r["credibility"]) for r in supports) / max(1, len(supports))
        independent = len(support_lineages)
        contradiction_count = len(contradiction_lineages)

        if supports and contradicts:
            status = "disputed"
        elif independent >= 3 and avg_cred >= 0.60:
            status = "corroborated"
        elif independent >= 2 and avg_cred >= 0.45:
            status = "partially_corroborated"
        else:
            status = "unverified"

        confidence = (
            0.16 * independent
            + 0.35 * avg_cred
            + 0.05 * min(3, len(support_domains))
            - 0.12 * contradiction_count
        )
        confidence = min(0.97, max(0.02, confidence))
        conn.execute(
            "UPDATE intelligence_claims SET status=?,confidence=?,evidence_count=?,independent_sources=?,contradictions=?,updated_at=? WHERE id=?",
            (
                status, confidence, len(rows), independent, contradiction_count,
                datetime.now(timezone.utc).isoformat(), claim_id,
            ),
        )
        conn.commit()


def recent_claims(limit=100):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM intelligence_claims ORDER BY updated_at DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def start_agent_run(user_request: str, goal: str = "", mode: str = "unknown", plan=None, metadata=None):
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO agent_runs(created_at,user_request,goal,mode,status,plan_json,metadata_json)
               VALUES(?,?,?,?,?,?,?)""",
            (
                now, user_request, goal, mode, "running",
                json.dumps(plan or {}, ensure_ascii=False),
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def add_agent_step(run_id: int, step_no: int, step_type: str, label: str = "", status: str = "ok",
                   input_data=None, output_data=None, duration_ms=None, error=None):
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO agent_steps(run_id,created_at,step_no,step_type,label,status,input_json,output_json,duration_ms,error)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                int(run_id), now, int(step_no), step_type, label, status,
                json.dumps(input_data or {}, ensure_ascii=False),
                json.dumps(output_data or {}, ensure_ascii=False),
                int(duration_ms) if duration_ms is not None else None,
                str(error)[:1000] if error else None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def finish_agent_run(run_id: int, status: str, final_response: str = "", critique=None, error=None, metadata=None):
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            """UPDATE agent_runs
               SET finished_at=?,status=?,final_response=?,critique_json=?,error=?,metadata_json=?
               WHERE id=?""",
            (
                now,
                status,
                final_response,
                json.dumps(critique or {}, ensure_ascii=False),
                str(error)[:2000] if error else None,
                json.dumps(metadata or {}, ensure_ascii=False),
                int(run_id),
            ),
        )
        conn.commit()


def recent_agent_runs(limit=20, include_steps=False):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        result = [dict(r) for r in rows]
        if include_steps:
            for run in result:
                steps = conn.execute(
                    "SELECT * FROM agent_steps WHERE run_id=? ORDER BY step_no ASC,id ASC", (run["id"],)
                ).fetchall()
                run["steps"] = [dict(s) for s in steps]
    return result
