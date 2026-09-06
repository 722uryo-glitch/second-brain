import math
import re
from datetime import datetime, timezone

from .db import _connect, _lock, search_external_items, search_claims

_FTS_READY = False
_FTS_MODE = "uninitialized"


def _clean_terms(terms, max_terms=12):
    out = []
    seen = set()
    for raw in terms or []:
        term = re.sub(r"\s+", " ", str(raw or "").strip())
        key = term.lower()
        if len(term) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(term[:80])
        if len(out) >= max_terms:
            break
    return out


def init_search_index():
    """Create local FTS indexes when SQLite supports them.

    Trigram tokenization is preferred because it works much better for Japanese
    and substring-like entity lookup. We fall back to unicode61 and finally to
    the existing LIKE retrieval without making startup fail.
    """
    global _FTS_READY, _FTS_MODE
    if _FTS_READY:
        return True
    for tokenizer in ("trigram", "unicode61"):
        try:
            with _lock, _connect() as conn:
                conn.executescript(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS external_items_fts USING fts5(
                        title, summary, source,
                        content='external_items', content_rowid='id',
                        tokenize='{tokenizer}'
                    );
                    CREATE VIRTUAL TABLE IF NOT EXISTS intelligence_claims_fts USING fts5(
                        claim_text, claim_key,
                        content='intelligence_claims', content_rowid='id',
                        tokenize='{tokenizer}'
                    );

                    CREATE TRIGGER IF NOT EXISTS external_items_ai AFTER INSERT ON external_items BEGIN
                      INSERT INTO external_items_fts(rowid,title,summary,source)
                      VALUES(new.id,new.title,COALESCE(new.summary,''),new.source);
                    END;
                    CREATE TRIGGER IF NOT EXISTS external_items_ad AFTER DELETE ON external_items BEGIN
                      INSERT INTO external_items_fts(external_items_fts,rowid,title,summary,source)
                      VALUES('delete',old.id,old.title,COALESCE(old.summary,''),old.source);
                    END;
                    CREATE TRIGGER IF NOT EXISTS external_items_au AFTER UPDATE ON external_items BEGIN
                      INSERT INTO external_items_fts(external_items_fts,rowid,title,summary,source)
                      VALUES('delete',old.id,old.title,COALESCE(old.summary,''),old.source);
                      INSERT INTO external_items_fts(rowid,title,summary,source)
                      VALUES(new.id,new.title,COALESCE(new.summary,''),new.source);
                    END;

                    CREATE TRIGGER IF NOT EXISTS intelligence_claims_ai AFTER INSERT ON intelligence_claims BEGIN
                      INSERT INTO intelligence_claims_fts(rowid,claim_text,claim_key)
                      VALUES(new.id,new.claim_text,new.claim_key);
                    END;
                    CREATE TRIGGER IF NOT EXISTS intelligence_claims_ad AFTER DELETE ON intelligence_claims BEGIN
                      INSERT INTO intelligence_claims_fts(intelligence_claims_fts,rowid,claim_text,claim_key)
                      VALUES('delete',old.id,old.claim_text,old.claim_key);
                    END;
                    CREATE TRIGGER IF NOT EXISTS intelligence_claims_au AFTER UPDATE ON intelligence_claims BEGIN
                      INSERT INTO intelligence_claims_fts(intelligence_claims_fts,rowid,claim_text,claim_key)
                      VALUES('delete',old.id,old.claim_text,old.claim_key);
                      INSERT INTO intelligence_claims_fts(rowid,claim_text,claim_key)
                      VALUES(new.id,new.claim_text,new.claim_key);
                    END;
                    """
                )
                conn.execute("INSERT INTO external_items_fts(external_items_fts) VALUES('rebuild')")
                conn.execute("INSERT INTO intelligence_claims_fts(intelligence_claims_fts) VALUES('rebuild')")
                conn.commit()
            _FTS_READY = True
            _FTS_MODE = tokenizer
            return True
        except Exception as e:
            _FTS_MODE = f"fallback:{type(e).__name__}"
    return False


def _fts_query(terms):
    parts = []
    for term in terms:
        cleaned = term.replace('"', ' ').strip()
        if len(cleaned) >= 3:
            parts.append(f'"{cleaned}"')
    return " OR ".join(parts[:10])


def _parse_time(value):
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _freshness_score(row):
    dt = _parse_time(row.get("published_at") or row.get("collected_at") or row.get("updated_at"))
    if not dt:
        return 0.0
    age_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
    # Smooth decay: ~1 at now, 0.5 around one week, still non-zero for older evergreen evidence.
    return 1.0 / (1.0 + age_hours / 168.0)


def _lexical_score(row, terms, fields):
    score = 0.0
    matched = 0
    for term in terms:
        t = term.lower()
        best = 0.0
        for field, weight in fields:
            value = str(row.get(field) or "").lower()
            if t and t in value:
                best = max(best, weight)
        if best:
            matched += 1
            score += best
    if terms:
        score += 2.0 * (matched / len(terms))
    return score


def search_external_ranked(terms, limit=30):
    terms = _clean_terms(terms)
    if not terms:
        return []
    candidates = {}

    # FTS retrieves older but highly relevant evidence that newest-first LIKE can miss.
    if init_search_index():
        query = _fts_query(terms)
        if query:
            try:
                with _lock, _connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT e.*, bm25(external_items_fts, 3.0, 1.2, 0.5) AS fts_rank
                        FROM external_items_fts
                        JOIN external_items e ON e.id=external_items_fts.rowid
                        WHERE external_items_fts MATCH ?
                        ORDER BY fts_rank ASC
                        LIMIT ?
                        """,
                        (query, max(int(limit) * 5, 60)),
                    ).fetchall()
                for row in rows:
                    d = dict(row)
                    candidates[d["id"]] = d
            except Exception:
                pass

    # Keep the old retrieval as a resilient fallback and for 2-char/Japanese terms.
    for row in search_external_items(terms, limit=max(int(limit) * 5, 80)):
        candidates[row["id"]] = row

    scored = []
    for row in candidates.values():
        score = _lexical_score(row, terms, (("title", 3.0), ("summary", 1.4), ("source", 0.4)))
        score += 0.9 * _freshness_score(row)
        if "fts_rank" in row:
            try:
                score += min(2.0, 1.0 / (0.1 + abs(float(row["fts_rank"]))))
            except Exception:
                pass
        row["retrieval_score"] = round(score, 4)
        scored.append(row)
    scored.sort(key=lambda r: (r.get("retrieval_score", 0.0), r.get("id", 0)), reverse=True)
    return scored[: int(limit)]


def search_claims_ranked(terms, limit=20):
    terms = _clean_terms(terms)
    if not terms:
        return []
    candidates = {}
    if init_search_index():
        query = _fts_query(terms)
        if query:
            try:
                with _lock, _connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT c.*, bm25(intelligence_claims_fts, 3.0, 1.0) AS fts_rank
                        FROM intelligence_claims_fts
                        JOIN intelligence_claims c ON c.id=intelligence_claims_fts.rowid
                        WHERE intelligence_claims_fts MATCH ?
                        ORDER BY fts_rank ASC
                        LIMIT ?
                        """,
                        (query, max(int(limit) * 4, 40)),
                    ).fetchall()
                for row in rows:
                    d = dict(row)
                    candidates[d["id"]] = d
            except Exception:
                pass
    for row in search_claims(terms, limit=max(int(limit) * 4, 40)):
        candidates[row["id"]] = row

    scored = []
    for row in candidates.values():
        score = _lexical_score(row, terms, (("claim_text", 3.0), ("claim_key", 1.2)))
        score += 1.4 * float(row.get("confidence") or 0.0)
        score += 0.5 * _freshness_score(row)
        if row.get("status") == "corroborated":
            score += 0.8
        elif row.get("status") == "disputed":
            score += 0.25
        row["retrieval_score"] = round(score, 4)
        scored.append(row)
    scored.sort(key=lambda r: (r.get("retrieval_score", 0.0), r.get("confidence", 0.0)), reverse=True)
    return scored[: int(limit)]


def retrieval_status():
    return {"fts_ready": bool(_FTS_READY), "fts_mode": _FTS_MODE}
