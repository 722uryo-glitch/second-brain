import json
from datetime import datetime, timezone

from .db import _connect, _lock


def init_runtime_state():
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_state (
                state_key TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                value_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_state(key: str, default=None):
    init_runtime_state()
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT value_json FROM runtime_state WHERE state_key=?",
            (str(key),),
        ).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value_json"])
    except Exception:
        return default


def set_state(key: str, value):
    init_runtime_state()
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO runtime_state(state_key,updated_at,value_json)
            VALUES(?,?,?)
            ON CONFLICT(state_key) DO UPDATE SET
              updated_at=excluded.updated_at,
              value_json=excluded.value_json
            """,
            (str(key), now, json.dumps(value, ensure_ascii=False)),
        )
        conn.commit()


def runtime_state_status():
    init_runtime_state()
    with _lock, _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM runtime_state").fetchone()
    return {"entries": int(row["n"] or 0)}
