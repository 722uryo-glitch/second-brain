import asyncio
import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DB_PATH
from .executive import run as executive_run
from .orchestrator import is_research_task, is_current_task

_lock = threading.RLock()

JOB_WORKER_ENABLED = os.getenv("JOB_WORKER_ENABLED", "true").lower() == "true"
JOB_POLL_SECONDS = float(os.getenv("JOB_POLL_SECONDS", "0.8"))
JOB_LEASE_SECONDS = int(os.getenv("JOB_LEASE_SECONDS", "90"))
JOB_HEARTBEAT_SECONDS = int(os.getenv("JOB_HEARTBEAT_SECONDS", "20"))
JOB_MAX_RUNTIME_SECONDS = int(os.getenv("JOB_MAX_RUNTIME_SECONDS", "900"))

_LONG_MARKERS = (
    "記事を書", "記事作成", "ブログを書", "レポート", "論文", "台本", "原稿", "企画書", "提案書",
    "調べて", "調査", "比較して", "検証して", "探して", "分析して", "徹底的", "完全版",
    "article", "blog post", "research", "report", "compare", "investigate", "deep research",
)


def _connect():
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat()


def _future(seconds):
    return (datetime.now(timezone.utc) + timedelta(seconds=max(1, int(seconds)))).isoformat()


def init_jobs():
    with _lock, _connect() as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=5000;

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                request TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                current_step TEXT,
                worker_id TEXT,
                lease_expires_at TEXT,
                heartbeat_at TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_runtime_seconds INTEGER NOT NULL DEFAULT 900,
                agent_run_id INTEGER,
                result_text TEXT,
                error TEXT,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(status, lease_expires_at);

            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT,
                data_json TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, id);

            CREATE TABLE IF NOT EXISTS job_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT,
                content TEXT,
                metadata_json TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_job_artifacts_job ON job_artifacts(job_id, id);
            """
        )
        # A process may have died while owning a job. Requeue expired/stale work.
        now = _now()
        conn.execute(
            """UPDATE jobs
               SET status='queued', worker_id=NULL, lease_expires_at=NULL, current_step='recovered_after_restart', updated_at=?
               WHERE status='running' AND (lease_expires_at IS NULL OR lease_expires_at < ?)""",
            (now, now),
        )
        conn.commit()


def should_enqueue(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False
    if is_research_task(t) or is_current_task(t):
        return True
    return len(t) > 220 or any(m.lower() in t for m in _LONG_MARKERS)


def add_event(job_id: str, event_type: str, message: str = "", data=None):
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO job_events(job_id,created_at,event_type,message,data_json) VALUES(?,?,?,?,?)",
            (job_id, _now(), event_type, str(message or "")[:1000], json.dumps(data or {}, ensure_ascii=False)),
        )
        conn.commit()


def create_job(request: str, max_runtime_seconds=None, metadata=None):
    job_id = uuid.uuid4().hex
    now = _now()
    runtime = int(max_runtime_seconds or JOB_MAX_RUNTIME_SECONDS)
    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO jobs(id,created_at,updated_at,request,status,current_step,max_runtime_seconds,metadata_json)
               VALUES(?,?,?,?,?,?,?,?)""",
            (job_id, now, now, request, "queued", "accepted", runtime, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        conn.commit()
    add_event(job_id, "accepted", "依頼を保存しました。バックグラウンドで処理します。")
    return get_job(job_id)


def _decode(row):
    if not row:
        return None
    d = dict(row)
    try:
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
    except Exception:
        d["metadata"] = {}
    d["cancel_requested"] = bool(d.get("cancel_requested"))
    return d


def get_job(job_id: str):
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _decode(row)


def list_jobs(limit=30):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
    return [_decode(r) for r in rows]


def job_events(job_id: str, after_id=0, limit=200):
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM job_events WHERE job_id=? AND id>? ORDER BY id ASC LIMIT ?",
            (job_id, int(after_id), int(limit)),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d["data"] = json.loads(d.pop("data_json") or "{}")
        except Exception:
            d["data"] = {}
        out.append(d)
    return out


def artifacts(job_id: str):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM job_artifacts WHERE job_id=? ORDER BY id ASC", (job_id,)).fetchall()
    return [dict(r) for r in rows]


def request_cancel(job_id: str):
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET cancel_requested=1,updated_at=? WHERE id=? AND status IN ('queued','running')",
            (_now(), job_id),
        )
        conn.commit()
    if cur.rowcount:
        add_event(job_id, "cancel_requested", "取消要求を受け付けました。")
    return bool(cur.rowcount)


def _claim_job(worker_id: str):
    now = _now()
    with _lock, _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        # Recover abandoned leases before selecting new work.
        conn.execute(
            """UPDATE jobs SET status='queued',worker_id=NULL,lease_expires_at=NULL,current_step='recovered_after_lease_expiry',updated_at=?
               WHERE status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?""",
            (now, now),
        )
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' AND cancel_requested=0 ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            conn.commit()
            return None
        conn.execute(
            """UPDATE jobs SET status='running',started_at=COALESCE(started_at,?),updated_at=?,current_step='starting',
               worker_id=?,lease_expires_at=?,heartbeat_at=?,attempts=attempts+1 WHERE id=?""",
            (now, now, worker_id, _future(JOB_LEASE_SECONDS), now, row["id"]),
        )
        conn.commit()
    add_event(row["id"], "started", "Workerが処理を開始しました。", {"worker_id": worker_id})
    return get_job(row["id"])


def _heartbeat(job_id: str, worker_id: str, step=None):
    with _lock, _connect() as conn:
        conn.execute(
            """UPDATE jobs SET heartbeat_at=?,updated_at=?,lease_expires_at=?,current_step=COALESCE(?,current_step)
               WHERE id=? AND worker_id=? AND status='running'""",
            (_now(), _now(), _future(JOB_LEASE_SECONDS), step, job_id, worker_id),
        )
        conn.commit()


def _finish(job_id: str, worker_id: str, status: str, result_text="", error="", agent_run_id=None):
    now = _now()
    with _lock, _connect() as conn:
        conn.execute(
            """UPDATE jobs SET status=?,finished_at=?,updated_at=?,current_step=?,result_text=?,error=?,agent_run_id=?,
               worker_id=NULL,lease_expires_at=NULL,heartbeat_at=?
               WHERE id=? AND worker_id=?""",
            (status, now, now, status, result_text, str(error or "")[:3000], agent_run_id, now, job_id, worker_id),
        )
        if result_text:
            conn.execute(
                "INSERT INTO job_artifacts(job_id,created_at,kind,title,content,metadata_json) VALUES(?,?,?,?,?,?)",
                (job_id, now, "final_response", "Final response", result_text, json.dumps({"agent_run_id": agent_run_id}, ensure_ascii=False)),
            )
        conn.commit()
    add_event(job_id, status, "処理が完了しました。" if status == "succeeded" else str(error or status)[:1000])


async def _heartbeat_loop(job_id: str, worker_id: str):
    while True:
        await asyncio.sleep(max(3, JOB_HEARTBEAT_SECONDS))
        _heartbeat(job_id, worker_id)


async def _execute(job: dict, worker_id: str):
    job_id = job["id"]
    if get_job(job_id).get("cancel_requested"):
        _finish(job_id, worker_id, "cancelled", error="cancelled before start")
        return

    _heartbeat(job_id, worker_id, "executive")
    add_event(job_id, "progress", "目的整理・記憶検索・必要情報の調査を開始しました。", {"step": "executive"})
    hb = asyncio.create_task(_heartbeat_loop(job_id, worker_id))
    try:
        timeout = max(30, int(job.get("max_runtime_seconds") or JOB_MAX_RUNTIME_SECONDS))
        result = await asyncio.wait_for(executive_run(job["request"]), timeout=timeout)
        current = get_job(job_id)
        if current and current.get("cancel_requested"):
            _finish(job_id, worker_id, "cancelled", error="cancel requested")
            return
        if result.mode == "timeout":
            _finish(job_id, worker_id, "partial", result_text=result.response, error="executive timeout", agent_run_id=result.run_id)
        else:
            _finish(job_id, worker_id, "succeeded", result_text=result.response, agent_run_id=result.run_id)
    except asyncio.TimeoutError:
        _finish(job_id, worker_id, "partial", error="job runtime budget exhausted")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _finish(job_id, worker_id, "failed", error=str(e))
    finally:
        hb.cancel()


async def worker_loop():
    worker_id = f"local-{uuid.uuid4().hex[:10]}"
    print(f"[JOBS] worker started id={worker_id}")
    while True:
        try:
            job = _claim_job(worker_id)
            if job:
                await _execute(job, worker_id)
            else:
                await asyncio.sleep(max(0.2, JOB_POLL_SECONDS))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[JOBS] worker loop error: {e}")
            await asyncio.sleep(1.5)


def status():
    with _lock, _connect() as conn:
        counts = {r["status"]: int(r["c"]) for r in conn.execute("SELECT status,COUNT(*) c FROM jobs GROUP BY status").fetchall()}
        oldest = conn.execute("SELECT created_at FROM jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1").fetchone()
    return {
        "enabled": JOB_WORKER_ENABLED,
        "counts": counts,
        "poll_seconds": JOB_POLL_SECONDS,
        "lease_seconds": JOB_LEASE_SECONDS,
        "default_max_runtime_seconds": JOB_MAX_RUNTIME_SECONDS,
        "oldest_queued_at": oldest["created_at"] if oldest else None,
    }
