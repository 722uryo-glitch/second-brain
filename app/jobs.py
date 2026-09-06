import asyncio
import json
import inspect
from contextlib import closing
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .job_context import current as job_context, JobStopped, BudgetExceeded, YieldForUser
from .config import DB_PATH
from .executive import run as executive_run
from .orchestrator import is_research_task, is_current_task
from .progress import set_reporter, reset_reporter

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
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat()


def _future(seconds):
    return (datetime.now(timezone.utc) + timedelta(seconds=max(1, int(seconds)))).isoformat()


def init_jobs():
    with _lock, closing(_connect()) as conn:
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
        conn.execute("BEGIN IMMEDIATE")
        columns = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
        for name, declaration in {
            "deadline_at": "TEXT", "calls_used": "INTEGER NOT NULL DEFAULT 0",
            "retries_used": "INTEGER NOT NULL DEFAULT 0", "output_tokens_reserved": "INTEGER NOT NULL DEFAULT 0",
            "max_calls": "INTEGER NOT NULL DEFAULT 120", "max_retries": "INTEGER NOT NULL DEFAULT 12",
            "max_output_tokens": "INTEGER NOT NULL DEFAULT 40000",
        }.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")
        conn.execute("""CREATE TABLE IF NOT EXISTS job_steps (
            job_id TEXT NOT NULL, step_key TEXT NOT NULL, input_json TEXT NOT NULL,
            status TEXT NOT NULL, output_json TEXT, attempts INTEGER NOT NULL DEFAULT 0,
            started_at TEXT, finished_at TEXT, error TEXT, worker_id TEXT,
            PRIMARY KEY(job_id, step_key))""")
        # Initializing a second API process must not revoke a live worker.
        _recover(conn)
        conn.commit()


def _event(conn, job_id, event_type, message, data=None):
    conn.execute("INSERT INTO job_events(job_id,created_at,event_type,message,data_json) VALUES(?,?,?,?,?)",
                 (job_id, _now(), event_type, str(message)[:1000], json.dumps(data or {}, ensure_ascii=False)))


def _recover(conn):
    now = _now()
    expired = conn.execute("""SELECT id,cancel_requested FROM jobs WHERE status='running'
        AND (lease_expires_at IS NULL OR lease_expires_at <= ?)""", (now,)).fetchall()
    for row in expired:
        state = "cancelled" if row["cancel_requested"] else "queued"
        conn.execute("""UPDATE jobs SET status=?,worker_id=NULL,lease_expires_at=NULL,
            current_step=?,updated_at=?,finished_at=? WHERE id=?""",
            (state, "recovered_after_lease_expiry", now, now if state == "cancelled" else None, row["id"]))
        conn.execute("UPDATE job_steps SET status='interrupted' WHERE job_id=? AND status='running'", (row["id"],))
        _event(conn, row["id"], "recovered", "実行権の期限切れを検出しました。保存済み工程から再開します。")


def _owned(conn, job_id, worker_id, allow_cancel=False):
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row or row["status"] != "running" or row["worker_id"] != worker_id or not row["lease_expires_at"] or row["lease_expires_at"] <= _now():
        raise JobStopped("worker lease lost")
    if row["cancel_requested"] and not allow_cancel:
        raise JobStopped("cancel requested")
    return row



def should_enqueue(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False
    if is_research_task(t) or is_current_task(t):
        return True
    return len(t) > 220 or any(m.lower() in t for m in _LONG_MARKERS)


def add_event(job_id: str, event_type: str, message: str = "", data=None):
    with _lock, closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO job_events(job_id,created_at,event_type,message,data_json) VALUES(?,?,?,?,?)",
            (job_id, _now(), event_type, str(message or "")[:1000], json.dumps(data or {}, ensure_ascii=False)),
        )
        conn.commit()


def create_job(request: str, max_runtime_seconds=None, metadata=None):
    job_id = uuid.uuid4().hex
    now = _now()
    runtime = max(1, int(max_runtime_seconds if max_runtime_seconds is not None else JOB_MAX_RUNTIME_SECONDS))
    with _lock, closing(_connect()) as conn:
        conn.execute(
            """INSERT INTO jobs(id,created_at,updated_at,request,status,current_step,max_runtime_seconds,metadata_json)
               VALUES(?,?,?,?,?,?,?,?)""",
            (job_id, now, now, request, "queued", "accepted", runtime, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        _event(conn, job_id, "accepted", "依頼を保存しました。バックグラウンドで処理します。")
        conn.commit()
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
    with _lock, closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _decode(row)


def list_jobs(limit=30):
    with _lock, closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
    return [_decode(r) for r in rows]


def job_events(job_id: str, after_id=0, limit=200, latest=False):
    with _lock, closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM job_events WHERE job_id=? AND id>? ORDER BY id " + ("DESC" if latest else "ASC") + " LIMIT ?",
            (job_id, int(after_id), int(limit)),
        ).fetchall()
    if latest:
        rows = list(reversed(rows))
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
    with _lock, closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM job_artifacts WHERE job_id=? ORDER BY id ASC", (job_id,)).fetchall()
    return [dict(r) for r in rows]


def request_cancel(job_id: str):
    with _lock, closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row or row["status"] not in {"queued", "running"}:
            return False
        # Terminal cancellation also fences any in-flight completion.
        conn.execute("""UPDATE jobs SET cancel_requested=1,status='cancelled',finished_at=?,
            updated_at=?,current_step='cancelled',worker_id=NULL,lease_expires_at=NULL WHERE id=?""",
            (_now(), _now(), job_id))
        conn.execute("UPDATE job_steps SET status='interrupted' WHERE job_id=? AND status='running'", (job_id,))
        _event(conn, job_id, "cancelled", "取消済み。実行中の子処理を終了します。途中成果は保持します。")
        conn.commit()
    return True


def _claim_job(worker_id: str):
    worker_id = worker_id + ":" + uuid.uuid4().hex
    now = _now()
    with _lock, closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _recover(conn)
        if conn.execute("SELECT 1 FROM jobs WHERE status='running' LIMIT 1").fetchone():
            conn.commit()
            return None
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' AND cancel_requested=0 ORDER BY CASE WHEN json_extract(metadata_json,'$.autonomous')=1 THEN 1 ELSE 0 END, created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            conn.commit()
            return None
        conn.execute(
            """UPDATE jobs SET status='running',started_at=COALESCE(started_at,?),updated_at=?,current_step='starting',
               worker_id=?,lease_expires_at=?,heartbeat_at=?,attempts=attempts+1,deadline_at=COALESCE(deadline_at,?) WHERE id=?""",
            (now, now, worker_id, _future(JOB_LEASE_SECONDS), now, _future(row["max_runtime_seconds"]), row["id"]),
        )
        _event(conn, row["id"], "started", "Workerが処理を開始しました。", {"worker_id": worker_id})
        claimed = conn.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
        conn.commit()
    return _decode(claimed)


def _heartbeat(job_id: str, worker_id: str, step=None):
    with _lock, closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _owned(conn, job_id, worker_id)
        conn.execute("""UPDATE jobs SET heartbeat_at=?,updated_at=?,lease_expires_at=?,
            current_step=COALESCE(?,current_step) WHERE id=?""",
            (_now(), _now(), _future(JOB_LEASE_SECONDS), step, job_id))
        conn.commit()


def _finish(job_id, worker_id, status, result_text="", error="", agent_run_id=None):
    with _lock, closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _owned(conn, job_id, worker_id)
        except JobStopped:
            return False
        conn.execute("""UPDATE jobs SET status=?,finished_at=?,updated_at=?,current_step=?,
            result_text=?,error=?,agent_run_id=?,worker_id=NULL,lease_expires_at=NULL WHERE id=?""",
            (status, _now(), _now(), status, result_text, str(error)[:3000], agent_run_id, job_id))
        if result_text:
            conn.execute("INSERT INTO job_artifacts(job_id,created_at,kind,title,content,metadata_json) VALUES(?,?,?,?,?,?)",
                (job_id, _now(), "final_response", "最終応答", result_text, "{}"))
        conn.execute("UPDATE job_steps SET status='interrupted' WHERE job_id=? AND status='running'", (job_id,))
        _event(conn, job_id, status, "完了しました。" if status == "succeeded" else (error or status))
        conn.commit()
    return True


def steps(job_id):
    with _lock, closing(_connect()) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM job_steps WHERE job_id=? ORDER BY started_at,step_key", (job_id,))]


def resume_job(job_id, extend_budget=False):
    with _lock, closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row or row["status"] not in {"failed", "partial", "cancelled"}:
            return False
        complete = conn.execute("SELECT 1 FROM job_steps WHERE job_id=? AND step_key='v1:final' AND status='succeeded'", (job_id,)).fetchone()
        if complete and not row["error"]:
            return False
        if extend_budget:
            # Explicit user action grants one more original time/call/token allowance.
            conn.execute("""UPDATE jobs SET deadline_at=?,max_calls=max_calls+120,
                max_retries=max_retries+12,max_output_tokens=max_output_tokens+40000 WHERE id=?""",
                (_future(row["max_runtime_seconds"]), job_id))
        elif row["deadline_at"] and row["deadline_at"] <= _now():
            return False
        conn.execute("""UPDATE jobs SET status='queued',cancel_requested=0,worker_id=NULL,
            lease_expires_at=NULL,finished_at=NULL,error=NULL,result_text=NULL,current_step='resume_queued',updated_at=? WHERE id=?""", (_now(), job_id))
        _event(conn, job_id, "resume_queued", "保存済み工程から再開します。", {"budget_extended": extend_budget})
        conn.commit()
    return True


class Execution:
    def __init__(self, job, worker_id):
        self.job_id, self.worker_id = job["id"], worker_id
        self.local_only = bool(job.get("metadata", {}).get("autonomous"))

    def check(self, conn):
        row = _owned(conn, self.job_id, self.worker_id)
        if self.local_only and conn.execute("SELECT 1 FROM jobs WHERE status='queued' AND COALESCE(json_extract(metadata_json,'$.autonomous'),0)<>1 LIMIT 1").fetchone():
            raise YieldForUser("foreground request takes priority")
        if row["deadline_at"] and row["deadline_at"] <= _now():
            raise BudgetExceeded("job runtime budget exhausted")
        return row

    def reserve(self, retry=False, output_tokens=0):
        with _lock, closing(_connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self.check(conn)
            if (row["calls_used"] + 1 > row["max_calls"] or
                row["retries_used"] + int(retry) > row["max_retries"] or
                row["output_tokens_reserved"] + output_tokens > row["max_output_tokens"]):
                raise BudgetExceeded("job call/retry/output-token budget exhausted")
            conn.execute("""UPDATE jobs SET calls_used=calls_used+1,retries_used=retries_used+?,
                output_tokens_reserved=output_tokens_reserved+? WHERE id=?""", (int(retry), output_tokens, self.job_id))
            conn.commit()

    async def step(self, key, inputs, operation, kind):
        encoded = json.dumps(inputs, ensure_ascii=False, sort_keys=True)
        # Versioned keys prevent accidental reuse under a changed workflow contract.
        key = "v1:" + key
        with _lock, closing(_connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.check(conn)
            old = conn.execute("SELECT * FROM job_steps WHERE job_id=? AND step_key=?", (self.job_id, key)).fetchone()
            if old and old["input_json"] != encoded:
                raise RuntimeError("checkpoint input changed: " + key)
            if old and old["status"] == "succeeded":
                _event(conn, self.job_id, "step_reused", "保存済み工程を再利用: " + key, {"step": key})
                conn.commit()
                return json.loads(old["output_json"])
            if old:
                row = self.check(conn)
                if row["retries_used"] >= row["max_retries"]:
                    raise BudgetExceeded("job retry budget exhausted on step resume")
                conn.execute("UPDATE jobs SET retries_used=retries_used+1 WHERE id=?", (self.job_id,))
            conn.execute("""INSERT INTO job_steps(job_id,step_key,input_json,status,attempts,started_at,worker_id)
                VALUES(?,?,?,'running',1,?,?) ON CONFLICT(job_id,step_key) DO UPDATE SET
                status='running',attempts=attempts+1,worker_id=excluded.worker_id,error=NULL""",
                (self.job_id, key, encoded, _now(), self.worker_id))
            conn.execute("UPDATE jobs SET current_step=?,updated_at=? WHERE id=?", (key, _now(), self.job_id))
            _event(conn, self.job_id, "step_started", "実行中: " + key, {"step": key})
            conn.commit()
        try:
            value = operation()
            if inspect.isawaitable(value):
                value = await value
            output = json.dumps(value, ensure_ascii=False)
            with _lock, closing(_connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self.check(conn)
                conn.execute("""UPDATE job_steps SET status='succeeded',output_json=?,finished_at=?,error=NULL
                    WHERE job_id=? AND step_key=? AND worker_id=?""", (output, _now(), self.job_id, key, self.worker_id))
                conn.execute("INSERT INTO job_artifacts(job_id,created_at,kind,title,content,metadata_json) VALUES(?,?,?,?,?,?)",
                    (self.job_id, _now(), kind, key, value if isinstance(value, str) else output,
                     json.dumps({"step": key, "verified": False})))
                _event(conn, self.job_id, "step_completed", "保存済み: " + key, {"step": key})
                conn.commit()
            return json.loads(output)
        except BaseException as exc:
            with _lock, closing(_connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    _owned(conn, self.job_id, self.worker_id)
                except JobStopped:
                    pass
                else:
                    conn.execute("UPDATE job_steps SET status=?,error=? WHERE job_id=? AND step_key=?",
                        ("interrupted" if isinstance(exc, asyncio.CancelledError) else "failed", str(exc)[:1000], self.job_id, key))
                    conn.commit()
            raise


async def _execute(job, worker_id):
    job_id = job["id"]
    context = Execution(job, worker_id)
    token = job_context.set(context)

    def reporter(step, message, data):
        with _lock, closing(_connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            context.check(conn)
            _event(conn, job_id, "progress", message or step, {"step": step, **(data or {})})
            conn.commit()

    progress_token = set_reporter(reporter)
    next_heartbeat = 0.0
    if job.get("metadata", {}).get("autonomous"):
        from .autonomy import run_job
        operation = run_job(job)
    else:
        operation = executive_run(job["request"])
    task = asyncio.create_task(operation)
    try:
        while not task.done():
            with closing(_connect()) as conn:
                context.check(conn)
            now_mono = asyncio.get_running_loop().time()
            if now_mono >= next_heartbeat:
                _heartbeat(job_id, worker_id)
                next_heartbeat = now_mono + max(0.2, min(JOB_HEARTBEAT_SECONDS, JOB_LEASE_SECONDS / 3))
            await asyncio.wait({task}, timeout=min(0.25, max(0.05, JOB_HEARTBEAT_SECONDS)))
        result = task.result()
        outcome = "partial" if result.critique and not result.critique.get("pass", False) else "succeeded"
        _finish(job_id, worker_id, outcome, result.response, agent_run_id=result.run_id)
    except YieldForUser:
        with _lock, closing(_connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned(conn, job_id, worker_id)
            conn.execute("UPDATE jobs SET status='queued',worker_id=NULL,lease_expires_at=NULL,current_step='yielded_to_user' WHERE id=?",(job_id,))
            conn.execute("UPDATE job_steps SET status='interrupted' WHERE job_id=? AND status='running'",(job_id,))
            _event(conn,job_id,'yielded','ユーザーの依頼を優先し、自主探索を一時中断しました。')
            conn.commit()
    except BudgetExceeded as exc:
        saved = [a for a in artifacts(job_id) if a["kind"] == "answer"]
        _finish(job_id, worker_id, "partial", result_text=saved[-1]["content"] if saved else "", error=str(exc))
    except JobStopped:
        pass  # Cancel endpoint or a newer lease owns the terminal state.
    except asyncio.CancelledError:
        # Graceful shutdown releases only this lease; hard crashes use lease expiry.
        with _lock, closing(_connect()) as conn:
            conn.execute("UPDATE jobs SET lease_expires_at=? WHERE id=? AND worker_id=? AND status='running'",
                         (_now(), job_id, worker_id))
            conn.commit()
        raise
    except Exception as exc:
        _finish(job_id, worker_id, "partial" if artifacts(job_id) else "failed", error=str(exc))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        reset_reporter(progress_token)
        job_context.reset(token)


async def worker_loop():
    worker_id = f"local-{uuid.uuid4().hex[:10]}"
    print(f"[JOBS] worker started id={worker_id}")
    while True:
        try:
            job = _claim_job(worker_id)
            if job:
                await _execute(job, job["worker_id"])
            else:
                await asyncio.sleep(max(0.2, JOB_POLL_SECONDS))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[JOBS] worker loop error: {e}")
            await asyncio.sleep(1.5)


def status():
    with _lock, closing(_connect()) as conn:
        counts = {r["status"]: int(r["c"]) for r in conn.execute("SELECT status,COUNT(*) c FROM jobs GROUP BY status").fetchall()}
        oldest = conn.execute("SELECT created_at FROM jobs WHERE status='queued' ORDER BY CASE WHEN json_extract(metadata_json,'$.autonomous')=1 THEN 1 ELSE 0 END, created_at ASC LIMIT 1").fetchone()
    return {
        "enabled": JOB_WORKER_ENABLED,
        "counts": counts,
        "poll_seconds": JOB_POLL_SECONDS,
        "lease_seconds": JOB_LEASE_SECONDS,
        "default_max_runtime_seconds": JOB_MAX_RUNTIME_SECONDS,
        "oldest_queued_at": oldest["created_at"] if oldest else None,
    }
