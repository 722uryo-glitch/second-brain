import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import (
    init_db,
    recent_memories,
    delete_memory,
    recent_external_items,
    recent_claims,
    recent_agent_runs,
)
from .brain import answer_detailed, reflect
from .memory import store_memory
from .ollama_client import health, router_status
from .executive import status as executive_status
from .retrieval import init_search_index, retrieval_status
from .runtime_state import init_runtime_state, runtime_state_status
from .orchestrator import orchestrator_status
from .whisper_service import transcribe_bytes
from .global_intelligence import (
    collect_global_information,
    global_collection_loop,
    factcheck_batch,
    factcheck_loop,
    intelligence_status,
    fetch_document_bodies,
)
from .v1_storage import init_v1_storage, source_health, queue_metrics
from .obsidian_export import export_to_obsidian, obsidian_export_loop
from .config import (
    DMN_ENABLED,
    DMN_INTERVAL_MINUTES,
    EXTERNAL_COLLECTION_ENABLED,
    EXTERNAL_COLLECTION_INTERVAL_MINUTES,
    DOCUMENT_FETCH_ENABLED,
    DOCUMENT_FETCH_INTERVAL_SECONDS,
    FACTCHECK_ENABLED,
    FACTCHECK_INTERVAL_SECONDS,
    OBSIDIAN_ENABLED,
    OBSIDIAN_EXPORT_INTERVAL_MINUTES,
)


async def dmn_loop():
    while True:
        await asyncio.sleep(max(1, DMN_INTERVAL_MINUTES) * 60)
        try:
            await reflect()
        except Exception as e:
            print(f"[DMN] reflection failed: {e}")


async def document_fetch_loop():
    while True:
        try:
            result = await fetch_document_bodies()
            if result.get("attempted"):
                print(
                    f"[DOCUMENTS] attempted={result.get('attempted', 0)} "
                    f"ok={result.get('ok', 0)} failed={result.get('failed', 0)}"
                )
        except Exception as e:
            print(f"[DOCUMENTS] fetch failed: {e}")
        await asyncio.sleep(max(2, DOCUMENT_FETCH_INTERVAL_SECONDS))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_v1_storage()
    init_runtime_state()
    try:
        init_search_index()
    except Exception as e:
        print(f"[RETRIEVAL] FTS initialization failed; LIKE fallback remains available: {e}")

    tasks = []
    if DMN_ENABLED:
        tasks.append(asyncio.create_task(dmn_loop()))
    if EXTERNAL_COLLECTION_ENABLED:
        tasks.append(asyncio.create_task(global_collection_loop(EXTERNAL_COLLECTION_INTERVAL_MINUTES)))
    if DOCUMENT_FETCH_ENABLED:
        tasks.append(asyncio.create_task(document_fetch_loop()))
    if FACTCHECK_ENABLED:
        tasks.append(asyncio.create_task(factcheck_loop(FACTCHECK_INTERVAL_SECONDS)))
    if OBSIDIAN_ENABLED:
        tasks.append(asyncio.create_task(obsidian_export_loop(OBSIDIAN_EXPORT_INTERVAL_MINUTES)))
    yield
    for task in tasks:
        task.cancel()


app = FastAPI(title="Second Brain V2 Executive Intelligence", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatIn(BaseModel):
    message: str


class MemoryIn(BaseModel):
    content: str
    kind: str = "note"
    importance: float = 0.6


@app.get("/", response_class=HTMLResponse)
async def home():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def api_health():
    try:
        tags = await health()
        return {
            "ok": True,
            "version": "V2-executive-research",
            "ollama": True,
            "models": [m.get("name") for m in tags.get("models", [])],
            "ai_router": router_status(),
            "executive": executive_status(),
            "retrieval": retrieval_status(),
            "runtime_state": runtime_state_status(),
            "research": orchestrator_status(),
            "external_collection": EXTERNAL_COLLECTION_ENABLED,
            "external_collection_interval_minutes": EXTERNAL_COLLECTION_INTERVAL_MINUTES,
            "document_fetch": DOCUMENT_FETCH_ENABLED,
            "document_fetch_interval_seconds": DOCUMENT_FETCH_INTERVAL_SECONDS,
            "factcheck": FACTCHECK_ENABLED,
            "factcheck_interval_seconds": FACTCHECK_INTERVAL_SECONDS,
            "obsidian": OBSIDIAN_ENABLED,
            "obsidian_export_interval_minutes": OBSIDIAN_EXPORT_INTERVAL_MINUTES,
            "intelligence": intelligence_status(),
        }
    except Exception as e:
        return {"ok": False, "ollama": False, "error": str(e)}


@app.get("/api/ai-router/status")
async def api_ai_router_status():
    return router_status()


@app.get("/api/executive/status")
async def api_executive_status():
    return executive_status()


@app.get("/api/executive/runs")
async def api_executive_runs(limit: int = 20, steps: bool = False):
    return recent_agent_runs(min(max(limit, 1), 100), include_steps=steps)


@app.get("/api/research/status")
async def api_research_status():
    return {
        "orchestrator": orchestrator_status(),
        "retrieval": retrieval_status(),
        "runtime_state": runtime_state_status(),
    }


@app.post("/api/chat")
async def api_chat(data: ChatIn):
    try:
        result = await answer_detailed(data.message)
        return {
            "response": result.response,
            "recalled": [
                {"id": m["id"], "kind": m["kind"], "content": m["content"], "score": round(m["score"], 3)}
                for m in result.memories[:6]
            ],
            "agent_run_id": result.run_id,
            "mode": result.mode,
            "plan": result.plan,
            "critique": result.critique,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memories")
async def api_memories(limit: int = 50):
    return recent_memories(min(max(limit, 1), 500))


@app.post("/api/memories")
async def api_add_memory(data: MemoryIn):
    mid = await store_memory(data.kind, "manual", data.content, data.importance)
    return {"id": mid, "ok": True}


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int):
    delete_memory(memory_id)
    return {"ok": True}


@app.post("/api/reflect")
async def api_reflect():
    return {"reflection": await reflect()}


@app.post("/api/collect")
async def api_collect():
    return await collect_global_information()


@app.post("/api/documents/fetch")
async def api_documents_fetch(limit: int = 120):
    return await fetch_document_bodies(limit=min(max(limit, 1), 1000))


@app.post("/api/factcheck")
async def api_factcheck(limit: int = 30):
    return await factcheck_batch(min(max(limit, 1), 80))


@app.post("/api/obsidian/export")
async def api_obsidian_export():
    try:
        return await asyncio.to_thread(export_to_obsidian)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/external/latest")
async def api_external_latest(limit: int = 100):
    return recent_external_items(min(max(limit, 1), 5000))


@app.get("/api/claims/latest")
async def api_claims_latest(limit: int = 100):
    return recent_claims(min(max(limit, 1), 2000))


@app.get("/api/intelligence/status")
async def api_intelligence_status():
    return intelligence_status()


@app.get("/api/intelligence/queues")
async def api_intelligence_queues():
    return queue_metrics()


@app.get("/api/intelligence/sources")
async def api_intelligence_sources(limit: int = 200):
    return source_health(min(max(limit, 1), 1000))


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    data = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    try:
        text = await asyncio.to_thread(transcribe_bytes, data, suffix)
        if text:
            await store_memory("transcript", "whisper", text, 0.65, {"filename": file.filename})
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
