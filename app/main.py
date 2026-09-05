import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import init_db, recent_memories, delete_memory
from .brain import answer, reflect
from .memory import store_memory
from .ollama_client import health
from .whisper_service import transcribe_bytes
from .config import DMN_ENABLED, DMN_INTERVAL_MINUTES


async def dmn_loop():
    while True:
        await asyncio.sleep(max(1, DMN_INTERVAL_MINUTES) * 60)
        try:
            await reflect()
        except Exception as e:
            print(f"[DMN] reflection failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    task = None
    if DMN_ENABLED:
        task = asyncio.create_task(dmn_loop())
    yield
    if task:
        task.cancel()


app = FastAPI(title="Second Brain v1", lifespan=lifespan)
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
        return {"ok": True, "ollama": True, "models": [m.get("name") for m in tags.get("models", [])]}
    except Exception as e:
        return {"ok": False, "ollama": False, "error": str(e)}


@app.post("/api/chat")
async def api_chat(data: ChatIn):
    try:
        response, memories = await answer(data.message)
        return {
            "response": response,
            "recalled": [
                {"id": m["id"], "kind": m["kind"], "content": m["content"], "score": round(m["score"], 3)}
                for m in memories[:6]
            ],
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
