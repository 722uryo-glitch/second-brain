import httpx
from .config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL


def _url(path: str) -> str:
    return f"{OLLAMA_BASE_URL.rstrip('/')}{path}"


async def health():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(_url("/api/tags"))
        r.raise_for_status()
        return r.json()


async def chat(messages, model=None, temperature=0.4):
    payload = {
        "model": model or OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(_url("/api/chat"), json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]


async def embed(text: str, model=None):
    payload = {"model": model or OLLAMA_EMBED_MODEL, "input": text}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(_url("/api/embed"), json=payload)
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Ollama returned no embedding")
        return embeddings[0]
