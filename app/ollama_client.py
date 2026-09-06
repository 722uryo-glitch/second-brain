import httpx

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    UNOROUTER_ENABLED,
    UNOROUTER_BASE_URL,
    UNOROUTER_API_KEY,
    UNOROUTER_PRIVATE_CHAT,
    UNOROUTER_TIMEOUT_SECONDS,
    UNOROUTER_CHAT_MODELS,
    UNOROUTER_VERIFY_MODELS,
)

_ROUTER_STATS = {
    "uno_success": 0,
    "uno_failures": 0,
    "ollama_calls": 0,
    "last_provider": "none",
    "last_model": None,
    "last_error": None,
}


def _url(path: str) -> str:
    return f"{OLLAMA_BASE_URL.rstrip('/')}{path}"


def _uno_url(path: str) -> str:
    return f"{UNOROUTER_BASE_URL.rstrip('/')}{path}"


def _looks_like_public_intelligence(messages) -> bool:
    text = "\n".join(str(m.get("content", "")) for m in messages).lower()
    markers = (
        "multilingual evidence normalizer",
        "fact-check",
        "fact check",
        "claim_key",
        "canonical_claim",
        "evidence items",
        "intelligence collector",
    )
    return any(marker in text for marker in markers)


async def health():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(_url("/api/tags"))
        r.raise_for_status()
        return r.json()


async def _ollama_chat(messages, model=None, temperature=0.4, num_predict=384):
    payload = {
        "model": model or OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(_url("/api/chat"), json=payload)
        r.raise_for_status()
        data = r.json()
        _ROUTER_STATS["ollama_calls"] += 1
        _ROUTER_STATS["last_provider"] = "ollama"
        _ROUTER_STATS["last_model"] = payload["model"]
        return data["message"]["content"]


async def _uno_chat(messages, models, temperature=0.4, num_predict=384):
    if not UNOROUTER_API_KEY:
        raise RuntimeError("UNOROUTER_API_KEY is not configured")

    headers = {
        "Authorization": f"Bearer {UNOROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    errors = []
    async with httpx.AsyncClient(timeout=UNOROUTER_TIMEOUT_SECONDS) as client:
        for model in models:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": temperature,
                "max_tokens": num_predict,
            }
            try:
                r = await client.post(_uno_url("/chat/completions"), headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                if not content:
                    raise RuntimeError("empty response")
                _ROUTER_STATS["uno_success"] += 1
                _ROUTER_STATS["last_provider"] = "unorouter"
                _ROUTER_STATS["last_model"] = model
                _ROUTER_STATS["last_error"] = None
                return content
            except Exception as exc:
                errors.append(f"{model}: {str(exc)[:120]}")
                _ROUTER_STATS["uno_failures"] += 1

    message = " | ".join(errors[-3:]) or "all UnoRouter models failed"
    _ROUTER_STATS["last_error"] = message
    raise RuntimeError(message)


async def chat(messages, model=None, temperature=0.4, num_predict=384, route="auto"):
    """Route reasoning work between UnoRouter free models and local Ollama.

    Defaults:
    - public intelligence/fact-check normalization -> UnoRouter when configured
    - personal chat/reflection -> local Ollama unless UNOROUTER_PRIVATE_CHAT=true
    - any cloud failure -> local Ollama fallback
    - explicit route='local' always stays local
    """
    public_intelligence = _looks_like_public_intelligence(messages)
    cloud_allowed = (
        UNOROUTER_ENABLED
        and bool(UNOROUTER_API_KEY)
        and route != "local"
        and (public_intelligence or UNOROUTER_PRIVATE_CHAT or route == "cloud")
    )

    if cloud_allowed:
        models = UNOROUTER_VERIFY_MODELS if public_intelligence else UNOROUTER_CHAT_MODELS
        try:
            return await _uno_chat(messages, models, temperature, num_predict)
        except Exception as exc:
            print(f"[AI-ROUTER] UnoRouter failed; fallback=ollama error={str(exc)[:220]}")

    return await _ollama_chat(messages, model, temperature, num_predict)


def router_status():
    return {
        "unorouter_enabled": UNOROUTER_ENABLED,
        "unorouter_configured": bool(UNOROUTER_API_KEY),
        "private_chat_cloud_enabled": UNOROUTER_PRIVATE_CHAT,
        "chat_models": list(UNOROUTER_CHAT_MODELS),
        "verify_models": list(UNOROUTER_VERIFY_MODELS),
        "stats": dict(_ROUTER_STATS),
        "policy": "public intelligence uses UnoRouter; personal chat stays local unless explicitly enabled",
    }


async def embed(text: str, model=None):
    payload = {
        "model": model or OLLAMA_EMBED_MODEL,
        "input": text,
        "keep_alive": "30m",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(_url("/api/embed"), json=payload)
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Ollama returned no embedding")
        return embeddings[0]
