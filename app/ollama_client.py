import asyncio
import os
import time

import httpx

from .job_context import reserve_call, current as job_context

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

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "0.7"))
LLM_MODEL_COOLDOWN_SECONDS = int(os.getenv("LLM_MODEL_COOLDOWN_SECONDS", "90"))
OLLAMA_INTERACTIVE_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_INTERACTIVE_TIMEOUT_SECONDS", "55"))

_ROUTER_STATS = {
    "uno_success": 0,
    "uno_failures": 0,
    "ollama_calls": 0,
    "retries": 0,
    "fallbacks": 0,
    "last_provider": "none",
    "last_model": None,
    "last_route": None,
    "last_latency_ms": None,
    "last_error": None,
    "models": {},
}
_MODEL_COOLDOWN_UNTIL = {}


def _url(path: str) -> str:
    return f"{OLLAMA_BASE_URL.rstrip('/')}{path}"


def _uno_url(path: str) -> str:
    return f"{UNOROUTER_BASE_URL.rstrip('/')}{path}"


def _looks_like_public_intelligence(messages) -> bool:
    text = "\n".join(str(m.get("content", "")) for m in messages).lower()
    markers = (
        "multilingual evidence normalizer", "fact-check", "fact check", "claim_key",
        "canonical_claim", "evidence items", "intelligence collector", "evidence-gap",
        "second brain evidence pack", "fact-checked claims",
    )
    return any(marker in text for marker in markers)


def _model_stats(model):
    return _ROUTER_STATS["models"].setdefault(model, {
        "success": 0, "failures": 0, "last_latency_ms": None, "last_error": None,
    })


def _fast_model_order(models):
    def score(name: str):
        n = name.lower()
        penalty = 0
        if "think" in n or "reason" in n or "search" in n:
            penalty += 10
        if "flash" in n:
            penalty -= 3
        return penalty
    return sorted(list(dict.fromkeys(models)), key=score)


def _reasoning_model_order(models):
    def score(name: str):
        n = name.lower()
        score = 0
        if "think" in n or "reason" in n:
            score -= 4
        if "search" in n:
            score -= 1
        if "flash" in n:
            score += 1
        return score
    return sorted(list(dict.fromkeys(models)), key=score)


def _model_available(model):
    return time.monotonic() >= float(_MODEL_COOLDOWN_UNTIL.get(model, 0.0))


def _mark_model_failure(model, error):
    stats = _model_stats(model)
    stats["failures"] += 1
    stats["last_error"] = str(error)[:220]
    _MODEL_COOLDOWN_UNTIL[model] = time.monotonic() + max(10, LLM_MODEL_COOLDOWN_SECONDS)


def _mark_model_success(model, latency_ms):
    stats = _model_stats(model)
    stats["success"] += 1
    stats["last_latency_ms"] = int(latency_ms)
    stats["last_error"] = None
    _MODEL_COOLDOWN_UNTIL.pop(model, None)


def _is_transient(exc):
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 425, 429, 500, 502, 503, 504}
    return False


async def health():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(_url("/api/tags"))
        r.raise_for_status()
        return r.json()


async def _ollama_chat(messages, model=None, temperature=0.4, num_predict=384, is_fallback=False):
    payload = {
        "model": model or OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    start = time.perf_counter()
    last = None
    # Interactive fallback must never turn into a multi-minute retry chain.
    attempts = 2
    for attempt in range(attempts):
        try:
            reserve_call(retry=attempt > 0 or is_fallback, output_tokens=num_predict)
            async with httpx.AsyncClient(timeout=max(15, OLLAMA_INTERACTIVE_TIMEOUT_SECONDS)) as client:
                r = await client.post(_url("/api/chat"), json=payload)
                r.raise_for_status()
                data = r.json()
            latency = int((time.perf_counter() - start) * 1000)
            _ROUTER_STATS["ollama_calls"] += 1
            _ROUTER_STATS["last_provider"] = "ollama"
            _ROUTER_STATS["last_model"] = payload["model"]
            _ROUTER_STATS["last_latency_ms"] = latency
            _ROUTER_STATS["last_error"] = None
            return data["message"]["content"]
        except Exception as e:
            last = e
            if attempt + 1 < attempts and _is_transient(e):
                _ROUTER_STATS["retries"] += 1
                await asyncio.sleep(0.5)
                continue
            break
    raise last or RuntimeError("Ollama chat failed")


async def _uno_one(client, headers, model, messages, temperature, num_predict, timeout_seconds, max_attempts=1, is_fallback=False):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": num_predict,
    }
    last = None
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        start = time.perf_counter()
        try:
            reserve_call(retry=attempt > 0 or is_fallback, output_tokens=num_predict)
            r = await client.post(_uno_url("/chat/completions"), headers=headers, json=payload, timeout=timeout_seconds)
            r.raise_for_status()
            data = r.json()
            content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            if not content:
                raise RuntimeError("empty response")
            latency = int((time.perf_counter() - start) * 1000)
            _mark_model_success(model, latency)
            return content, latency
        except Exception as e:
            last = e
            if attempt + 1 < attempts and _is_transient(e):
                _ROUTER_STATS["retries"] += 1
                await asyncio.sleep(min(1.0, LLM_RETRY_BASE_SECONDS * (2 ** attempt)))
                continue
            break
    _mark_model_failure(model, last)
    raise last or RuntimeError("UnoRouter model failed")


async def _uno_chat(messages, models, temperature=0.4, num_predict=384, route="cloud"):
    if not UNOROUTER_API_KEY:
        raise RuntimeError("UNOROUTER_API_KEY is not configured")

    headers = {"Authorization": f"Bearer {UNOROUTER_API_KEY}", "Content-Type": "application/json"}
    if route == "fast_cloud":
        model_list = _fast_model_order(models)
        per_model_timeout = min(16, max(7, UNOROUTER_TIMEOUT_SECONDS))
        max_models = 2
        max_attempts = 1
    elif route in {"verify", "reasoning"}:
        model_list = _reasoning_model_order(models)
        # Reasoning used to allow 3 models x retries x 70s, which could make a
        # single browser request appear frozen for several minutes.
        per_model_timeout = min(28, max(12, UNOROUTER_TIMEOUT_SECONDS))
        max_models = 2
        max_attempts = 1
    else:
        model_list = list(dict.fromkeys(models))
        per_model_timeout = min(24, max(10, UNOROUTER_TIMEOUT_SECONDS))
        max_models = 2
        max_attempts = 1

    available = [m for m in model_list if _model_available(m)]
    if not available and model_list:
        available = model_list[:1]

    errors = []
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for model_index, model in enumerate(available[:max_models]):
            try:
                content, latency = await _uno_one(
                    client, headers, model, messages, temperature, num_predict,
                    per_model_timeout, max_attempts=max_attempts, is_fallback=model_index > 0,
                )
                _ROUTER_STATS["uno_success"] += 1
                _ROUTER_STATS["last_provider"] = "unorouter"
                _ROUTER_STATS["last_model"] = model
                _ROUTER_STATS["last_route"] = route
                _ROUTER_STATS["last_latency_ms"] = latency
                _ROUTER_STATS["last_error"] = None
                return content
            except Exception as exc:
                errors.append(f"{model}: {str(exc)[:140]}")
                _ROUTER_STATS["uno_failures"] += 1

    message = " | ".join(errors[-3:]) or "all UnoRouter models unavailable"
    _ROUTER_STATS["last_error"] = message
    raise RuntimeError(message)


async def chat(messages, model=None, temperature=0.4, num_predict=384, route="auto"):
    """Role-aware routing with bounded latency and local fallback."""
    public_intelligence = _looks_like_public_intelligence(messages)
    _ROUTER_STATS["last_route"] = route
    cloud_allowed = (
        UNOROUTER_ENABLED
        and bool(UNOROUTER_API_KEY)
        and route != "local"
        and (public_intelligence or UNOROUTER_PRIVATE_CHAT or route in {"cloud", "fast_cloud", "verify", "reasoning"})
    )

    if cloud_allowed:
        if route == "fast_cloud":
            models = UNOROUTER_CHAT_MODELS
        elif route in {"verify", "reasoning"}:
            models = UNOROUTER_VERIFY_MODELS
        elif public_intelligence:
            models = UNOROUTER_VERIFY_MODELS
        else:
            models = UNOROUTER_CHAT_MODELS
        try:
            return await _uno_chat(messages, models, temperature, num_predict, route=route)
        except Exception as exc:
            _ROUTER_STATS["fallbacks"] += 1
            print(f"[AI-ROUTER] UnoRouter failed; fallback=ollama route={route} error={str(exc)[:220]}")

    return await _ollama_chat(messages, model, temperature, num_predict, is_fallback=cloud_allowed)


def router_status():
    cooldowns = {}
    now = time.monotonic()
    for model, until in _MODEL_COOLDOWN_UNTIL.items():
        remaining = max(0, int(until - now))
        if remaining:
            cooldowns[model] = remaining
    return {
        "unorouter_enabled": UNOROUTER_ENABLED,
        "unorouter_configured": bool(UNOROUTER_API_KEY),
        "private_chat_cloud_enabled": UNOROUTER_PRIVATE_CHAT,
        "chat_models": list(UNOROUTER_CHAT_MODELS),
        "verify_reasoning_models": list(UNOROUTER_VERIFY_MODELS),
        "cooldowns_seconds": cooldowns,
        "stats": dict(_ROUTER_STATS),
        "policy": "interactive routes are latency-bounded; fast -> chat models; reasoning/verify -> verify models; failures fall back locally",
    }


async def embed(text: str, model=None):
    payload = {"model": model or OLLAMA_EMBED_MODEL, "input": text, "keep_alive": "30m"}
    last = None
    attempts = max(1, min(2, LLM_MAX_RETRIES + 1))
    for attempt in range(attempts):
        try:
            reserve_call(retry=attempt > 0)
            async with httpx.AsyncClient(timeout=45) as client:
                r = await client.post(_url("/api/embed"), json=payload)
                r.raise_for_status()
                data = r.json()
            embeddings = data.get("embeddings") or []
            if not embeddings:
                raise RuntimeError("Ollama returned no embedding")
            return embeddings[0]
        except Exception as e:
            last = e
            if attempt + 1 < attempts and _is_transient(e):
                _ROUTER_STATS["retries"] += 1
                await asyncio.sleep(0.5)
                continue
            break
    raise last or RuntimeError("embedding failed")
