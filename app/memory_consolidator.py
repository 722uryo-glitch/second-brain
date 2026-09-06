import json
import re

from .db import recent_memories
from .memory import store_memory
from .ollama_client import chat

_ALLOWED_KINDS = {"preference", "user_fact", "decision", "task", "goal"}


def _extract_array(text: str):
    if not text:
        return []
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(cleaned[start:end + 1])
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _normalize(text: str):
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _already_exists(content: str):
    needle = _normalize(content)
    if not needle:
        return True
    for row in recent_memories(300):
        if row.get("source") not in {"user", "manual"}:
            continue
        existing = _normalize(row.get("content"))
        if existing == needle:
            return True
    return False


async def consolidate_user_turn(user_text: str, run_id=None):
    """Extract durable user-authored memory after a turn without slowing the reply.

    This runs on local Ollama only so personal information is not sent to the
    cloud merely for memory extraction. It never promotes assistant statements
    into user facts.
    """
    text = str(user_text or "").strip()
    if len(text) < 4:
        return {"stored": 0, "items": []}

    prompt = f"""Extract durable memories from this USER message only.
Return one strict JSON array and nothing else.

Each item:
{{"kind":"preference|user_fact|decision|task|goal","content":"...","importance":0.0}}

Rules:
- Store only facts/preferences/decisions/tasks/goals explicitly stated or clearly committed by the user.
- Do NOT infer hidden traits, identity, health, politics, religion, sexuality, or other sensitive attributes.
- Do NOT turn a question, hypothetical, example, or assistant claim into a user fact.
- Rewrite content so it can stand alone later, but preserve the user's meaning.
- Use preference for likes/dislikes; decision for chosen approaches; task for concrete unfinished actions; goal for longer-term desired outcomes.
- importance: 0.55 ordinary, 0.70 important, 0.85 explicitly emphasized/long-term.
- If nothing durable exists, return [].

USER MESSAGE:
{text[:5000]}
"""
    try:
        raw = await chat(
            [
                {"role": "system", "content": "You are a private local memory consolidator. JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            num_predict=420,
            route="local",
        )
        items = _extract_array(raw)
    except Exception as e:
        print(f"[MEMORY] consolidation failed: {e}")
        return {"stored": 0, "items": [], "error": str(e)}

    stored = []
    for item in items[:6]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        content = str(item.get("content") or "").strip()
        if kind not in _ALLOWED_KINDS or len(content) < 3 or _already_exists(content):
            continue
        try:
            importance = float(item.get("importance", 0.6))
        except Exception:
            importance = 0.6
        importance = min(0.95, max(0.45, importance))
        try:
            mid = await store_memory(
                kind,
                "user",
                content,
                importance,
                {"consolidated": True, "agent_run_id": run_id},
            )
            stored.append({"id": mid, "kind": kind, "content": content})
        except Exception as e:
            print(f"[MEMORY] durable store failed: {e}")

    if stored:
        print(f"[MEMORY] consolidated={len(stored)}")
    return {"stored": len(stored), "items": stored}
