import math
from datetime import datetime, timezone

from .db import add_memory, all_memories_with_embeddings
from .ollama_client import embed


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def _recency_bonus(created_at: str):
    if not created_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
        # Fresh explicit memories win close ties without erasing durable older facts.
        return 0.06 / (1.0 + days / 90.0)
    except Exception:
        return 0.0


async def store_memory(kind, source, content, importance=0.5, metadata=None):
    vector = await embed(content)
    return add_memory(kind, source, content, importance, vector, metadata)


async def find_near_duplicate(content: str, kinds=None, sources=None, threshold=0.88, limit=5000):
    """Return the closest semantically equivalent memory above threshold."""
    qv = await embed(content)
    best = None
    kinds = set(kinds or [])
    sources = set(sources or [])
    for item in all_memories_with_embeddings(limit=limit):
        if kinds and item.get("kind") not in kinds:
            continue
        if sources and item.get("source") not in sources:
            continue
        similarity = cosine(qv, item.get("embedding"))
        if similarity < threshold:
            continue
        if best is None or similarity > best["similarity"]:
            best = {**item, "similarity": similarity}
    return best


async def recall(query, top_k=8):
    """Recall relevant durable memory with trust, importance and recency ranking."""
    qv = await embed(query)
    scored = []

    kind_bonus = {
        "preference": 0.18,
        "user_fact": 0.18,
        "decision": 0.16,
        "goal": 0.16,
        "task": 0.14,
        "note": 0.10,
        "transcript": 0.06,
        "reflection": 0.02,
        "conversation": 0.00,
    }

    source_bonus = {
        "manual": 0.18,
        "user": 0.10,
        "whisper": 0.05,
        "dmn": 0.00,
        "assistant": -0.16,
    }

    for item in all_memories_with_embeddings(limit=5000):
        similarity = cosine(qv, item["embedding"])
        kind = item.get("kind", "conversation")
        source = item.get("source", "")
        min_similarity = 0.50 if source == "assistant" else (0.44 if kind == "conversation" else 0.40)
        if similarity < min_similarity:
            continue

        importance = float(item.get("importance", 0.5))
        score = (
            similarity * 0.76
            + importance * 0.08
            + kind_bonus.get(kind, 0.03)
            + source_bonus.get(source, 0.0)
            + _recency_bonus(item.get("created_at"))
        )
        if kind in {"preference", "user_fact", "decision", "goal"} and source in {"manual", "user"}:
            score += 0.06

        item["similarity"] = similarity
        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda x: (x["score"], x.get("id", 0)), reverse=True)
    return scored[:top_k]
