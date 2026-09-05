import math
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


async def store_memory(kind, source, content, importance=0.5, metadata=None):
    vector = await embed(content)
    return add_memory(kind, source, content, importance, vector, metadata)


async def recall(query, top_k=8):
    """Recall memories using semantic similarity plus trust/importance weighting.

    The main goal is to prevent an old assistant guess from outranking an
    explicit user/manual memory.
    """
    qv = await embed(query)
    scored = []

    kind_bonus = {
        "preference": 0.30,
        "user_fact": 0.30,
        "decision": 0.28,
        "task": 0.24,
        "note": 0.20,
        "transcript": 0.10,
        "reflection": 0.04,
        "conversation": 0.00,
    }

    source_bonus = {
        "manual": 0.30,
        "user": 0.16,
        "whisper": 0.10,
        "dmn": 0.00,
        "assistant": -0.24,
    }

    for item in all_memories_with_embeddings():
        similarity = cosine(qv, item["embedding"])
        importance = float(item.get("importance", 0.5))
        kind = item.get("kind", "conversation")
        source = item.get("source", "")

        score = (
            similarity * 0.68
            + importance * 0.12
            + kind_bonus.get(kind, 0.05)
            + source_bonus.get(source, 0.0)
        )

        # Explicitly saved user knowledge should be hard to displace by
        # generated assistant text.
        if source == "manual":
            score += 0.10
        if kind in {"preference", "user_fact", "decision"} and source in {"manual", "user"}:
            score += 0.08

        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
