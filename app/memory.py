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
    """Recall only memories that are actually relevant to the current query.

    Trust/importance bonuses are applied only *after* semantic relevance passes
    a minimum gate. This prevents a high-trust manual memory from appearing in
    unrelated small talk just because it has a large source bonus.
    """
    qv = await embed(query)
    scored = []

    kind_bonus = {
        "preference": 0.18,
        "user_fact": 0.18,
        "decision": 0.16,
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

    for item in all_memories_with_embeddings():
        similarity = cosine(qv, item["embedding"])
        kind = item.get("kind", "conversation")
        source = item.get("source", "")

        # Relevance gate comes FIRST. A trustworthy but unrelated memory must
        # not leak into the prompt. Assistant text needs an even stronger match.
        min_similarity = 0.50 if source == "assistant" else 0.40
        if similarity < min_similarity:
            continue

        importance = float(item.get("importance", 0.5))
        score = (
            similarity * 0.76
            + importance * 0.08
            + kind_bonus.get(kind, 0.03)
            + source_bonus.get(source, 0.0)
        )

        # Explicit user knowledge wins among memories that are already relevant.
        if kind in {"preference", "user_fact", "decision"} and source in {"manual", "user"}:
            score += 0.06

        item["similarity"] = similarity
        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
