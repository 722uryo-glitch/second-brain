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
    qv = await embed(query)
    scored = []
    for item in all_memories_with_embeddings():
        score = cosine(qv, item["embedding"])
        score = score * 0.85 + float(item.get("importance", 0.5)) * 0.15
        item["score"] = score
        scored.append(item)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
