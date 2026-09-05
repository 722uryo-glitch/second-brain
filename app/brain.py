from .memory import recall, store_memory
from .ollama_client import chat
from .db import recent_memories

SYSTEM = """あなたはユーザー専用の『第2の脳』です。
目的は、会話をその場限りで終わらせず、過去の記憶・現在の文脈・未完了事項を統合して支援することです。

原則:
- 取り出した記憶は事実候補として扱い、矛盾があれば断定しない。
- 重要な決定、好み、未完了タスク、失敗と成功の理由を重視する。
- 記憶にないことを『覚えている』とは言わない。
- 返答は簡潔だが、次の行動が明確になるようにする。
"""


async def answer(user_text: str):
    memories = await recall(user_text, top_k=10)
    memory_text = "\n".join(
        f"- [{m['kind']}] {m['content']}" for m in memories if m.get("score", 0) > 0.15
    ) or "（関連記憶なし）"

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "system",
            "content": "以下は関連する長期記憶です。必要なものだけ使ってください。\n" + memory_text,
        },
        {"role": "user", "content": user_text},
    ]
    response = await chat(messages)
    await store_memory("conversation", "user", user_text, 0.55)
    await store_memory("conversation", "assistant", response, 0.45)
    return response, memories


async def reflect():
    recent = list(reversed(recent_memories(25)))
    if not recent:
        return "記憶がまだないため、内省をスキップしました。"
    feed = "\n".join(f"[{m['kind']}/{m['source']}] {m['content']}" for m in recent)
    prompt = f"""最近の記憶を読み、内部用の短い内省を1つ作ってください。
次のどれかがある場合だけ価値があります:
- 繰り返している傾向
- 未完了の重要事項
- 明確な矛盾
- 成功/失敗から得られる再利用可能な教訓
- 将来役立つ関連付け

なければ『NO_REFLECTION』だけ返してください。
ある場合は、事実と推測を分け、200文字以内で書いてください。

最近の記憶:
{feed}
"""
    result = await chat([
        {"role": "system", "content": "あなたは第2の脳の内省モジュールです。"},
        {"role": "user", "content": prompt},
    ], temperature=0.3)
    if result.strip() != "NO_REFLECTION":
        await store_memory("reflection", "dmn", result.strip(), 0.75)
    return result.strip()
