import asyncio
from .memory import recall, store_memory
from .ollama_client import chat
from .db import recent_memories

SYSTEM = """あなたはユーザー専用の『第2の脳』です。
目的は、会話をその場限りで終わらせず、過去の記憶・現在の文脈・未完了事項を統合して支援することです。

記憶の信頼順位:
1. manual で保存された記憶
2. user 由来の preference / user_fact / decision / task
3. user の conversation / whisper transcript
4. reflection / dmn
5. assistant の過去発言

原則:
- 記憶は、現在の発言に直接関係するときだけ使う。
- 関係のない記憶を、会話に無理やり差し込まない。
- あいさつや雑談には、関連する記憶がなければ普通に返す。
- manual または user 由来の明示的な記憶は、関連している場合に最優先で扱う。
- assistant の過去発言は事実ではなく、過去の生成結果にすぎない。
- assistant の過去発言と user/manual の記憶が矛盾する場合、user/manual を優先する。
- 記憶にないことを『覚えている』とは言わない。
- 明示的な好み・事実・決定事項については勝手に別の答えを推測しない。
- 返答は自然で簡潔にする。
"""


def _is_plain_greeting(text: str) -> bool:
    t = text.strip().lower().replace("！", "!").replace("？", "?")
    greetings = {
        "こんにちは", "こんばんは", "おはよう", "おはようございます",
        "やあ", "どうも", "hello", "hi", "hey", "こんばんは!", "こんにちは!"
    }
    return t in greetings


async def _store_conversation_later(user_text: str, response: str):
    """Persist conversation after the response is already ready for the user."""
    try:
        await store_memory("conversation", "user", user_text, 0.55)
        # Assistant output is stored only as low-trust conversation history.
        await store_memory("conversation", "assistant", response, 0.20)
    except Exception as e:
        print(f"[MEMORY] background save failed: {e}")


async def answer(user_text: str):
    # Plain greetings should not trigger long-term memory retrieval. This avoids
    # unrelated facts such as preferences leaking into ordinary small talk.
    memories = [] if _is_plain_greeting(user_text) else await recall(user_text, top_k=6)

    memory_text = "\n".join(
        f"- [{m['kind']}/{m['source']}] {m['content']}"
        for m in memories
    ) or "（関連記憶なし）"

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "system",
            "content": (
                "以下は検索で見つかった長期記憶です。現在の発言に直接関係するものだけ使ってください。"
                "関係が薄いものは完全に無視してください。\n" + memory_text
            ),
        },
        {"role": "user", "content": user_text},
    ]

    response = await chat(messages, num_predict=320)
    asyncio.create_task(_store_conversation_later(user_text, response))
    return response, memories


async def reflect():
    recent = list(reversed(recent_memories(25)))
    if not recent:
        return "記憶がまだないため、内省をスキップしました。"

    feed = "\n".join(
        f"[{m['kind']}/{m['source']}] {m['content']}" for m in recent
    )
    prompt = f"""最近の記憶を読み、内部用の短い内省を1つ作ってください。

信頼ルール:
- manual / user の記憶を優先する。
- assistant の過去発言をユーザーの事実として扱わない。
- assistant と user/manual が矛盾する場合、user/manual を優先する。

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
    ], temperature=0.3, num_predict=180)

    if result.strip() != "NO_REFLECTION":
        await store_memory("reflection", "dmn", result.strip(), 0.60)
    return result.strip()
