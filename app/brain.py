import asyncio
import os

from .db import recent_memories
from .executive import ExecutiveResult, run as executive_run
from .memory import store_memory
from .ollama_client import chat

CHAT_HARD_TIMEOUT_SECONDS = int(os.getenv("CHAT_HARD_TIMEOUT_SECONDS", "120"))


async def answer_detailed(user_text: str):
    """Run the full Second Brain with a hard wall-clock deadline."""
    try:
        return await asyncio.wait_for(
            executive_run(user_text),
            timeout=max(30, CHAT_HARD_TIMEOUT_SECONDS),
        )
    except asyncio.TimeoutError:
        print(f"[CHAT] hard timeout after {CHAT_HARD_TIMEOUT_SECONDS}s")
        return ExecutiveResult(
            response=(
                "処理が時間上限に達したため、この実行を中断しました。"
                "外部検索かAIモデルのどこかが応答待ちになった可能性があります。"
                "無限待ちを防ぐ保護が働いた状態です。もう一度送ると、失敗したモデルのクールダウンを避けて再試行します。"
            ),
            memories=[],
            run_id=None,
            mode="timeout",
            plan={},
            critique={"pass": False, "issues": ["hard_timeout"]},
        )


async def answer(user_text: str):
    """Backward-compatible primary chat entry point."""
    result = await answer_detailed(user_text)
    return result.response, result.memories


async def reflect():
    recent = list(reversed(recent_memories(30)))
    if not recent:
        return "記憶がまだないため、内省をスキップしました。"

    feed = "\n".join(f"[{m['kind']}/{m['source']}] {m['content']}" for m in recent)
    prompt = f"""最近の記憶を読み、第2の脳の内部用内省を1つ作ってください。

優先順位:
- manual / user の記憶を最優先する。
- assistant の過去発言は事実ではない。
- 事実と推測を分ける。

価値があるのは次のどれかがある場合だけ:
- 繰り返している傾向
- 未完了の重要事項
- 明確な矛盾
- 成功/失敗から得られる再利用可能な教訓
- 将来役立つ関連付け
- 以前の目標と現在の行動のズレ

なければ NO_REFLECTION だけ返してください。
ある場合は200文字以内で、将来の判断に再利用できる形で書いてください。

最近の記憶:
{feed}
"""
    result = await chat(
        [
            {"role": "system", "content": "あなたは第2の脳の内省モジュールです。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        num_predict=180,
        route="local",
    )

    result = result.strip()
    if result and result != "NO_REFLECTION":
        try:
            await store_memory("reflection", "dmn", result, 0.60)
        except Exception as e:
            print(f"[DMN] reflection save failed: {e}")
    return result
