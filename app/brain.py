import asyncio

from .memory import recall, store_memory
from .ollama_client import chat
from .db import recent_memories, recent_conversation
from .orchestrator import gather_research_context, is_research_task, is_current_task

SYSTEM = """あなたはユーザー専用の『第2の脳』です。
目的は、過去の文脈・長期記憶・蓄積した外部知識・現在の依頼を統合して、実際に役立つ答えと成果物を返すことです。

行動原則:
- ユーザーが『任せる』『なんでもいい』『調べてやって』と言ったら、合理的な仮定を置いて自分で進める。不要な確認質問をしない。
- 前の発言で既に分かっていることを、もう一度質問しない。
- 調査が必要な依頼では、想像で埋めず、第2の脳が持つ外部知識・ファクトチェック・オンデマンド調査を先に使う。
- 根拠が足りないときは『足りない』と明示し、需要・供給・競合・市場性を作り話で断定しない。
- 質問に答えるだけでなく、依頼なら実行可能な形まで具体化する。
- 迷った場合は最も妥当な選択肢を選び、その選択を短く明示して進める。
- ありきたりな一般論だけで終わらせない。

記憶の信頼順位:
1. manual
2. user 由来の preference / user_fact / decision / task
3. user の conversation / whisper transcript
4. reflection / dmn
5. assistant の過去発言

記憶の原則:
- 関係する記憶だけ使う。
- assistant の過去発言は事実として扱わない。
- user/manual と assistant が矛盾する場合は user/manual を優先する。
- 記憶にないことを『覚えている』とは言わない。
"""

_PERSONAL_MARKERS = (
    "私の", "僕の", "俺の", "自分の", "覚えて", "前に話", "前回", "好み", "予定", "タスク",
    "住所", "電話", "メール", "家族", "友達", "学校", "職場", "仕事の", "名前", "誕生日",
)
_LONG_FORM_MARKERS = (
    "記事を書", "記事作成", "ブログを書", "レポートを書", "論文を書", "台本を書", "原稿を書",
    "企画書", "提案書", "完全版", "詳しくまとめ", "詳細にまとめ", "長文", "徹底的に",
    "article", "blog post", "write an article", "report", "full draft", "long-form",
)


def _is_plain_greeting(text: str) -> bool:
    t = text.strip().lower().replace("！", "!").replace("？", "?")
    return t in {
        "こんにちは", "こんばんは", "おはよう", "おはようございます",
        "やあ", "どうも", "hello", "hi", "hey", "こんばんは!", "こんにちは!"
    }


def _looks_personal(text: str) -> bool:
    t = text.lower()
    return any(x.lower() in t for x in _PERSONAL_MARKERS)


def _is_long_form_task(text: str) -> bool:
    t = text.lower()
    return any(x.lower() in t for x in _LONG_FORM_MARKERS)


def _short_history(limit=8):
    rows = recent_conversation(limit)
    messages = []
    for row in rows:
        role = "user" if row.get("source") == "user" else "assistant"
        content = str(row.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content[:1800]})
    return messages


async def _store_conversation_later(user_text: str, response: str):
    try:
        await store_memory("conversation", "user", user_text, 0.55)
        await store_memory("conversation", "assistant", response, 0.20)
    except Exception as e:
        print(f"[MEMORY] background save failed: {e}")


async def answer(user_text: str):
    history = _short_history(8)
    long_form = _is_long_form_task(user_text)

    # 1) Research/current-affairs lane: this is the actual Second Brain path.
    # It consults the persistent intelligence archive, fact-checked claims and
    # one on-demand public research request before the model writes anything.
    if is_research_task(user_text) or is_current_task(user_text):
        context, refs, enough = await gather_research_context(user_text, on_demand=True)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the research/execution module of a persistent Second Brain. "
                    "The evidence pack below comes from the Second Brain's stored external intelligence, "
                    "fact-check database, and on-demand public research. Use it before reasoning from general knowledge. "
                    "Do not pretend that market demand, low competition, supply gaps, trends, or current events were researched "
                    "unless the evidence pack supports them. If evidence is insufficient, say exactly what is unproven. "
                    "When the user asks for an article after research, first choose the strongest evidence-backed angle and then "
                    "write the complete deliverable. Cite [S1], [S2] inline for factual claims when useful. "
                    "Do not ask the user to choose a topic when they delegated that choice. Answer in the user's language."
                ),
            },
            {"role": "system", "content": context},
            *history,
            {"role": "user", "content": user_text},
        ]
        response = await chat(
            messages,
            temperature=0.18,
            num_predict=1800 if long_form else 700,
            route="fast_cloud",
        )

        if refs:
            used = [r for r in refs[:10] if f"[{r['ref']}]" in response]
            if used:
                response += "\n\n参照した情報源:\n" + "\n".join(
                    f"[{r['ref']}] {r['source']} — {r['title']}\n{r['url']}" for r in used[:6]
                )
        if not enough:
            response += "\n\n※ 第2の脳内の根拠が十分でない部分は、断定ではなく仮説として扱っています。"

        asyncio.create_task(_store_conversation_later(user_text, response))
        return response, []

    # 2) Generic work/questions: recent dialogue + strong model.
    if not _looks_personal(user_text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the main reasoning/working module of a persistent Second Brain. "
                    "Continue naturally using recent dialogue. Do not repeat questions already answered. "
                    "When choices are delegated, choose and proceed. Prefer concrete execution over generic advice. "
                    "For long-form deliverables, finish the deliverable in this response. Answer in the user's language."
                ),
            },
            *history,
            {"role": "user", "content": user_text},
        ]
        response = await chat(
            messages,
            temperature=0.25,
            num_predict=1800 if long_form else 520,
            route="fast_cloud",
        )
        asyncio.create_task(_store_conversation_later(user_text, response))
        return response, []

    # 3) Personal-memory lane: semantic long-term memory stays local.
    memories = [] if _is_plain_greeting(user_text) else await recall(user_text, top_k=6)
    memory_text = "\n".join(
        f"- [{m['kind']}/{m['source']}] {m['content']}" for m in memories
    ) or "（関連記憶なし）"
    messages = [
        {"role": "system", "content": SYSTEM},
        *history,
        {
            "role": "system",
            "content": (
                "以下は第2の脳の長期記憶検索結果です。現在の発言に直接関係するものだけ使ってください。"
                "関係が薄いものは完全に無視してください。\n" + memory_text
            ),
        },
        {"role": "user", "content": user_text},
    ]
    response = await chat(messages, num_predict=360, route="local")
    asyncio.create_task(_store_conversation_later(user_text, response))
    return response, memories


async def reflect():
    recent = list(reversed(recent_memories(25)))
    if not recent:
        return "記憶がまだないため、内省をスキップしました。"
    feed = "\n".join(f"[{m['kind']}/{m['source']}] {m['content']}" for m in recent)
    prompt = f"""最近の記憶を読み、内部用の短い内省を1つ作ってください。
manual / user の記憶を優先し、assistant の過去発言を事実扱いしないでください。
繰り返す傾向、未完了事項、矛盾、再利用可能な教訓、将来役立つ関連付けのどれもなければ NO_REFLECTION だけ返してください。
ある場合は事実と推測を分け、200文字以内で書いてください。

最近の記憶:
{feed}
"""
    result = await chat([
        {"role": "system", "content": "あなたは第2の脳の内省モジュールです。"},
        {"role": "user", "content": prompt},
    ], temperature=0.3, num_predict=160, route="local")
    if result.strip() != "NO_REFLECTION":
        await store_memory("reflection", "dmn", result.strip(), 0.60)
    return result.strip()
