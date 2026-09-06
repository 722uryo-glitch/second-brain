import asyncio
import re

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
    return t in {"こんにちは", "こんばんは", "おはよう", "おはようございます", "やあ", "どうも", "hello", "hi", "hey", "こんばんは!", "こんにちは!"}


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


def _valid_ref_count(text: str, refs) -> int:
    valid = {r.get("ref") for r in refs}
    found = set(re.findall(r"\[(S\d+)\]", text or ""))
    return len(found & valid)


async def _research_brief(user_text: str, context: str, refs, enough: bool):
    """Create an evidence-constrained analyst brief before any final writing.

    This deliberately uses the verify/reasoning lane rather than the fast chat lane.
    The final writer is not allowed to invent a niche that the analyst could not support.
    """
    prompt = f"""USER REQUEST:\n{user_text}\n\nSECOND BRAIN EVIDENCE:\n{context}\n\nCreate a compact research brief in Japanese.

Hard rules:
- Every material factual claim about demand, competition/supply, market trend, or user need MUST cite one or more source ids like [S1].
- Do not use general world knowledge to fill evidence gaps.
- Do not say a niche has low competition unless the evidence actually supports low competition/supply.
- If the evidence only supports demand but not low supply, write that explicitly.
- Reject internally contradictory candidates.
- For an affiliate-niche task, evaluate at least 3 candidate angles if evidence permits, then pick one only if BOTH demand and a plausible supply/competition gap are evidenced.
- If no candidate meets that bar, output exactly: NO_EVIDENCE_BACKED_NICHE
- Do not write the final article yet.
"""
    brief = await chat(
        [
            {"role": "system", "content": "You are the evidence-auditing analyst inside a persistent Second Brain. Be skeptical and citation-bound."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.05,
        num_predict=900,
        route="cloud",
    )
    if brief.strip() == "NO_EVIDENCE_BACKED_NICHE":
        return brief
    # If the model produced an uncited 'research' brief, treat it as invalid rather than laundering hallucination.
    if refs and _valid_ref_count(brief, refs) < 2:
        return "NO_EVIDENCE_BACKED_NICHE"
    if not enough:
        return "NO_EVIDENCE_BACKED_NICHE"
    return brief


async def _store_conversation_later(user_text: str, response: str):
    try:
        await store_memory("conversation", "user", user_text, 0.55)
        await store_memory("conversation", "assistant", response, 0.20)
    except Exception as e:
        print(f"[MEMORY] background save failed: {e}")


async def answer(user_text: str):
    history = _short_history(8)
    long_form = _is_long_form_task(user_text)

    # 1) Research/current-affairs lane: always pass through the actual Second Brain.
    if is_research_task(user_text) or is_current_task(user_text):
        context, refs, enough = await gather_research_context(user_text, on_demand=True)
        brief = await _research_brief(user_text, context, refs, enough)

        if brief.strip() == "NO_EVIDENCE_BACKED_NICHE" and is_research_task(user_text):
            response = (
                "第2の脳で調査しましたが、現時点の取得データだけでは『需要がある』と『供給・競合が少ない』を同時に裏付けられる候補を確認できませんでした。\n\n"
                "ここでテーマをでっち上げて記事を書くのはやめます。今の第2の脳はニュース・SNS・GitHub・論文などの外部情報は大量に持っていますが、"
                "検索ボリュームやSERP競合数、広告単価、アフィリエイト案件数まで直接測れていないため、『需要×供給ギャップ』判定にはまだデータが足りません。\n\n"
                "次に必要なのは、市場調査用のデータ源を第2の脳へ追加してから記事生成へ進むことです。"
            )
            asyncio.create_task(_store_conversation_later(user_text, response))
            return response, []

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the execution/writing module of a persistent Second Brain. "
                    "The ANALYST BRIEF was produced by an evidence-auditing stage. Treat it as the only allowed basis for researched market claims. "
                    "Do not invent demand, low competition, supply gaps, trends, prices, popularity, or market size. "
                    "If the brief says a point is unproven, preserve that uncertainty. "
                    "When asked for an article, write the complete useful article, but do not turn a hypothesis into a fact. "
                    "Keep source ids [S1], [S2] attached to factual claims. Answer in the user's language."
                ),
            },
            {"role": "system", "content": "ANALYST BRIEF:\n" + brief},
            *history,
            {"role": "user", "content": user_text},
        ]
        response = await chat(
            messages,
            temperature=0.15,
            num_predict=1800 if long_form else 700,
            route="fast_cloud",
        )

        used = [r for r in refs[:20] if f"[{r['ref']}]" in response]
        if used:
            response += "\n\n参照した情報源:\n" + "\n".join(
                f"[{r['ref']}] {r['source']} — {r['title']}\n{r['url']}" for r in used[:8]
            )
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
        response = await chat(messages, temperature=0.25, num_predict=1800 if long_form else 520, route="fast_cloud")
        asyncio.create_task(_store_conversation_later(user_text, response))
        return response, []

    # 3) Personal-memory lane: semantic long-term memory stays local.
    memories = [] if _is_plain_greeting(user_text) else await recall(user_text, top_k=6)
    memory_text = "\n".join(f"- [{m['kind']}/{m['source']}] {m['content']}" for m in memories) or "（関連記憶なし）"
    messages = [
        {"role": "system", "content": SYSTEM},
        *history,
        {"role": "system", "content": "以下は第2の脳の長期記憶検索結果です。現在の発言に直接関係するものだけ使ってください。関係が薄いものは完全に無視してください。\n" + memory_text},
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

最近の記憶:\n{feed}
"""
    result = await chat([
        {"role": "system", "content": "あなたは第2の脳の内省モジュールです。"},
        {"role": "user", "content": prompt},
    ], temperature=0.3, num_predict=160, route="local")
    if result.strip() != "NO_REFLECTION":
        await store_memory("reflection", "dmn", result.strip(), 0.60)
    return result.strip()
