import asyncio
import json
import re
from datetime import datetime, timezone

from .memory import recall, store_memory
from .ollama_client import chat
from .db import (
    recent_memories,
    search_external_items,
    search_claims,
    claim_evidence_sources,
)

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

_CURRENT_MARKERS = (
    "現在", "今", "いま", "最新", "今日", "昨日", "今週", "情勢", "ニュース", "速報",
    "どうなって", "どうなった", "最近", "現状", "リアルタイム", "latest", "current",
    "today", "now", "news", "situation", "update",
)
_PUBLIC_TOPIC_MARKERS = (
    "アメリカ", "米国", "iran", "イラン", "中国", "china", "ロシア", "russia", "ウクライナ",
    "israel", "イスラエル", "gaza", "ガザ", "nato", "選挙", "政府", "戦争", "紛争", "市場",
    "株", "ai", "openai", "google", "microsoft", "apple", "github", "cyber", "気候", "地震",
)


def _is_plain_greeting(text: str) -> bool:
    t = text.strip().lower().replace("！", "!").replace("？", "?")
    greetings = {
        "こんにちは", "こんばんは", "おはよう", "おはようございます",
        "やあ", "どうも", "hello", "hi", "hey", "こんばんは!", "こんにちは!"
    }
    return t in greetings


def _is_current_public_question(text: str) -> bool:
    t = text.lower()
    has_current = any(x in t for x in _CURRENT_MARKERS)
    has_public = any(x in t for x in _PUBLIC_TOPIC_MARKERS)
    # "現在の○○情勢" and similar questions should always use the live knowledge store.
    return has_current and (has_public or "情勢" in t or "ニュース" in t)


def _extract_json_object(text: str):
    try:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        pass
    return {}


async def _search_terms_for_question(user_text: str):
    prompt = f"""Convert the user's current-affairs question into compact search terms for a multilingual news database.
Return JSON only: {{"terms":[...]}}.
Rules:
- include important entity names in English and in the user's language when useful
- include common aliases/acronyms (for example United States, US, U.S.)
- do not answer the question
- maximum 10 short terms
Question: {user_text}
"""
    try:
        raw = await chat([
            {"role": "system", "content": "You generate database search keywords only."},
            {"role": "user", "content": prompt},
        ], temperature=0.0, num_predict=180, route="cloud")
        obj = _extract_json_object(raw)
        terms = [str(x).strip() for x in obj.get("terms", []) if str(x).strip()]
        if terms:
            return terms[:10]
    except Exception as e:
        print(f"[LIVE-RAG] keyword extraction failed: {e}")

    # Safe fallback: extract meaningful chunks from the original query.
    chunks = re.findall(r"[A-Za-z][A-Za-z0-9.\-]{1,30}|[一-龥ァ-ヶぁ-ん]{2,20}", user_text)
    return chunks[:10]


async def _public_intelligence_context(user_text: str):
    terms = await _search_terms_for_question(user_text)
    items = search_external_items(terms, limit=24)
    claims = search_claims(terms, limit=12)

    lines = [
        f"CURRENT UTC TIME: {datetime.now(timezone.utc).isoformat()}",
        f"SEARCH TERMS: {', '.join(terms)}",
        "",
        "RECENT COLLECTED SOURCES:",
    ]
    source_refs = []
    for idx, item in enumerate(items, 1):
        ref = f"S{idx}"
        source_refs.append({
            "ref": ref,
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "source": item.get("source") or "",
            "time": item.get("published_at") or item.get("collected_at") or "",
        })
        summary = (item.get("summary") or "").replace("\n", " ")[:500]
        lines.append(
            f"[{ref}] time={item.get('published_at') or item.get('collected_at') or ''} "
            f"source={item.get('source') or ''} title={item.get('title') or ''} "
            f"url={item.get('url') or ''} summary={summary}"
        )

    lines += ["", "FACT-CHECK CLAIMS:"]
    for claim in claims:
        lines.append(
            f"- status={claim.get('status')} confidence={float(claim.get('confidence') or 0):.2f} "
            f"independent_sources={claim.get('independent_sources')} contradictions={claim.get('contradictions')} "
            f"claim={claim.get('claim_text')}"
        )
        for ev in claim_evidence_sources(claim.get("id"), limit=3):
            lines.append(
                f"  evidence: stance={ev.get('stance')} credibility={ev.get('credibility')} "
                f"source={ev.get('source')} time={ev.get('published_at') or ev.get('collected_at')} url={ev.get('url')}"
            )

    return "\n".join(lines), source_refs


async def _store_conversation_later(user_text: str, response: str):
    try:
        await store_memory("conversation", "user", user_text, 0.55)
        await store_memory("conversation", "assistant", response, 0.20)
    except Exception as e:
        print(f"[MEMORY] background save failed: {e}")


async def answer(user_text: str):
    # Current public-affairs questions use the live collected intelligence store.
    # Personal memories are intentionally NOT sent to the cloud in this path.
    if _is_current_public_question(user_text):
        context, refs = await _public_intelligence_context(user_text)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the current-affairs analysis module of a Second Brain. "
                    "Use ONLY the supplied collected intelligence for claims about current events. "
                    "Do not claim you lack real-time access: you have a local continuously collected intelligence store. "
                    "If evidence is sparse or conflicting, say that explicitly. "
                    "Prefer corroborated/primary evidence over social posts. "
                    "Answer in the user's language. Cite source markers like [S1], [S2] inline. "
                    "Do not invent events, dates, quotes, casualties, decisions, or diplomatic actions."
                ),
            },
            {"role": "system", "content": context},
            {"role": "user", "content": user_text},
        ]
        response = await chat(messages, temperature=0.15, num_predict=700, route="cloud")
        if refs:
            used = []
            for ref in refs[:8]:
                if f"[{ref['ref']}]" in response:
                    used.append(ref)
            if not used:
                used = refs[:5]
            response += "\n\n情報源:\n" + "\n".join(
                f"[{r['ref']}] {r['source']} — {r['title']}\n{r['url']}" for r in used
            )
        asyncio.create_task(_store_conversation_later(user_text, response))
        return response, []

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
