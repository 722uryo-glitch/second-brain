import asyncio
import re
from datetime import datetime, timezone

from .memory import recall, store_memory
from .ollama_client import chat
from .db import (
    recent_memories,
    recent_conversation,
    search_external_items,
    search_claims,
    claim_evidence_sources,
)

SYSTEM = """あなたはユーザー専用の『第2の脳』です。
目的は、過去の文脈・現在の依頼・必要な知識を統合して、実際に役立つ答えと成果物を返すことです。

行動原則:
- ユーザーが『任せる』『なんでもいい』『調べてやって』と言ったら、合理的な仮定を置いて自分で進める。不要な確認質問をしない。
- 前の発言で既に分かっていることを、もう一度質問しない。
- 質問に答えるだけでなく、依頼なら実行可能な形まで具体化する。
- 迷った場合は最も妥当な選択肢を選び、その選択を短く明示して進める。
- ありきたりな一般論だけで終わらせない。
- 返答は自然で、必要十分な具体性を持たせる。

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
_PERSONAL_MARKERS = (
    "私の", "僕の", "俺の", "自分の", "覚えて", "前に話", "前回", "好み", "予定", "タスク",
    "住所", "電話", "メール", "家族", "友達", "学校", "職場", "仕事の", "名前", "誕生日",
)
_ALIASES = {
    "アメリカ": ["アメリカ", "米国", "United States", "US"],
    "米国": ["米国", "アメリカ", "United States", "US"],
    "イラン": ["イラン", "Iran"],
    "中国": ["中国", "China"],
    "ロシア": ["ロシア", "Russia"],
    "ウクライナ": ["ウクライナ", "Ukraine"],
    "イスラエル": ["イスラエル", "Israel"],
    "ガザ": ["ガザ", "Gaza"],
    "北朝鮮": ["北朝鮮", "North Korea", "DPRK"],
    "韓国": ["韓国", "South Korea"],
    "日本": ["日本", "Japan"],
    "台湾": ["台湾", "Taiwan"],
    "openai": ["OpenAI"],
    "chatgpt": ["ChatGPT", "OpenAI"],
    "gemini": ["Gemini", "Google"],
    "claude": ["Claude", "Anthropic"],
}


def _is_plain_greeting(text: str) -> bool:
    t = text.strip().lower().replace("！", "!").replace("？", "?")
    return t in {
        "こんにちは", "こんばんは", "おはよう", "おはようございます",
        "やあ", "どうも", "hello", "hi", "hey", "こんばんは!", "こんにちは!"
    }


def _is_current_public_question(text: str) -> bool:
    t = text.lower()
    has_current = any(x in t for x in _CURRENT_MARKERS)
    has_public = any(x in t for x in _PUBLIC_TOPIC_MARKERS)
    return has_current and (has_public or "情勢" in t or "ニュース" in t)


def _looks_personal(text: str) -> bool:
    t = text.lower()
    return any(x.lower() in t for x in _PERSONAL_MARKERS)


def _fast_search_terms(user_text: str):
    t = user_text.lower()
    terms = []
    for needle, aliases in _ALIASES.items():
        if needle.lower() in t:
            terms.extend(aliases)
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9.\-]{1,30}", user_text))
    for seq in re.findall(r"[ァ-ヶー]{4,20}", user_text):
        for size in (6, 5, 4, 3):
            if len(seq) >= size:
                for i in range(0, len(seq) - size + 1):
                    terms.append(seq[i:i + size])
    cleaned = user_text
    for stop in ("について教えて", "について", "教えて", "現在", "最新", "今日", "情勢", "ニュース", "現状"):
        cleaned = cleaned.replace(stop, " ")
    terms.extend(re.findall(r"[一-龥]{2,8}", cleaned))
    out, seen = [], set()
    for term in terms:
        term = term.strip()
        key = term.lower()
        if len(term) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= 12:
            break
    return out or [user_text[:40]]


def _short_history(limit=8):
    rows = recent_conversation(limit)
    messages = []
    for row in rows:
        role = "user" if row.get("source") == "user" else "assistant"
        content = str(row.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content[:1800]})
    return messages


async def _public_intelligence_context(user_text: str):
    terms = _fast_search_terms(user_text)
    items = search_external_items(terms, limit=12)
    claims = search_claims(terms, limit=5)
    lines = [
        f"CURRENT UTC TIME: {datetime.now(timezone.utc).isoformat()}",
        f"SEARCH TERMS: {', '.join(terms)}",
        "RECENT SOURCES:",
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
        summary = (item.get("summary") or "").replace("\n", " ")[:280]
        lines.append(
            f"[{ref}] {item.get('published_at') or item.get('collected_at') or ''} | "
            f"{item.get('source') or ''} | {item.get('title') or ''} | {summary} | {item.get('url') or ''}"
        )
    if claims:
        lines.append("FACT-CHECK CLAIMS:")
    for claim in claims:
        lines.append(
            f"- {claim.get('status')} conf={float(claim.get('confidence') or 0):.2f} "
            f"sources={claim.get('independent_sources')} contradictions={claim.get('contradictions')} | "
            f"{claim.get('claim_text')}"
        )
        for ev in claim_evidence_sources(claim.get("id"), limit=2):
            lines.append(
                f"  evidence {ev.get('stance')} cred={ev.get('credibility')} | "
                f"{ev.get('source')} | {ev.get('url')}"
            )
    return "\n".join(lines), source_refs


async def _store_conversation_later(user_text: str, response: str):
    try:
        await store_memory("conversation", "user", user_text, 0.55)
        await store_memory("conversation", "assistant", response, 0.20)
    except Exception as e:
        print(f"[MEMORY] background save failed: {e}")


async def answer(user_text: str):
    # 1) Current public affairs: local RAG + cloud synthesis.
    if _is_current_public_question(user_text):
        context, refs = await _public_intelligence_context(user_text)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the current-affairs analysis module of a Second Brain. "
                    "Use only supplied collected intelligence for current-event claims. "
                    "If evidence is sparse or conflicting, say so. Prefer primary/corroborated evidence. "
                    "Answer in the user's language. Cite [S1], [S2] inline. Do not invent facts."
                ),
            },
            {"role": "system", "content": context},
            {"role": "user", "content": user_text},
        ]
        response = await chat(messages, temperature=0.1, num_predict=420, route="fast_cloud")
        if refs:
            used = [r for r in refs[:6] if f"[{r['ref']}]" in response] or refs[:4]
            response += "\n\n情報源:\n" + "\n".join(
                f"[{r['ref']}] {r['source']} — {r['title']}\n{r['url']}" for r in used
            )
        asyncio.create_task(_store_conversation_later(user_text, response))
        return response, []

    # 2) Generic work/questions: use the stronger cloud model with recent dialogue.
    #    No long-term personal memory is sent in this lane.
    if not _looks_personal(user_text):
        history = _short_history(8)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the main reasoning/working module of a user's Second Brain. "
                    "Continue the conversation naturally using the recent dialogue. "
                    "Do not repeat questions already answered. "
                    "When the user delegates choices or says 'anything is fine', choose a sensible option and proceed. "
                    "Prefer concrete execution over generic advice or unnecessary clarification. "
                    "Answer in the user's language."
                ),
            },
            *history,
            {"role": "user", "content": user_text},
        ]
        response = await chat(messages, temperature=0.25, num_predict=520, route="fast_cloud")
        asyncio.create_task(_store_conversation_later(user_text, response))
        return response, []

    # 3) Personal-memory lane: keep it local and use semantic long-term memory.
    memories = [] if _is_plain_greeting(user_text) else await recall(user_text, top_k=6)
    memory_text = "\n".join(
        f"- [{m['kind']}/{m['source']}] {m['content']}" for m in memories
    ) or "（関連記憶なし）"
    messages = [
        {"role": "system", "content": SYSTEM},
        *_short_history(8),
        {
            "role": "system",
            "content": (
                "以下は検索で見つかった長期記憶です。現在の発言に直接関係するものだけ使ってください。"
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
