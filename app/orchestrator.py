import asyncio
import re
from datetime import datetime, timezone
from urllib.parse import quote

import httpx

from .db import search_external_items, search_claims, claim_evidence_sources
from .external_collector import _parse_feed


RESEARCH_MARKERS = (
    "調べ", "リサーチ", "海外", "需要", "供給", "市場", "競合", "ニッチ", "トレンド",
    "比較", "検証", "根拠", "ソース", "記事", "アフィリエイト", "ブログ", "レポート",
    "research", "market", "demand", "supply", "competitor", "trend", "affiliate",
)

CURRENT_MARKERS = (
    "現在", "今", "いま", "最新", "今日", "昨日", "今週", "情勢", "ニュース", "速報",
    "どうなって", "どうなった", "最近", "現状", "リアルタイム", "latest", "current",
    "today", "now", "news", "situation", "update",
)

ALIASES = {
    "アメリカ": ["アメリカ", "米国", "United States", "US"],
    "米国": ["米国", "アメリカ", "United States", "US"],
    "イラン": ["イラン", "Iran"],
    "中国": ["中国", "China"],
    "ロシア": ["ロシア", "Russia"],
    "ウクライナ": ["ウクライナ", "Ukraine"],
    "イスラエル": ["イスラエル", "Israel"],
    "ガザ": ["ガザ", "Gaza"],
    "日本": ["日本", "Japan"],
    "台湾": ["台湾", "Taiwan"],
    "openai": ["OpenAI"],
    "chatgpt": ["ChatGPT", "OpenAI"],
    "gemini": ["Gemini", "Google"],
    "claude": ["Claude", "Anthropic"],
    "アフィリエイト": ["affiliate marketing", "affiliate niche", "アフィリエイト"],
    "需要": ["demand", "需要"],
    "供給": ["supply", "competition", "供給", "競合"],
}


def is_research_task(text: str) -> bool:
    t = text.lower()
    return any(marker.lower() in t for marker in RESEARCH_MARKERS)


def is_current_task(text: str) -> bool:
    t = text.lower()
    return any(marker.lower() in t for marker in CURRENT_MARKERS)


def search_terms(text: str, max_terms: int = 14):
    t = text.lower()
    terms = []
    for needle, aliases in ALIASES.items():
        if needle.lower() in t:
            terms.extend(aliases)

    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9.\-]{1,40}", text))
    cleaned = text
    for stop in (
        "について教えて", "について", "教えて", "調べて", "調べ", "現在", "最新", "今日", "情勢",
        "ニュース", "現状", "記事を書いて", "記事", "書いて", "作って", "してほしい", "して",
    ):
        cleaned = cleaned.replace(stop, " ")
    terms.extend(re.findall(r"[一-龥ぁ-んァ-ヶー]{2,14}", cleaned))

    out = []
    seen = set()
    for term in terms:
        term = term.strip()
        key = term.lower()
        if len(term) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(term[:80])
        if len(out) >= max_terms:
            break
    return out or [text[:60]]


def _local_evidence(text: str, item_limit: int = 18, claim_limit: int = 8):
    terms = search_terms(text)
    items = search_external_items(terms, limit=item_limit)
    claims = search_claims(terms, limit=claim_limit)
    return terms, items, claims


async def _google_news_search(query: str, limit: int = 12):
    # Public, no-key on-demand research. This complements the background archive.
    url = (
        "https://news.google.com/rss/search?q=" + quote(query[:240]) +
        "&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        async with httpx.AsyncClient(
            timeout=12,
            follow_redirects=True,
            headers={"User-Agent": "SecondBrain-V1/1.1 research-orchestrator"},
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            return _parse_feed(r.text)[:limit]
    except Exception as e:
        print(f"[SECOND-BRAIN] on-demand Google News failed: {str(e)[:180]}")
        return []


async def gather_research_context(user_text: str, on_demand: bool = True):
    """Build an evidence pack from the Second Brain's archive + live public research.

    Important: this does not ask the model to invent research. It supplies the model
    with actual collected evidence, and explicitly exposes when evidence is weak.
    """
    terms, items, claims = _local_evidence(user_text)

    live_rows = []
    if on_demand:
        # Keep this to one request so research mode does not become painfully slow.
        query_terms = [t for t in terms if len(t) <= 40][:6]
        query = " ".join(query_terms) if query_terms else user_text
        live_rows = await _google_news_search(query, limit=12)

    lines = [
        "SECOND BRAIN EVIDENCE PACK",
        f"UTC_NOW={datetime.now(timezone.utc).isoformat()}",
        f"TERMS={', '.join(terms)}",
        f"ARCHIVE_ITEMS={len(items)} ARCHIVE_CLAIMS={len(claims)} LIVE_ITEMS={len(live_rows)}",
        "",
        "ARCHIVED EXTERNAL INTELLIGENCE:",
    ]

    refs = []
    ref_no = 1
    for item in items:
        ref = f"S{ref_no}"
        ref_no += 1
        refs.append({"ref": ref, "title": item.get("title") or "", "url": item.get("url") or "", "source": item.get("source") or ""})
        summary = (item.get("summary") or "").replace("\n", " ")[:320]
        lines.append(
            f"[{ref}] source={item.get('source') or ''} time={item.get('published_at') or item.get('collected_at') or ''} "
            f"title={item.get('title') or ''} summary={summary} url={item.get('url') or ''}"
        )

    if live_rows:
        lines.append("")
        lines.append("ON-DEMAND PUBLIC RESEARCH:")
        for item in live_rows:
            ref = f"S{ref_no}"
            ref_no += 1
            refs.append({"ref": ref, "title": item.get("title") or "", "url": item.get("url") or "", "source": "Google News search"})
            summary = (item.get("summary") or "").replace("\n", " ")[:320]
            lines.append(
                f"[{ref}] source=Google News search time={item.get('published_at') or ''} "
                f"title={item.get('title') or ''} summary={summary} url={item.get('url') or ''}"
            )

    if claims:
        lines.append("")
        lines.append("FACT-CHECKED CLAIMS:")
    for claim in claims:
        lines.append(
            f"- status={claim.get('status')} confidence={float(claim.get('confidence') or 0):.2f} "
            f"independent_sources={claim.get('independent_sources')} contradictions={claim.get('contradictions')} "
            f"claim={claim.get('claim_text')}"
        )
        for ev in claim_evidence_sources(claim.get("id"), limit=2):
            lines.append(
                f"  evidence stance={ev.get('stance')} credibility={ev.get('credibility')} "
                f"source={ev.get('source')} url={ev.get('url')}"
            )

    enough = (len(items) + len(live_rows)) >= 4 or len(claims) >= 2
    lines.append("")
    lines.append(f"EVIDENCE_SUFFICIENT={'yes' if enough else 'no'}")
    return "\n".join(lines), refs, enough
