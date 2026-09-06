import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from .db import claim_evidence_sources
from .retrieval import search_external_ranked, search_claims_ranked
from .web_research import research_web, research_status


RESEARCH_MARKERS = (
    "調べ", "リサーチ", "海外", "需要", "供給", "市場", "競合", "ニッチ", "トレンド",
    "比較", "検証", "根拠", "ソース", "記事", "アフィリエイト", "ブログ", "レポート",
    "research", "market", "demand", "supply", "competitor", "trend", "affiliate",
    "search", "evidence", "verify",
)

CURRENT_MARKERS = (
    "現在", "今", "いま", "最新", "今日", "昨日", "今週", "情勢", "ニュース", "速報",
    "どうなって", "どうなった", "最近", "現状", "リアルタイム", "latest", "current",
    "today", "now", "news", "situation", "update",
)

ALIASES = {
    "アメリカ": ["United States", "US", "米国"],
    "米国": ["United States", "US", "アメリカ"],
    "イラン": ["Iran", "イラン"],
    "中国": ["China", "中国"],
    "ロシア": ["Russia", "ロシア"],
    "ウクライナ": ["Ukraine", "ウクライナ"],
    "イスラエル": ["Israel", "イスラエル"],
    "ガザ": ["Gaza", "ガザ"],
    "日本": ["Japan", "日本"],
    "台湾": ["Taiwan", "台湾"],
    "openai": ["OpenAI"],
    "chatgpt": ["ChatGPT", "OpenAI"],
    "gemini": ["Gemini", "Google"],
    "claude": ["Claude", "Anthropic"],
    "アフィリエイト": ["affiliate marketing", "affiliate niche", "アフィリエイト"],
    "需要": ["demand", "需要"],
    "供給": ["supply", "competition", "供給", "競合"],
}

_STOP_PHRASES = (
    "について教えて", "について", "教えて", "調べて", "調べ", "現在", "最新", "今日", "情勢",
    "ニュース", "現状", "記事を書いて", "記事", "書いて", "作って", "してほしい", "して",
    "お願いします", "よろしく", "元に", "含めて",
)


def is_research_task(text: str) -> bool:
    t = str(text or "").lower()
    return any(marker.lower() in t for marker in RESEARCH_MARKERS)


def is_current_task(text: str) -> bool:
    t = str(text or "").lower()
    return any(marker.lower() in t for marker in CURRENT_MARKERS)


def search_terms(text: str, max_terms: int = 14):
    text = str(text or "")
    t = text.lower()
    terms = []
    for needle, aliases in ALIASES.items():
        if needle.lower() in t:
            terms.extend(aliases)

    # Keep explicit Latin entities/products intact.
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9._\-]{1,50}", text))

    cleaned = text
    for stop in _STOP_PHRASES:
        cleaned = cleaned.replace(stop, " ")

    # Japanese noun-like spans. The hybrid retrieval layer handles substrings.
    terms.extend(re.findall(r"[一-龥ぁ-んァ-ヶー]{2,18}", cleaned))

    out = []
    seen = set()
    for raw in terms:
        term = re.sub(r"\s+", " ", raw).strip(" 、。,.!?！？:：")
        key = term.lower()
        if len(term) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(term[:80])
        if len(out) >= max_terms:
            break
    return out or [text[:80]]


def _domain(url: str):
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _local_evidence(text: str, item_limit: int = 18, claim_limit: int = 8):
    terms = search_terms(text)
    items = search_external_ranked(terms, limit=item_limit)
    claims = search_claims_ranked(terms, limit=claim_limit)
    return terms, items, claims


def _ref_dict(ref, source, title, url, time_value="", domain="", provider="archive", source_type=""):
    return {
        "ref": ref,
        "source": source or "",
        "title": title or "",
        "url": url or "",
        "time": time_value or "",
        "domain": domain or _domain(url or ""),
        "provider": provider,
        "source_type": source_type or "",
    }


def _independent_domains(refs):
    domains = set()
    for ref in refs:
        d = (ref.get("domain") or "").lower()
        if not d or d in {"news.google.com"}:
            continue
        domains.add(d)
    return domains


async def gather_research_context(user_text: str, on_demand: bool = True):
    """Build an evidence pack from persistent memory plus on-demand web research.

    The pack is intentionally evidence-first. The writer receives source ids,
    provenance, timestamps, direct URLs and (when available) extracted page body.
    """
    terms, items, claims = _local_evidence(user_text)
    live_rows = []
    if on_demand:
        live_rows = await research_web(
            user_text,
            current=is_current_task(user_text),
        )

    lines = [
        "SECOND BRAIN EVIDENCE PACK",
        f"UTC_NOW={datetime.now(timezone.utc).isoformat()}",
        f"QUERY={user_text}",
        f"TERMS={', '.join(terms)}",
        f"ARCHIVE_ITEMS={len(items)} ARCHIVE_CLAIMS={len(claims)} LIVE_ITEMS={len(live_rows)}",
        "",
        "ARCHIVED EXTERNAL INTELLIGENCE:",
    ]

    refs = []
    ref_no = 1
    seen_urls = set()

    for item in items:
        url = item.get("url") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        ref = f"S{ref_no}"
        ref_no += 1
        refs.append(_ref_dict(
            ref,
            item.get("source"),
            item.get("title"),
            url,
            item.get("published_at") or item.get("collected_at"),
            _domain(url),
            "archive",
            "archive",
        ))
        summary = (item.get("summary") or "").replace("\n", " ")[:700]
        lines.append(
            f"[{ref}] provider=archive domain={_domain(url)} "
            f"time={item.get('published_at') or item.get('collected_at') or ''} "
            f"source={item.get('source') or ''} retrieval_score={item.get('retrieval_score', '')} "
            f"title={item.get('title') or ''} summary={summary} url={url}"
        )

    if live_rows:
        lines.extend(["", "ON-DEMAND WEB RESEARCH:"])
        for item in live_rows:
            url = item.get("url") or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            ref = f"S{ref_no}"
            ref_no += 1
            domain = item.get("domain") or _domain(url)
            refs.append(_ref_dict(
                ref,
                item.get("source") or item.get("provider"),
                item.get("title"),
                url,
                item.get("published_at"),
                domain,
                item.get("provider") or "web",
                item.get("source_type") or "web_search",
            ))
            summary = (item.get("summary") or "").replace("\n", " ")[:700]
            body = (item.get("body") or "").replace("\n", " ")[:2200]
            lines.append(
                f"[{ref}] provider={item.get('provider') or 'web'} domain={domain} "
                f"time={item.get('published_at') or ''} source={item.get('source') or ''} "
                f"title={item.get('title') or ''} summary={summary} "
                f"body_excerpt={body} url={url}"
            )

    if claims:
        lines.extend(["", "FACT-CHECKED CLAIMS:"])
    for claim in claims:
        lines.append(
            f"- status={claim.get('status')} confidence={float(claim.get('confidence') or 0):.2f} "
            f"independent_sources={claim.get('independent_sources')} contradictions={claim.get('contradictions')} "
            f"retrieval_score={claim.get('retrieval_score', '')} claim={claim.get('claim_text')}"
        )
        for ev in claim_evidence_sources(claim.get("id"), limit=4):
            lines.append(
                f"  evidence stance={ev.get('stance')} credibility={ev.get('credibility')} "
                f"domain={ev.get('source_domain') or _domain(ev.get('url') or '')} "
                f"source_type={ev.get('source_type')} source={ev.get('source')} url={ev.get('url')}"
            )

    domains = _independent_domains(refs)
    # This is a structural sufficiency check, not a semantic proof. The executive
    # layer performs a second evidence-gap audit against the actual subquestions.
    enough = (
        (len(refs) >= 4 and len(domains) >= 2)
        or any(c.get("status") == "corroborated" for c in claims)
    )
    lines.extend([
        "",
        f"INDEPENDENT_DIRECT_DOMAINS={len(domains)}",
        f"EVIDENCE_STRUCTURALLY_SUFFICIENT={'yes' if enough else 'no'}",
        "NOTE=Structural sufficiency does not prove every requested facet. The executive must audit coverage before drafting.",
    ])
    return "\n".join(lines), refs, enough


def orchestrator_status():
    return {
        "research": research_status(),
        "capabilities": [
            "persistent_archive",
            "ranked_hybrid_retrieval",
            "fact_checked_claims",
            "searxng_general_web",
            "google_news_fallback",
            "direct_page_extraction",
            "independent_domain_counting",
        ],
    }
