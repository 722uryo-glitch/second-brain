import asyncio
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import quote, urlparse, urlunparse

import httpx

from .db import add_external_item
from .external_collector import _parse_feed

try:
    from trafilatura import extract as trafilatura_extract
except Exception:  # optional fallback until dependency is installed
    trafilatura_extract = None

SEARXNG_ENABLED = os.getenv("SEARXNG_ENABLED", "true").lower() == "true"
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
SEARXNG_TIMEOUT_SECONDS = int(os.getenv("SEARXNG_TIMEOUT_SECONDS", "12"))
RESEARCH_WEB_LIMIT = int(os.getenv("RESEARCH_WEB_LIMIT", "12"))
RESEARCH_FETCH_TOP_K = int(os.getenv("RESEARCH_FETCH_TOP_K", "5"))
RESEARCH_FETCH_TIMEOUT_SECONDS = int(os.getenv("RESEARCH_FETCH_TIMEOUT_SECONDS", "12"))
RESEARCH_HTTP_RETRIES = int(os.getenv("RESEARCH_HTTP_RETRIES", "2"))

_STATS = {
    "searxng_queries": 0,
    "searxng_success": 0,
    "searxng_failures": 0,
    "google_news_queries": 0,
    "page_fetches": 0,
    "page_fetch_failures": 0,
    "last_error": None,
    "last_query": None,
}

_UA = "SecondBrain/2.0 local-research-agent"
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _domain(url: str):
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _canonical_url(url: str):
    try:
        p = urlparse(url)
        if not p.scheme or not p.netloc:
            return url
        # Drop fragments and common tracking parameters by discarding query only
        # when it is clearly tracking-heavy. Preserve functional queries.
        query = p.query
        if query and any(x in query.lower() for x in ("utm_", "gclid=", "fbclid=")):
            kept = []
            for part in query.split("&"):
                low = part.lower()
                if low.startswith("utm_") or low.startswith("gclid=") or low.startswith("fbclid="):
                    continue
                kept.append(part)
            query = "&".join(kept)
        return urlunparse((p.scheme, p.netloc, p.path or "/", p.params, query, ""))
    except Exception:
        return url


async def _request_with_retry(client, method, url, **kwargs):
    last = None
    attempts = max(1, RESEARCH_HTTP_RETRIES + 1)
    for attempt in range(attempts):
        try:
            r = await client.request(method, url, **kwargs)
            if r.status_code in {408, 425, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                await asyncio.sleep(min(2.5, 0.5 * (2 ** attempt)))
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if attempt + 1 < attempts:
                await asyncio.sleep(min(2.5, 0.5 * (2 ** attempt)))
    raise last or RuntimeError("request failed")


async def searxng_search(query: str, limit=None, language="all", time_range=None):
    if not SEARXNG_ENABLED:
        return []
    _STATS["searxng_queries"] += 1
    _STATS["last_query"] = query
    params = {
        "q": query[:500],
        "format": "json",
        "language": language or "all",
        "safesearch": 1,
    }
    if time_range in {"day", "week", "month", "year"}:
        params["time_range"] = time_range
    try:
        async with httpx.AsyncClient(
            timeout=SEARXNG_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            r = await _request_with_retry(client, "GET", f"{SEARXNG_BASE_URL}/search", params=params)
            data = r.json()
        rows = []
        for item in (data.get("results") or [])[: int(limit or RESEARCH_WEB_LIMIT)]:
            url = _canonical_url(str(item.get("url") or "").strip())
            title = str(item.get("title") or "").strip()
            if not url or not title:
                continue
            rows.append({
                "title": title,
                "url": url,
                "summary": str(item.get("content") or "").strip(),
                "published_at": item.get("publishedDate") or item.get("published_date"),
                "source": item.get("engine") or ",".join(item.get("engines") or []) or "SearXNG",
                "source_type": "web_search",
                "domain": _domain(url),
                "provider": "searxng",
            })
        _STATS["searxng_success"] += 1
        _STATS["last_error"] = None
        return rows
    except Exception as e:
        _STATS["searxng_failures"] += 1
        _STATS["last_error"] = f"searxng: {str(e)[:180]}"
        print(f"[RESEARCH] SearXNG failed: {str(e)[:180]}")
        return []


async def google_news_search(query: str, limit=10):
    _STATS["google_news_queries"] += 1
    url = (
        "https://news.google.com/rss/search?q=" + quote(query[:300]) +
        "&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        async with httpx.AsyncClient(
            timeout=12,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            r = await _request_with_retry(client, "GET", url)
        out = []
        for item in _parse_feed(r.text)[:limit]:
            item_url = _canonical_url(item.get("url") or "")
            out.append({
                "title": item.get("title") or "",
                "url": item_url,
                "summary": item.get("summary") or "",
                "published_at": item.get("published_at"),
                "source": "Google News",
                "source_type": "news_search",
                "domain": _domain(item_url),
                "provider": "google_news",
            })
        return out
    except Exception as e:
        _STATS["last_error"] = f"google_news: {str(e)[:180]}"
        print(f"[RESEARCH] Google News failed: {str(e)[:180]}")
        return []


def _fallback_html_text(raw: str):
    raw = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw, flags=re.I | re.S)
    raw = _TAG_RE.sub(" ", raw)
    return _SPACE_RE.sub(" ", raw).strip()


def _extract_main_text(raw: str, url: str):
    if trafilatura_extract is not None:
        try:
            text = trafilatura_extract(
                raw,
                url=url,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if text and len(text.strip()) >= 120:
                return text.strip()
        except Exception:
            pass
    return _fallback_html_text(raw)


async def fetch_result_body(item: dict):
    url = item.get("url") or ""
    if not url or not url.startswith(("http://", "https://")):
        return item
    _STATS["page_fetches"] += 1
    try:
        async with httpx.AsyncClient(
            timeout=RESEARCH_FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            r = await _request_with_retry(client, "GET", url)
        ctype = (r.headers.get("content-type") or "").lower()
        body = ""
        if "html" in ctype or not ctype:
            body = _extract_main_text(r.text, str(r.url))[:14000]
        copied = dict(item)
        copied["final_url"] = str(r.url)
        copied["url"] = _canonical_url(str(r.url))
        copied["domain"] = _domain(copied["url"])
        copied["body"] = body
        return copied
    except Exception as e:
        _STATS["page_fetch_failures"] += 1
        copied = dict(item)
        copied["fetch_error"] = str(e)[:180]
        return copied


def _dedupe(rows):
    out = []
    seen_urls = set()
    seen_title_domain = set()
    for row in rows:
        url = _canonical_url(row.get("url") or "")
        domain = row.get("domain") or _domain(url)
        title_key = re.sub(r"\W+", " ", (row.get("title") or "").lower()).strip()[:120]
        td = (title_key, domain)
        if url and url in seen_urls:
            continue
        if title_key and td in seen_title_domain:
            continue
        if url:
            seen_urls.add(url)
        if title_key:
            seen_title_domain.add(td)
        copied = dict(row)
        copied["url"] = url
        copied["domain"] = domain
        out.append(copied)
    return out


def _persist(rows, query):
    for row in rows:
        url = row.get("url") or ""
        title = row.get("title") or ""
        if not url or not title:
            continue
        md = {
            "source_type": row.get("source_type") or "web_search",
            "provider": row.get("provider"),
            "domain": row.get("domain"),
            "query": query,
            "on_demand": True,
        }
        summary = row.get("body") or row.get("summary") or ""
        try:
            add_external_item(
                f"On-demand:{row.get('provider') or 'web'}",
                title,
                url,
                row.get("published_at"),
                summary[:4000],
                md,
            )
        except Exception:
            pass


async def research_web(query: str, current=False, limit=None):
    """Search the general web and news, then fetch top pages for evidence.

    SearXNG is the preferred general-web layer because it is local, private and
    metasearch-capable. Google News remains a no-key fallback/supplement for
    current-affairs coverage.
    """
    limit = int(limit or RESEARCH_WEB_LIMIT)
    time_range = "month" if current else None
    general_task = asyncio.create_task(searxng_search(query, limit=limit, time_range=time_range))
    news_task = asyncio.create_task(google_news_search(query, limit=min(8, limit)))
    general, news = await asyncio.gather(general_task, news_task)

    rows = _dedupe(general + news)
    # Prioritize direct web results for body extraction; news redirect pages are
    # less useful as independent evidence until resolved.
    fetch_candidates = [r for r in rows if r.get("provider") == "searxng"][:max(0, RESEARCH_FETCH_TOP_K)]
    if fetch_candidates:
        fetched = await asyncio.gather(*(fetch_result_body(r) for r in fetch_candidates))
        by_url = {r.get("url"): r for r in fetched}
        merged = []
        for row in rows:
            merged.append(by_url.get(row.get("url"), row))
        rows = _dedupe(merged)

    rows = rows[:limit]
    _persist(rows, query)
    return rows


def research_status():
    return {
        "searxng_enabled": SEARXNG_ENABLED,
        "searxng_base_url": SEARXNG_BASE_URL,
        "trafilatura_available": trafilatura_extract is not None,
        "web_limit": RESEARCH_WEB_LIMIT,
        "fetch_top_k": RESEARCH_FETCH_TOP_K,
        "stats": dict(_STATS),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
