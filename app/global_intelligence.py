import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

import httpx

from .config import (
    EXTERNAL_CONCURRENCY, EXTERNAL_FEEDS, EXTERNAL_ITEMS_PER_FEED,
    GLOBAL_NEWS_LOCALES, GDELT_ENABLED, GDELT_QUERIES,
    GITHUB_ENABLED, GITHUB_EVENT_PAGES, GITHUB_TOKEN,
    X_ENABLED, X_BEARER_TOKEN, X_QUERIES,
    FACTCHECK_BATCH_SIZE,
)
from .db import (
    add_external_item, unprocessed_external_items, upsert_claim,
    add_claim_evidence, recent_claims,
)
from .external_collector import _parse_feed
from .ollama_client import chat

UA = "SecondBrain-V1/1.0 local-intelligence-agent"


def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _credibility(domain: str, source_type: str) -> float:
    if source_type == "github":
        return 0.85
    if source_type == "x":
        return 0.25
    if domain.endswith(".gov") or domain.endswith(".gov.uk") or domain.endswith(".europa.eu"):
        return 0.92
    if any(x in domain for x in ("who.int", "un.org", "nasa.gov", "ec.europa.eu", "arxiv.org")):
        return 0.88
    if source_type in {"news", "gdelt", "rss"}:
        return 0.62
    return 0.5


def _google_news_url(country: str, hl: str, language: str) -> str:
    ceid_lang = language if language not in {"zh"} else ("zh-Hant" if country in {"TW", "HK"} else "zh-Hans")
    return f"https://news.google.com/rss?hl={quote(hl)}&gl={country}&ceid={country}:{quote(ceid_lang)}"


async def _ingest_feed(client, source, url, metadata, sem):
    async with sem:
        try:
            r = await client.get(url)
            r.raise_for_status()
            items = _parse_feed(r.text)[:EXTERNAL_ITEMS_PER_FEED]
            new = 0
            for item in items:
                md = dict(metadata)
                md.update({"feed_url": url, "source_type": metadata.get("source_type", "rss")})
                if add_external_item(source, item["title"], item["url"], item.get("published_at"), item.get("summary"), md):
                    new += 1
            return {"source": source, "fetched": len(items), "new": new}
        except Exception as e:
            return {"source": source, "fetched": 0, "new": 0, "error": str(e)[:240]}


async def _collect_gdelt(client, sem):
    if not GDELT_ENABLED:
        return []
    results = []
    for query in GDELT_QUERIES:
        async with sem:
            try:
                params = {"query": query, "mode": "artlist", "maxrecords": 250, "format": "json", "timespan": "30min", "sort": "datedesc"}
                r = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
                r.raise_for_status()
                data = r.json()
                articles = data.get("articles", [])
                new = 0
                for a in articles:
                    url = a.get("url") or ""
                    title = a.get("title") or ""
                    if not url or not title:
                        continue
                    md = {"source_type": "gdelt", "language": a.get("language"), "sourcecountry": a.get("sourcecountry"),
                          "domain": a.get("domain"), "query": query}
                    if add_external_item(f"GDELT:{a.get('domain') or 'global'}", title, url, a.get("seendate"), "", md):
                        new += 1
                results.append({"source": f"GDELT:{query[:24]}", "fetched": len(articles), "new": new})
            except Exception as e:
                results.append({"source": f"GDELT:{query[:24]}", "fetched": 0, "new": 0, "error": str(e)[:240]})
    return results


async def _collect_github(client, sem):
    if not GITHUB_ENABLED:
        return []
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    results = []
    for page in range(1, max(1, GITHUB_EVENT_PAGES) + 1):
        async with sem:
            try:
                r = await client.get("https://api.github.com/events", params={"per_page": 100, "page": page}, headers=headers)
                r.raise_for_status()
                events = r.json()
                new = 0
                for ev in events:
                    repo = (ev.get("repo") or {}).get("name") or "unknown"
                    etype = ev.get("type") or "GitHubEvent"
                    actor = (ev.get("actor") or {}).get("login") or "unknown"
                    eid = str(ev.get("id") or hashlib.sha1(json.dumps(ev, sort_keys=True).encode()).hexdigest())
                    url = f"https://github.com/{repo}#event-{eid}"
                    title = f"{etype}: {actor} @ {repo}"
                    md = {"source_type": "github", "event_type": etype, "actor": actor, "repo": repo, "payload": ev.get("payload")}
                    if add_external_item("GitHub Public Events", title, url, ev.get("created_at"), "", md):
                        new += 1
                results.append({"source": f"GitHub events p{page}", "fetched": len(events), "new": new})
            except Exception as e:
                results.append({"source": f"GitHub events p{page}", "fetched": 0, "new": 0, "error": str(e)[:240]})
                break
    return results


async def _collect_x(client, sem):
    if not X_ENABLED or not X_BEARER_TOKEN:
        return [{"source": "X", "fetched": 0, "new": 0, "skipped": "X_BEARER_TOKEN not configured"}]
    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    results = []
    for query in X_QUERIES:
        async with sem:
            try:
                params = {"query": f"({query}) -is:retweet", "max_results": 100, "tweet.fields": "created_at,lang,author_id,public_metrics"}
                r = await client.get("https://api.x.com/2/tweets/search/recent", params=params, headers=headers)
                r.raise_for_status()
                posts = r.json().get("data", [])
                new = 0
                for p in posts:
                    pid = p.get("id")
                    text = (p.get("text") or "").strip()
                    if not pid or not text:
                        continue
                    md = {"source_type": "x", "language": p.get("lang"), "author_id": p.get("author_id"), "metrics": p.get("public_metrics"), "query": query}
                    if add_external_item("X", text[:240], f"https://x.com/i/web/status/{pid}", p.get("created_at"), text, md):
                        new += 1
                results.append({"source": f"X:{query}", "fetched": len(posts), "new": new})
            except Exception as e:
                results.append({"source": f"X:{query}", "fetched": 0, "new": 0, "error": str(e)[:240]})
    return results


async def collect_global_information():
    sem = asyncio.Semaphore(max(2, EXTERNAL_CONCURRENCY))
    headers = {"User-Agent": UA}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
        feed_tasks = []
        for source, url in EXTERNAL_FEEDS:
            feed_tasks.append(_ingest_feed(client, source, url, {"source_type": "rss"}, sem))
        for country, hl, language in GLOBAL_NEWS_LOCALES:
            source = f"Google News {country}/{hl}"
            url = _google_news_url(country, hl, language)
            feed_tasks.append(_ingest_feed(client, source, url, {"source_type": "news", "country": country, "language": language}, sem))
        feed_results = await asyncio.gather(*feed_tasks)
        gdelt_results, github_results, x_results = await asyncio.gather(
            _collect_gdelt(client, sem), _collect_github(client, sem), _collect_x(client, sem)
        )
    all_results = list(feed_results) + gdelt_results + github_results + x_results
    return {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "sources": len(all_results),
        "fetched": sum(int(r.get("fetched", 0)) for r in all_results),
        "new": sum(int(r.get("new", 0)) for r in all_results),
        "errors": [r for r in all_results if r.get("error")],
        "skipped": [r for r in all_results if r.get("skipped")],
        "details": all_results,
    }


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    return []


async def factcheck_batch(limit=None):
    items = unprocessed_external_items(limit or FACTCHECK_BATCH_SIZE)
    if not items:
        return {"processed": 0, "claims": 0}
    compact = [{"id": i["id"], "source": i["source"], "title": i["title"], "summary": (i.get("summary") or "")[:500]} for i in items]
    prompt = """You are a multilingual claim normalizer for a fact-check system. For every item, return one JSON array only.
Each object: {id, claim_key, canonical_claim, stance, language}.
- canonical_claim: one concise factual proposition in English. Preserve uncertainty. If the item is not a checkable factual claim, use an informative event description.
- claim_key: stable snake_case key for the underlying proposition/event, omitting source wording and language so independent reports converge.
- stance: supports or contradicts relative to the affirmative canonical proposition.
Do not decide truth from one source. Do not add facts not present in the item.
Items:\n""" + json.dumps(compact, ensure_ascii=False)
    try:
        raw = await chat([{"role": "system", "content": "Return strict JSON only."}, {"role": "user", "content": prompt}], temperature=0.1, num_predict=1400)
        normalized = _extract_json(raw)
    except Exception:
        normalized = []
    by_id = {int(x.get("id")): x for x in normalized if str(x.get("id", "")).isdigit()}
    claims = 0
    for item in items:
        n = by_id.get(int(item["id"])) or {}
        canonical = (n.get("canonical_claim") or item["title"]).strip()
        key = (n.get("claim_key") or hashlib.sha1(canonical.lower().encode("utf-8")).hexdigest()).strip().lower()
        key = re.sub(r"[^a-z0-9_:-]+", "_", key)[:220]
        stance = n.get("stance") if n.get("stance") in {"supports", "contradicts"} else "supports"
        try:
            md = json.loads(item.get("metadata_json") or "{}")
        except Exception:
            md = {}
        source_type = md.get("source_type", "unknown")
        domain = md.get("domain") or _domain(item["url"])
        claim_id = upsert_claim(key, canonical, {"language": n.get("language"), "first_source": item["source"]})
        add_claim_evidence(claim_id, item["id"], domain, source_type, stance, _credibility(domain, source_type))
        claims += 1
    return {"processed": len(items), "claims": claims}


async def global_collection_loop(interval_minutes: int):
    while True:
        try:
            r = await collect_global_information()
            print(f"[GLOBAL] sources={r['sources']} fetched={r['fetched']} new={r['new']} errors={len(r['errors'])} skipped={len(r['skipped'])}")
        except Exception as e:
            print(f"[GLOBAL] failed: {e}")
        await asyncio.sleep(max(5, interval_minutes) * 60)


async def factcheck_loop(interval_seconds: int):
    while True:
        try:
            r = await factcheck_batch()
            if r["processed"]:
                print(f"[FACTCHECK] processed={r['processed']} claims={r['claims']}")
        except Exception as e:
            print(f"[FACTCHECK] failed: {e}")
        await asyncio.sleep(max(5, interval_seconds))


def intelligence_status():
    claims = recent_claims(50)
    counts = {}
    for c in claims:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"recent_claims": len(claims), "statuses": counts, "x_configured": bool(X_BEARER_TOKEN), "github_authenticated": bool(GITHUB_TOKEN)}
