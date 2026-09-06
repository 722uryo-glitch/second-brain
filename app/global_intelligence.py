import asyncio
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

import httpx

from .config import (
    EXTERNAL_CONCURRENCY, EXTERNAL_FEEDS, EXTERNAL_ITEMS_PER_FEED,
    GLOBAL_NEWS_LOCALES, GDELT_ENABLED, GDELT_QUERIES,
    GITHUB_ENABLED, GITHUB_EVENT_PAGES, GITHUB_TOKEN, GITHUB_SEARCH_QUERIES,
    X_ENABLED, X_BEARER_TOKEN, X_QUERIES,
    BLUESKY_ENABLED, REDDIT_ENABLED, MASTODON_ENABLED,
    SOCIAL_QUERIES, MASTODON_INSTANCES, PRIMARY_SOURCE_FEEDS,
    DOCUMENT_FETCH_ENABLED, DOCUMENT_FETCH_BATCH_SIZE, DOCUMENT_FETCH_CONCURRENCY,
    FACTCHECK_BATCH_SIZE, FACTCHECK_MAX_BATCH_SIZE,
)
from .db import (
    add_external_item, unprocessed_external_items, upsert_claim,
    add_claim_evidence, recent_claims,
)
from .external_collector import _parse_feed
from .ollama_client import chat
from .v1_storage import (
    record_source_result, start_collection_run, finish_collection_run,
    pending_document_items, save_document, document_for_item,
    queue_metrics, source_health,
)

UA = "SecondBrain-V1/1.0 local-intelligence-agent"
_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _credibility(domain: str, source_type: str) -> float:
    domain = (domain or "").lower()
    if source_type == "primary":
        return 0.96
    if source_type == "github":
        return 0.82
    if source_type in {"x", "reddit", "mastodon", "bluesky"}:
        return 0.28
    if domain.endswith(".gov") or ".gov." in domain or domain.endswith(".europa.eu"):
        return 0.94
    if any(x in domain for x in ("who.int", "un.org", "nasa.gov", "ec.europa.eu", "cisa.gov")):
        return 0.94
    if "arxiv.org" in domain:
        return 0.80
    if source_type in {"news", "gdelt", "rss"}:
        return 0.62
    return 0.50


def _google_news_url(country: str, hl: str, language: str) -> str:
    ceid_lang = language if language != "zh" else ("zh-Hant" if country in {"TW", "HK"} else "zh-Hans")
    return f"https://news.google.com/rss?hl={quote(hl)}&gl={country}&ceid={country}:{quote(ceid_lang)}"


def _result(source, source_type, fetched=0, new=0, error=None, skipped=None):
    out = {"source": source, "source_type": source_type, "fetched": int(fetched), "new": int(new)}
    if error:
        out["error"] = str(error)[:300]
    if skipped:
        out["skipped"] = skipped
    record_source_result(source, source_type, fetched, new, str(error)[:300] if error else None)
    return out


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
            return _result(source, md.get("source_type", "rss"), len(items), new)
        except Exception as e:
            return _result(source, metadata.get("source_type", "rss"), error=e)


async def _collect_gdelt(client, sem):
    if not GDELT_ENABLED:
        return [_result("GDELT", "gdelt", skipped="disabled")]
    results = []
    async def one(query):
        async with sem:
            source = f"GDELT:{query[:30]}"
            try:
                params = {"query": query, "mode": "artlist", "maxrecords": 250, "format": "json", "timespan": "30min", "sort": "datedesc"}
                r = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
                r.raise_for_status()
                articles = r.json().get("articles", [])
                new = 0
                for a in articles:
                    url, title = a.get("url") or "", a.get("title") or ""
                    if not url or not title:
                        continue
                    md = {"source_type": "gdelt", "language": a.get("language"), "sourcecountry": a.get("sourcecountry"),
                          "domain": a.get("domain"), "query": query}
                    if add_external_item(f"GDELT:{a.get('domain') or 'global'}", title, url, a.get("seendate"), "", md):
                        new += 1
                return _result(source, "gdelt", len(articles), new)
            except Exception as e:
                return _result(source, "gdelt", error=e)
    return await asyncio.gather(*(one(q) for q in GDELT_QUERIES))


async def _collect_github(client, sem):
    if not GITHUB_ENABLED:
        return [_result("GitHub", "github", skipped="disabled")]
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    tasks = []

    async def events(page):
        async with sem:
            source = f"GitHub events p{page}"
            try:
                r = await client.get("https://api.github.com/events", params={"per_page": 100, "page": page}, headers=headers)
                r.raise_for_status()
                rows = r.json()
                new = 0
                for ev in rows:
                    repo = (ev.get("repo") or {}).get("name") or "unknown"
                    etype = ev.get("type") or "GitHubEvent"
                    actor = (ev.get("actor") or {}).get("login") or "unknown"
                    eid = str(ev.get("id") or hashlib.sha1(json.dumps(ev, sort_keys=True).encode()).hexdigest())
                    title = f"{etype}: {actor} @ {repo}"
                    md = {"source_type": "github", "event_type": etype, "actor": actor, "repo": repo, "payload": ev.get("payload")}
                    if add_external_item("GitHub Public Events", title, f"https://github.com/{repo}#event-{eid}", ev.get("created_at"), "", md):
                        new += 1
                return _result(source, "github", len(rows), new)
            except Exception as e:
                return _result(source, "github", error=e)

    async def repo_search(query):
        async with sem:
            source = f"GitHub search:{query}"
            try:
                r = await client.get("https://api.github.com/search/repositories", params={"q": query, "sort": "updated", "order": "desc", "per_page": 100}, headers=headers)
                r.raise_for_status()
                rows = r.json().get("items", [])
                new = 0
                for repo in rows:
                    url = repo.get("html_url") or ""
                    name = repo.get("full_name") or ""
                    if not url or not name:
                        continue
                    summary = repo.get("description") or ""
                    md = {"source_type": "github", "kind": "repository", "language": repo.get("language"),
                          "stars": repo.get("stargazers_count"), "forks": repo.get("forks_count"), "query": query}
                    if add_external_item("GitHub Repository Search", f"Repository updated: {name}", url, repo.get("updated_at"), summary, md):
                        new += 1
                return _result(source, "github", len(rows), new)
            except Exception as e:
                return _result(source, "github", error=e)

    tasks += [events(p) for p in range(1, max(1, GITHUB_EVENT_PAGES) + 1)]
    tasks += [repo_search(q) for q in GITHUB_SEARCH_QUERIES]
    return await asyncio.gather(*tasks)


async def _collect_x(client, sem):
    if not X_ENABLED:
        return [_result("X", "x", skipped="disabled")]
    if not X_BEARER_TOKEN:
        return [_result("X", "x", skipped="X_BEARER_TOKEN not configured")]
    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    results = []
    async def one(query):
        async with sem:
            source = f"X:{query}"
            try:
                params = {"query": f"({query}) -is:retweet", "max_results": 100,
                          "tweet.fields": "created_at,lang,author_id,public_metrics"}
                r = await client.get("https://api.x.com/2/tweets/search/recent", params=params, headers=headers)
                r.raise_for_status()
                posts = r.json().get("data", [])
                new = 0
                for p in posts:
                    pid, text = p.get("id"), (p.get("text") or "").strip()
                    if not pid or not text:
                        continue
                    md = {"source_type": "x", "language": p.get("lang"), "author_id": p.get("author_id"),
                          "metrics": p.get("public_metrics"), "query": query}
                    if add_external_item("X", text[:240], f"https://x.com/i/web/status/{pid}", p.get("created_at"), text, md):
                        new += 1
                return _result(source, "x", len(posts), new)
            except Exception as e:
                return _result(source, "x", error=e)
    return await asyncio.gather(*(one(q) for q in X_QUERIES))


async def _collect_bluesky(client, sem):
    if not BLUESKY_ENABLED:
        return [_result("Bluesky", "bluesky", skipped="disabled")]
    async def one(query):
        async with sem:
            source = f"Bluesky:{query}"
            try:
                r = await client.get("https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts", params={"q": query, "limit": 100, "sort": "latest"})
                r.raise_for_status()
                posts = r.json().get("posts", [])
                new = 0
                for p in posts:
                    record = p.get("record") or {}
                    text = (record.get("text") or "").strip()
                    uri = p.get("uri") or ""
                    author = (p.get("author") or {}).get("handle") or "unknown"
                    if not text or not uri:
                        continue
                    rkey = uri.rsplit("/", 1)[-1]
                    url = f"https://bsky.app/profile/{author}/post/{rkey}"
                    md = {"source_type": "bluesky", "author": author, "query": query,
                          "like_count": p.get("likeCount"), "repost_count": p.get("repostCount")}
                    if add_external_item("Bluesky", text[:240], url, record.get("createdAt") or p.get("indexedAt"), text, md):
                        new += 1
                return _result(source, "bluesky", len(posts), new)
            except Exception as e:
                return _result(source, "bluesky", error=e)
    return await asyncio.gather(*(one(q) for q in SOCIAL_QUERIES))


async def _collect_reddit(client, sem):
    if not REDDIT_ENABLED:
        return [_result("Reddit", "reddit", skipped="disabled")]
    async def one(query):
        url = f"https://www.reddit.com/search.rss?q={quote(query)}&sort=new&t=day"
        return await _ingest_feed(client, f"Reddit:{query}", url, {"source_type": "reddit", "query": query}, sem)
    return await asyncio.gather(*(one(q) for q in SOCIAL_QUERIES))


async def _collect_mastodon(client, sem):
    if not MASTODON_ENABLED:
        return [_result("Mastodon", "mastodon", skipped="disabled")]
    tasks = []
    for instance in MASTODON_INSTANCES:
        for query in SOCIAL_QUERIES:
            tag = re.sub(r"[^A-Za-z0-9_]", "", query.replace(" ", ""))
            if not tag:
                continue
            url = f"{instance}/tags/{quote(tag)}.rss"
            tasks.append(_ingest_feed(client, f"Mastodon:{urlparse(instance).hostname}:{tag}", url,
                                      {"source_type": "mastodon", "query": query, "instance": instance}, sem))
    return await asyncio.gather(*tasks) if tasks else []


def _html_to_text(raw: str) -> str:
    raw = _SCRIPT_RE.sub(" ", raw)
    raw = _TAG_RE.sub(" ", raw)
    return _SPACE_RE.sub(" ", html.unescape(raw)).strip()


async def fetch_document_bodies(client=None, limit=None):
    if not DOCUMENT_FETCH_ENABLED:
        return {"attempted": 0, "ok": 0, "failed": 0, "skipped": "disabled"}
    items = pending_document_items(limit or DOCUMENT_FETCH_BATCH_SIZE)
    if not items:
        return {"attempted": 0, "ok": 0, "failed": 0}
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=20, follow_redirects=True, headers={"User-Agent": UA})
    sem = asyncio.Semaphore(max(2, DOCUMENT_FETCH_CONCURRENCY))

    async def one(item):
        try:
            md = json.loads(item.get("metadata_json") or "{}")
        except Exception:
            md = {}
        if md.get("source_type") in {"x", "bluesky", "reddit", "mastodon", "github"}:
            text = (item.get("summary") or item.get("title") or "")
            save_document(item["id"], item["url"], text, "ok", 200)
            return True
        async with sem:
            try:
                r = await client.get(item["url"])
                ctype = r.headers.get("content-type", "")
                if r.status_code >= 400:
                    save_document(item["id"], str(r.url), "", "error", r.status_code, f"HTTP {r.status_code}")
                    return False
                text = _html_to_text(r.text) if ("html" in ctype or not ctype) else (item.get("summary") or "")
                if len(text) < 80:
                    text = (item.get("summary") or item.get("title") or "") + " " + text
                save_document(item["id"], str(r.url), text[:50000], "ok", r.status_code)
                return True
            except Exception as e:
                save_document(item["id"], item["url"], "", "error", None, str(e)[:300])
                return False

    try:
        done = await asyncio.gather(*(one(i) for i in items))
        return {"attempted": len(items), "ok": sum(1 for x in done if x), "failed": sum(1 for x in done if not x)}
    finally:
        if own_client:
            await client.aclose()


async def collect_global_information():
    run_id = start_collection_run()
    sem = asyncio.Semaphore(max(2, EXTERNAL_CONCURRENCY))
    headers = {"User-Agent": UA, "Accept-Language": "*"}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
        feed_tasks = [
            _ingest_feed(client, source, url, {"source_type": "rss"}, sem)
            for source, url in EXTERNAL_FEEDS
        ]
        feed_tasks += [
            _ingest_feed(client, source, url, {"source_type": "primary"}, sem)
            for source, url in PRIMARY_SOURCE_FEEDS
        ]
        for country, hl, language in GLOBAL_NEWS_LOCALES:
            feed_tasks.append(_ingest_feed(client, f"Google News {country}/{hl}", _google_news_url(country, hl, language),
                                           {"source_type": "news", "country": country, "language": language}, sem))
        feed_results = await asyncio.gather(*feed_tasks)
        groups = await asyncio.gather(
            _collect_gdelt(client, sem), _collect_github(client, sem), _collect_x(client, sem),
            _collect_bluesky(client, sem), _collect_reddit(client, sem), _collect_mastodon(client, sem),
        )
        document_stats = await fetch_document_bodies(client, DOCUMENT_FETCH_BATCH_SIZE)

    all_results = list(feed_results)
    for group in groups:
        all_results.extend(group)
    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "sources": len(all_results),
        "fetched": sum(int(r.get("fetched", 0)) for r in all_results),
        "new": sum(int(r.get("new", 0)) for r in all_results),
        "errors": [r for r in all_results if r.get("error")],
        "skipped": [r for r in all_results if r.get("skipped")],
        "documents": document_stats,
        "details": all_results,
    }
    finish_collection_run(run_id, result["fetched"], result["new"], len(result["errors"]), len(result["skipped"]), all_results)
    return result


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return []
    return []


def _adaptive_factcheck_limit(requested=None):
    if requested:
        return min(max(1, int(requested)), FACTCHECK_MAX_BATCH_SIZE)
    backlog = queue_metrics().get("factcheck_backlog", 0)
    if backlog > 5000:
        return FACTCHECK_MAX_BATCH_SIZE
    if backlog > 1500:
        return min(60, FACTCHECK_MAX_BATCH_SIZE)
    if backlog > 500:
        return min(45, FACTCHECK_MAX_BATCH_SIZE)
    return min(FACTCHECK_BATCH_SIZE, FACTCHECK_MAX_BATCH_SIZE)


async def factcheck_batch(limit=None):
    batch_limit = _adaptive_factcheck_limit(limit)
    items = unprocessed_external_items(batch_limit)
    if not items:
        return {"processed": 0, "claims": 0, "backlog": 0}
    compact = []
    for i in items:
        doc = document_for_item(i["id"])
        compact.append({
            "id": i["id"], "source": i["source"], "url": i["url"], "title": i["title"],
            "summary": (i.get("summary") or "")[:700],
            "body": ((doc or {}).get("body_text") or "")[:1400],
        })
    prompt = """You normalize multilingual evidence for a fact-check engine. Return ONE strict JSON array and nothing else.
For every input object return: {id, claim_key, canonical_claim, stance, language, checkability}.
Rules:
- canonical_claim: concise factual proposition in English; preserve dates, numbers, entities and uncertainty.
- claim_key: stable lowercase snake_case identity for the SAME underlying real-world proposition across languages and publishers.
- stance: supports or contradicts the affirmative canonical proposition.
- checkability: factual, event, opinion, or unclear.
- Never decide truth from one source. Never invent missing facts.
- If social text is rumor/speculation, preserve that uncertainty in canonical_claim.
Evidence items:\n""" + json.dumps(compact, ensure_ascii=False)
    try:
        raw = await chat([
            {"role": "system", "content": "You are a multilingual evidence normalizer. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ], temperature=0.05, num_predict=2600)
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
        doc = document_for_item(item["id"])
        final_url = (doc or {}).get("final_url") or item["url"]
        domain = md.get("domain") or _domain(final_url)
        claim_id = upsert_claim(key, canonical, {
            "language": n.get("language"), "first_source": item["source"],
            "checkability": n.get("checkability"),
        })
        add_claim_evidence(claim_id, item["id"], domain, source_type, stance, _credibility(domain, source_type))
        claims += 1
    return {"processed": len(items), "claims": claims, "backlog": queue_metrics().get("factcheck_backlog", 0), "batch_size": batch_limit}


async def global_collection_loop(interval_minutes: int):
    while True:
        try:
            r = await collect_global_information()
            print(f"[GLOBAL] sources={r['sources']} fetched={r['fetched']} new={r['new']} errors={len(r['errors'])} skipped={len(r['skipped'])} docs={r['documents'].get('ok',0)}")
        except Exception as e:
            print(f"[GLOBAL] failed: {e}")
        await asyncio.sleep(max(5, interval_minutes) * 60)


async def factcheck_loop(interval_seconds: int):
    cycles = 0
    while True:
        try:
            r = await factcheck_batch()
            if r["processed"]:
                print(f"[FACTCHECK] processed={r['processed']} claims={r['claims']} backlog={r['backlog']} batch={r['batch_size']}")
                cycles += 1
                if cycles % 10 == 0:
                    try:
                        from .obsidian_export import export_to_obsidian
                        await asyncio.to_thread(export_to_obsidian)
                    except Exception as e:
                        print(f"[OBSIDIAN] post-factcheck export failed: {e}")
        except Exception as e:
            print(f"[FACTCHECK] failed: {e}")
        backlog = queue_metrics().get("factcheck_backlog", 0)
        delay = 1 if backlog > 1000 else max(3, interval_seconds)
        await asyncio.sleep(delay)


def intelligence_status():
    claims = recent_claims(100)
    counts = {}
    for c in claims:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    metrics = queue_metrics()
    return {
        "v1": True,
        "recent_claims": len(claims),
        "statuses": counts,
        "queues": metrics,
        "x_configured": bool(X_BEARER_TOKEN),
        "github_authenticated": bool(GITHUB_TOKEN),
        "collectors": {
            "google_news": True, "gdelt": GDELT_ENABLED, "github": GITHUB_ENABLED,
            "x": X_ENABLED and bool(X_BEARER_TOKEN), "bluesky": BLUESKY_ENABLED,
            "reddit": REDDIT_ENABLED, "mastodon": MASTODON_ENABLED,
            "primary_feeds": True, "document_fetch": DOCUMENT_FETCH_ENABLED,
        },
        "source_health": source_health(20),
    }
