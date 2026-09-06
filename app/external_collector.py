import asyncio
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from .config import EXTERNAL_FEEDS, EXTERNAL_ITEMS_PER_FEED
from .db import add_external_item

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    return " ".join(text.split()).strip()


def _first_text(node, names):
    for name in names:
        found = node.find(name)
        if found is not None and found.text:
            return found.text.strip()
    return ""


def _parse_feed(xml_text: str):
    root = ET.fromstring(xml_text)
    items = []

    # RSS 2.x
    for item in root.findall(".//item"):
        title = _first_text(item, ["title"])
        link = _first_text(item, ["link"])
        published = _first_text(item, ["pubDate", "date"])
        summary = _first_text(item, ["description", "summary"])
        if title and link:
            items.append({
                "title": _clean(title),
                "url": link.strip(),
                "published_at": published or None,
                "summary": _clean(summary),
            })

    if items:
        return items

    # Atom
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//atom:entry", ns):
        title = entry.findtext("atom:title", default="", namespaces=ns)
        published = (
            entry.findtext("atom:published", default="", namespaces=ns)
            or entry.findtext("atom:updated", default="", namespaces=ns)
        )
        summary = (
            entry.findtext("atom:summary", default="", namespaces=ns)
            or entry.findtext("atom:content", default="", namespaces=ns)
        )
        link = ""
        for link_node in entry.findall("atom:link", ns):
            href = link_node.attrib.get("href", "").strip()
            rel = link_node.attrib.get("rel", "alternate")
            if href and rel in {"alternate", ""}:
                link = href
                break
        if title and link:
            items.append({
                "title": _clean(title),
                "url": link,
                "published_at": published or None,
                "summary": _clean(summary),
            })
    return items


async def collect_external_information():
    """Collect external information into a deduplicated inbox.

    This intentionally does not promote every headline into long-term memory.
    Collection and memory are separate so noisy feeds cannot pollute user facts.
    """
    stats = {"feeds": 0, "fetched": 0, "new": 0, "errors": []}
    headers = {"User-Agent": "SecondBrain/1.0 (+local personal knowledge agent)"}

    async with httpx.AsyncClient(timeout=25, follow_redirects=True, headers=headers) as client:
        for source, url in EXTERNAL_FEEDS:
            stats["feeds"] += 1
            try:
                response = await client.get(url)
                response.raise_for_status()
                parsed = _parse_feed(response.text)[:EXTERNAL_ITEMS_PER_FEED]
                stats["fetched"] += len(parsed)
                for item in parsed:
                    if add_external_item(
                        source=source,
                        title=item["title"],
                        url=item["url"],
                        published_at=item.get("published_at"),
                        summary=item.get("summary"),
                        metadata={"feed_url": url},
                    ):
                        stats["new"] += 1
            except Exception as exc:
                stats["errors"].append({"source": source, "error": str(exc)[:300]})

    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    return stats


async def external_collection_loop(interval_minutes: int):
    # Collect immediately on startup, then continue periodically.
    while True:
        try:
            result = await collect_external_information()
            print(
                f"[COLLECTOR] feeds={result['feeds']} fetched={result['fetched']} "
                f"new={result['new']} errors={len(result['errors'])}"
            )
        except Exception as exc:
            print(f"[COLLECTOR] failed: {exc}")
        await asyncio.sleep(max(5, interval_minutes) * 60)
