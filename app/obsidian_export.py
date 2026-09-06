import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    OBSIDIAN_VAULT_PATH, OBSIDIAN_MAX_CLAIMS,
    OBSIDIAN_MAX_EXTERNAL, OBSIDIAN_MAX_MEMORIES,
)
from .db import recent_claims, recent_external_items, recent_memories
from .v1_storage import queue_metrics, source_health

_BAD_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(text: str, fallback: str) -> str:
    name = _BAD_FILENAME.sub(" ", text or "")
    name = " ".join(name.split()).strip(" .")
    return (name or fallback)[:120]


def _yaml_text(value: str) -> str:
    return json.dumps(value or "", ensure_ascii=False)


def _vault() -> Path:
    path = Path(OBSIDIAN_VAULT_PATH).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_to_obsidian():
    from .knowledge import init_knowledge
    from .knowledge_vault import export_knowledge
    init_knowledge()
    export_knowledge()
    root = _vault()
    generated = datetime.now(timezone.utc).isoformat()
    claims = recent_claims(OBSIDIAN_MAX_CLAIMS)
    external = recent_external_items(OBSIDIAN_MAX_EXTERNAL)
    memories = recent_memories(OBSIDIAN_MAX_MEMORIES)
    metrics = queue_metrics()
    health = source_health(100)

    claim_dirs = {
        "corroborated": root / "Verified Claims",
        "partially_corroborated": root / "Partially Verified Claims",
        "disputed": root / "Disputed Claims",
        "unverified": root / "Unverified Claims",
    }
    for claim in claims:
        status = claim.get("status") or "unverified"
        folder = claim_dirs.get(status, claim_dirs["unverified"])
        cid = int(claim.get("id") or 0)
        title = str(claim.get("claim_text") or f"Claim {cid}")
        body = (
            "---\n"
            "type: claim\n"
            f"status: {_yaml_text(status)}\n"
            f"confidence: {float(claim.get('confidence') or 0):.3f}\n"
            f"independent_sources: {int(claim.get('independent_sources') or 0)}\n"
            f"evidence_count: {int(claim.get('evidence_count') or 0)}\n"
            f"contradictions: {int(claim.get('contradictions') or 0)}\n"
            f"first_seen: {_yaml_text(str(claim.get('first_seen') or ''))}\n"
            f"updated_at: {_yaml_text(str(claim.get('updated_at') or ''))}\n"
            "---\n\n"
            f"# {title}\n\n"
            f"- Status: **{status}**\n"
            f"- Confidence: **{float(claim.get('confidence') or 0):.1%}**\n"
            f"- Independent sources: **{int(claim.get('independent_sources') or 0)}**\n"
            f"- Evidence: **{int(claim.get('evidence_count') or 0)}**\n"
            f"- Contradictions: **{int(claim.get('contradictions') or 0)}**\n\n"
            "## Machine metadata\n\n"
            f"```json\n{claim.get('metadata_json') or '{}'}\n```\n"
        )
        filename = f"{cid:08d} - {_safe_filename(title, f'Claim {cid}')}.md"
        _write(folder / filename, body)

    inbox = ["---", "type: external-inbox", f"generated_at: {_yaml_text(generated)}", "---", "", "# External Intelligence Inbox", ""]
    for item in external:
        title = str(item.get("title") or "Untitled")
        inbox += [f"## {title}", "", f"- Source: {item.get('source','unknown')}", f"- Time: {item.get('published_at') or item.get('collected_at') or ''}"]
        if item.get("url"):
            inbox.append(f"- URL: {item['url']}")
        if item.get("summary"):
            inbox += ["", str(item["summary"]).strip()]
        inbox.append("")
    _write(root / "External Inbox" / "Latest.md", "\n".join(inbox))

    memory_lines = ["---", "type: second-brain-memory-index", f"generated_at: {_yaml_text(generated)}", "---", "", "# Second Brain Memory Index", ""]
    for m in memories:
        content = str(m.get("content", "")).replace("\n", " ")[:500]
        memory_lines.append(f"- **{m.get('kind','note')}** · {m.get('source','unknown')} · {m.get('created_at','')} — {content}")
    _write(root / "Memory" / "Latest Memory Index.md", "\n".join(memory_lines) + "\n")

    status_counts = {}
    for c in claims:
        s = c.get("status") or "unverified"
        status_counts[s] = status_counts.get(s, 0) + 1

    dashboard = [
        "---", "type: intelligence-dashboard", f"generated_at: {_yaml_text(generated)}", "---", "",
        "# V1 Intelligence Dashboard", "",
        f"- Raw external items: **{metrics.get('external_items',0)}**",
        f"- Full documents fetched: **{metrics.get('documents',0)}**",
        f"- Claims: **{metrics.get('claims',0)}**",
        f"- Evidence records: **{metrics.get('evidence',0)}**",
        f"- Fact-check backlog: **{metrics.get('factcheck_backlog',0)}**",
        f"- Failing sources: **{metrics.get('failing_sources',0)}**", "",
        "## Claim status", "",
        f"- Corroborated: {status_counts.get('corroborated',0)}",
        f"- Partially corroborated: {status_counts.get('partially_corroborated',0)}",
        f"- Disputed: {status_counts.get('disputed',0)}",
        f"- Unverified: {status_counts.get('unverified',0)}", "",
        "## Source health", "",
    ]
    for h in health:
        marker = "FAIL" if int(h.get("consecutive_failures") or 0) else "OK"
        dashboard.append(f"- [{marker}] {h.get('source')} · fetched={h.get('fetched_total',0)} · new={h.get('new_total',0)} · failures={h.get('consecutive_failures',0)}")
    _write(root / "Dashboard" / "V1 Intelligence.md", "\n".join(dashboard) + "\n")

    home = (
        "# Second Brain V1 Knowledge Vault\n\n"
        f"Last export: {generated}\n\n"
        "SQLite is the machine source of truth. Obsidian is the human-readable knowledge layer.\n\n"
        "## Open\n\n"
        "- [[Dashboard/V1 Intelligence|V1 Intelligence Dashboard]]\n"
        "- [[External Inbox/Latest|External Intelligence Inbox]]\n"
        "- [[Memory/Latest Memory Index|Memory Index]]\n"
        "- Verified Claims/\n"
        "- Partially Verified Claims/\n"
        "- Disputed Claims/\n"
        "- Unverified Claims/\n"
    )
    _write(root / "HOME.md", home)
    return {"ok": True, "vault": str(root), "claims": len(claims), "external_items": len(external), "memories": len(memories), "generated_at": generated}


async def obsidian_export_loop(interval_minutes: int):
    while True:
        try:
            result = await asyncio.to_thread(export_to_obsidian)
            print(f"[OBSIDIAN] claims={result['claims']} external={result['external_items']} memories={result['memories']} vault={result['vault']}")
        except Exception as exc:
            print(f"[OBSIDIAN] export failed: {exc}")
        await asyncio.sleep(max(5, interval_minutes) * 60)
