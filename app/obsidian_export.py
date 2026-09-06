import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import OBSIDIAN_VAULT_PATH, OBSIDIAN_EXPORT_LIMIT
from .db import recent_claims, recent_external_items, recent_memories

_BAD_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(text: str, fallback: str) -> str:
    name = _BAD_FILENAME.sub(" ", text or "")
    name = " ".join(name.split()).strip(" .")
    if not name:
        name = fallback
    return name[:120]


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
    """Export curated Second Brain data as Markdown notes for Obsidian.

    SQLite remains the source of truth. The vault is a human-readable projection.
    Raw collection is intentionally not exported one-file-per-item.
    """
    root = _vault()
    now = datetime.now(timezone.utc)
    generated = now.isoformat()

    claims = recent_claims(OBSIDIAN_EXPORT_LIMIT)
    external = recent_external_items(min(OBSIDIAN_EXPORT_LIMIT, 1000))
    memories = recent_memories(min(OBSIDIAN_EXPORT_LIMIT, 500))

    claim_dirs = {
        "corroborated": root / "Verified Claims",
        "partially_corroborated": root / "Partially Verified Claims",
        "disputed": root / "Disputed Claims",
        "unverified": root / "Unverified Claims",
    }

    for claim in claims:
        status = claim.get("status") or "unverified"
        folder = claim_dirs.get(status, claim_dirs["unverified"])
        claim_id = int(claim.get("id") or 0)
        title = str(claim.get("claim_text") or f"Claim {claim_id}")
        filename = f"{claim_id:08d} - {_safe_filename(title, f'Claim {claim_id}')}.md"
        metadata = claim.get("metadata_json") or "{}"
        body = (
            "---\n"
            f"type: claim\nstatus: {_yaml_text(status)}\n"
            f"confidence: {float(claim.get('confidence') or 0):.3f}\n"
            f"independent_sources: {int(claim.get('independent_sources') or 0)}\n"
            f"evidence_count: {int(claim.get('evidence_count') or 0)}\n"
            f"contradictions: {int(claim.get('contradictions') or 0)}\n"
            f"first_seen: {_yaml_text(str(claim.get('first_seen') or ''))}\n"
            f"updated_at: {_yaml_text(str(claim.get('updated_at') or ''))}\n"
            "---\n\n"
            f"# {title}\n\n"
            f"**Status:** {status}\n\n"
            f"**Confidence:** {float(claim.get('confidence') or 0):.1%}\n\n"
            f"**Independent sources:** {int(claim.get('independent_sources') or 0)}\n\n"
            f"**Evidence:** {int(claim.get('evidence_count') or 0)}\n\n"
            f"**Contradictions:** {int(claim.get('contradictions') or 0)}\n\n"
            "## Machine metadata\n\n"
            f"```json\n{metadata}\n```\n"
        )
        _write(folder / filename, body)

    # One rolling external inbox note avoids creating thousands of noisy notes.
    inbox_lines = [
        "---",
        "type: external-inbox",
        f"generated_at: {_yaml_text(generated)}",
        "---",
        "",
        "# External Intelligence Inbox",
        "",
        "Latest collected information. SQLite keeps the full raw archive.",
        "",
    ]
    for item in external:
        title = str(item.get("title") or "Untitled")
        source = str(item.get("source") or "unknown")
        url = str(item.get("url") or "")
        published = str(item.get("published_at") or item.get("collected_at") or "")
        summary = str(item.get("summary") or "").strip()
        inbox_lines += [f"## {title}", "", f"- Source: {source}", f"- Time: {published}"]
        if url:
            inbox_lines.append(f"- URL: {url}")
        if summary:
            inbox_lines += ["", summary]
        inbox_lines.append("")
    _write(root / "External Inbox" / "Latest.md", "\n".join(inbox_lines))

    memory_lines = [
        "---",
        "type: second-brain-memory-index",
        f"generated_at: {_yaml_text(generated)}",
        "---",
        "",
        "# Second Brain Memory Index",
        "",
    ]
    for memory in memories:
        memory_lines.append(
            f"- **{memory.get('kind','note')}** · {memory.get('source','unknown')} · "
            f"{memory.get('created_at','')} — {str(memory.get('content','')).replace(chr(10), ' ')[:500]}"
        )
    _write(root / "Memory" / "Latest Memory Index.md", "\n".join(memory_lines) + "\n")

    status_counts = {}
    for claim in claims:
        status = claim.get("status") or "unverified"
        status_counts[status] = status_counts.get(status, 0) + 1

    home = (
        "# Second Brain Knowledge Vault\n\n"
        f"Last export: {generated}\n\n"
        "SQLite is the machine source of truth. This vault contains the human-readable knowledge layer.\n\n"
        "## Current claim status\n\n"
        f"- Corroborated: {status_counts.get('corroborated', 0)}\n"
        f"- Partially corroborated: {status_counts.get('partially_corroborated', 0)}\n"
        f"- Disputed: {status_counts.get('disputed', 0)}\n"
        f"- Unverified: {status_counts.get('unverified', 0)}\n\n"
        "## Main areas\n\n"
        "- [[External Inbox/Latest|External Intelligence Inbox]]\n"
        "- [[Memory/Latest Memory Index|Memory Index]]\n"
        "- Verified Claims/\n"
        "- Partially Verified Claims/\n"
        "- Disputed Claims/\n"
        "- Unverified Claims/\n"
    )
    _write(root / "HOME.md", home)

    return {
        "ok": True,
        "vault": str(root),
        "claims": len(claims),
        "external_items": len(external),
        "memories": len(memories),
        "generated_at": generated,
    }


async def obsidian_export_loop(interval_minutes: int):
    while True:
        try:
            result = await asyncio.to_thread(export_to_obsidian)
            print(
                f"[OBSIDIAN] claims={result['claims']} external={result['external_items']} "
                f"memories={result['memories']} vault={result['vault']}"
            )
        except Exception as exc:
            print(f"[OBSIDIAN] export failed: {exc}")
        await asyncio.sleep(max(5, interval_minutes) * 60)
