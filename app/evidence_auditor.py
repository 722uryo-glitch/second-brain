import json
import re

from .ollama_client import chat


def _extract_json(text: str):
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(cleaned[start:end + 1])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _pack_text(packs, max_chars=28000):
    chunks = []
    remaining = max_chars
    for idx, pack in enumerate(packs or [], 1):
        text = str(pack.get("context") or "")
        part = text[: max(0, min(len(text), remaining))]
        chunks.append(f"--- PACK {idx}: {pack.get('query') or ''} ---\n{part}")
        remaining -= len(part)
        if remaining <= 0:
            break
    return "\n".join(chunks)


async def assess_research_coverage(user_request: str, plan: dict, packs: list, refs: list):
    """Decide whether collected evidence covers the actual decision facets.

    This is semantic sufficiency, not merely 'we found N links'. It returns
    targeted follow-up queries when a facet is still unsupported.
    """
    if not plan.get("needs_research"):
        return {
            "sufficient": True,
            "missing_facets": [],
            "followup_queries": [],
            "supported_conclusions": [],
            "confidence": 1.0,
        }

    evidence = _pack_text(packs)
    prompt = f"""Assess whether the Second Brain has enough evidence to answer the user's research request.
Return one strict JSON object only.

USER REQUEST:
{user_request}

EXECUTION PLAN:
{json.dumps(plan, ensure_ascii=False)}

EVIDENCE PACKS:
{evidence}

Return:
{{
  "sufficient": true|false,
  "missing_facets": ["specific missing evidence"],
  "followup_queries": ["precise search query"],
  "supported_conclusions": ["conclusion supported by source ids like [S1]"],
  "confidence": 0.0
}}

Rules:
- Link count alone is NOT sufficiency.
- Check every research-dependent success criterion and subquestion.
- For market/niche tasks, separately require evidence for demand, supply/competition, and monetization when the user asks for them.
- Evidence of labor shortage is NOT evidence of low SEO/content competition.
- Evidence of popularity is NOT evidence of low supply.
- Prefer multiple independent sources for important conclusions.
- Never use unstated general knowledge to mark a facet as covered.
- If missing, generate at most 3 highly targeted follow-up queries that could fill the exact gaps.
- Use source ids only when they actually appear in the evidence.
"""
    try:
        raw = await chat(
            [
                {"role": "system", "content": "You are the evidence-gap controller of a persistent research agent. Be skeptical. JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            num_predict=700,
            route="verify",
        )
        result = _extract_json(raw)
    except Exception as e:
        return {
            "sufficient": False,
            "missing_facets": ["evidence_assessor_unavailable"],
            "followup_queries": [],
            "supported_conclusions": [],
            "confidence": 0.0,
            "error": str(e),
        }

    if not result:
        return {
            "sufficient": False,
            "missing_facets": ["evidence_assessor_parse_failed"],
            "followup_queries": [],
            "supported_conclusions": [],
            "confidence": 0.0,
        }

    followups = []
    seen = set()
    for q in result.get("followup_queries", []):
        q = str(q or "").strip()
        key = q.lower()
        if len(q) < 3 or key in seen:
            continue
        seen.add(key)
        followups.append(q[:500])
        if len(followups) >= 3:
            break

    missing = [str(x)[:500] for x in result.get("missing_facets", []) if str(x).strip()][:8]
    supported = [str(x)[:700] for x in result.get("supported_conclusions", []) if str(x).strip()][:8]
    try:
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    except Exception:
        confidence = 0.0

    return {
        "sufficient": bool(result.get("sufficient", False)) and not missing,
        "missing_facets": missing,
        "followup_queries": followups,
        "supported_conclusions": supported,
        "confidence": confidence,
    }
