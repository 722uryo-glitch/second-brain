import asyncio
import json
import re
import time
from dataclasses import dataclass

from .config import (
    EXECUTIVE_ENABLED,
    EXECUTIVE_REVIEW_ENABLED,
    EXECUTIVE_MAX_RESEARCH_QUERIES,
    EXECUTIVE_MAX_REVISIONS,
    EXECUTIVE_HISTORY_TURNS,
    EXECUTIVE_SIMPLE_MAX_TOKENS,
    EXECUTIVE_LONG_MAX_TOKENS,
    UNOROUTER_PRIVATE_CHAT,
)
from .db import (
    recent_conversation,
    start_agent_run,
    add_agent_step,
    finish_agent_run,
    recent_agent_runs,
)
from .memory import recall, store_memory
from .memory_consolidator import consolidate_user_turn
from .ollama_client import chat
from .orchestrator import gather_research_context, is_research_task, is_current_task


@dataclass
class ExecutiveResult:
    response: str
    memories: list
    run_id: int | None
    mode: str
    plan: dict
    critique: dict


_GREETING = {
    "こんにちは", "こんばんは", "おはよう", "おはようございます", "やあ", "どうも",
    "hello", "hi", "hey", "こんにちは!", "こんばんは!",
}

_PERSONAL_MARKERS = (
    "私の", "僕の", "俺の", "自分の", "覚えて", "前に話", "前回", "好み", "予定", "タスク",
    "住所", "電話", "メール", "家族", "友達", "学校", "職場", "仕事の", "名前", "誕生日",
)

_LONG_MARKERS = (
    "記事を書", "記事作成", "ブログを書", "レポート", "論文", "台本", "原稿", "企画書", "提案書",
    "完全版", "詳しく", "詳細に", "徹底的", "長文", "article", "blog post", "report", "full draft",
)

_DELEGATION_MARKERS = (
    "任せ", "全部やって", "一気に", "なんでもいい", "おまかせ", "調べて", "探して", "比較して",
    "考えて", "作って", "書いて", "実行して", "進めて",
)

_STRONG_RESEARCH_CLAIM_MARKERS = (
    "需要が", "需要は", "供給が少", "供給不足", "競合が少", "競合は少", "未開拓", "市場規模",
    "不足率", "成長率", "シェア", "最も", "圧倒的", "確実に", "確実な", "急増", "急拡大",
    "demand", "low competition", "undersupplied", "market size", "growth rate",
)


def _now_ms():
    return int(time.perf_counter() * 1000)


def _elapsed_ms(start_ms):
    return max(0, _now_ms() - start_ms)


def _extract_json_object(text: str):
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(cleaned[start:end + 1])
    except Exception:
        return {}


def _history(limit=None):
    rows = recent_conversation(limit or EXECUTIVE_HISTORY_TURNS)
    result = []
    for row in rows:
        role = "user" if row.get("source") == "user" else "assistant"
        content = str(row.get("content") or "").strip()
        if content:
            result.append({"role": role, "content": content[:2200]})
    return result


def _looks_personal(text: str):
    t = text.lower()
    return any(marker.lower() in t for marker in _PERSONAL_MARKERS)


def _looks_long(text: str):
    t = text.lower()
    return any(marker.lower() in t for marker in _LONG_MARKERS)


def _looks_delegated(text: str):
    t = text.lower()
    return any(marker.lower() in t for marker in _DELEGATION_MARKERS)


def _is_greeting(text: str):
    return text.strip().lower().replace("！", "!") in _GREETING


def _fallback_plan(user_text: str):
    personal = _looks_personal(user_text)
    research = is_research_task(user_text) or is_current_task(user_text)
    long_form = _looks_long(user_text)
    delegated = _looks_delegated(user_text)
    complexity = "simple"
    if research or long_form or delegated or len(user_text) > 180:
        complexity = "complex"
    return {
        "goal": user_text.strip()[:500],
        "mode": "personal" if personal else ("research" if research else "work"),
        "complexity": complexity,
        "needs_research": bool(research),
        "needs_memory": bool(personal),
        "subquestions": [user_text.strip()] if research else [],
        "success_criteria": ["ユーザーの依頼に直接答える", "不要な確認質問をしない"],
        "deliverable": "complete answer",
        "long_form": bool(long_form),
    }


async def _make_plan(user_text: str, history: list):
    fallback = _fallback_plan(user_text)
    if fallback["complexity"] == "simple":
        return fallback

    prompt = f"""Create a compact execution plan for a persistent Second Brain.
Return one strict JSON object only.

User request:
{user_text}

Return exactly these fields:
{{
  "goal": "...",
  "mode": "research|work|personal",
  "complexity": "simple|complex",
  "needs_research": true|false,
  "needs_memory": true|false,
  "subquestions": ["..."],
  "success_criteria": ["..."],
  "deliverable": "...",
  "long_form": true|false
}}

Rules:
- If the user delegated choices, do not plan a clarification question; choose and proceed.
- Research/current/market/comparison requests need research.
- Personal preferences, prior decisions, tasks, or autobiographical facts need memory.
- Use at most {EXECUTIVE_MAX_RESEARCH_QUERIES} research subquestions.
- Each subquestion must be independently researchable and useful to the final answer.
- Keep success criteria concrete and testable.
"""
    try:
        raw = await chat(
            [
                {"role": "system", "content": "You are an execution planner. JSON only."},
                *history[-4:],
                {"role": "user", "content": prompt},
            ],
            temperature=0.05,
            num_predict=420,
            route="fast_cloud",
        )
        plan = _extract_json_object(raw)
    except Exception as e:
        print(f"[EXECUTIVE] planner failed: {e}")
        return fallback

    if not plan:
        return fallback

    plan["goal"] = str(plan.get("goal") or fallback["goal"])[:800]
    plan["mode"] = plan.get("mode") if plan.get("mode") in {"research", "work", "personal"} else fallback["mode"]
    plan["complexity"] = plan.get("complexity") if plan.get("complexity") in {"simple", "complex"} else fallback["complexity"]
    plan["needs_research"] = bool(plan.get("needs_research", fallback["needs_research"]))
    plan["needs_memory"] = bool(plan.get("needs_memory", fallback["needs_memory"]))
    plan["long_form"] = bool(plan.get("long_form", fallback["long_form"]))
    plan["deliverable"] = str(plan.get("deliverable") or fallback["deliverable"])[:500]
    subq = [str(x).strip() for x in plan.get("subquestions", []) if str(x).strip()]
    if plan["needs_research"] and not subq:
        subq = [user_text]
    plan["subquestions"] = subq[:max(1, EXECUTIVE_MAX_RESEARCH_QUERIES)]
    criteria = [str(x).strip() for x in plan.get("success_criteria", []) if str(x).strip()]
    plan["success_criteria"] = criteria[:6] or fallback["success_criteria"]
    return plan


async def _gather_memory(user_text: str, needs_memory: bool):
    if not needs_memory:
        return []
    try:
        return await recall(user_text, top_k=8)
    except Exception as e:
        print(f"[EXECUTIVE] memory recall failed: {e}")
        return []


def _memory_context(memories: list):
    if not memories:
        return "RELATED LONG-TERM MEMORY: none"
    lines = ["RELATED LONG-TERM MEMORY:"]
    for m in memories[:8]:
        lines.append(
            f"- [{m.get('kind')}/{m.get('source')}] importance={m.get('importance')} {str(m.get('content') or '')[:700]}"
        )
    return "\n".join(lines)


def _renumber_pack(pack: dict, start_no: int):
    mapping = {}
    new_refs = []
    next_no = start_no
    for ref in pack.get("refs", []):
        old = str(ref.get("ref") or "")
        new = f"S{next_no}"
        next_no += 1
        if old:
            mapping[old] = new
        copied = dict(ref)
        copied["ref"] = new
        new_refs.append(copied)

    context = str(pack.get("context") or "")
    if mapping:
        context = re.sub(r"\[(S\d+)\]", lambda m: f"[{mapping.get(m.group(1), m.group(1))}]", context)
    return {**pack, "context": context, "refs": new_refs}, next_no


async def _gather_research(plan: dict, user_text: str):
    if not plan.get("needs_research"):
        return [], [], True
    questions = plan.get("subquestions") or [user_text]
    questions = questions[:max(1, EXECUTIVE_MAX_RESEARCH_QUERIES)]

    async def one(q):
        try:
            context, refs, enough = await gather_research_context(q, on_demand=True)
            return {"query": q, "context": context, "refs": refs, "enough": enough}
        except Exception as e:
            return {"query": q, "context": f"RESEARCH ERROR: {e}", "refs": [], "enough": False}

    raw_packs = await asyncio.gather(*(one(q) for q in questions))
    packs = []
    all_refs = []
    next_no = 1
    seen_urls = set()
    for raw_pack in raw_packs:
        pack, next_no = _renumber_pack(raw_pack, next_no)
        filtered_refs = []
        for ref in pack["refs"]:
            url = ref.get("url") or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            filtered_refs.append(ref)
            all_refs.append(ref)
        pack["refs"] = filtered_refs
        packs.append(pack)
    enough = bool(packs) and all(p.get("enough") for p in packs)
    return packs, all_refs, enough


def _research_context(packs: list):
    if not packs:
        return "SECOND BRAIN RESEARCH: not required"
    chunks = ["SECOND BRAIN RESEARCH PACKS:"]
    for idx, pack in enumerate(packs, 1):
        chunks.append(f"\n--- RESEARCH {idx}: {pack['query']} ---\n{pack['context'][:14000]}")
    return "\n".join(chunks)


def _answer_system(plan: dict, research_enough: bool, personal: bool):
    evidence_rule = (
        "Research evidence is sufficient enough to synthesize, but every researched factual or market claim still requires an inline [S#] citation."
        if research_enough else
        "Research evidence is incomplete. Explicitly label unsupported conclusions as hypotheses and never invent demand, supply, competition, prices, statistics, or current facts."
    )
    privacy = (
        "The memory block contains user-specific context. Treat it as authoritative when relevant."
        if personal else
        "Do not invent personal facts about the user."
    )
    return (
        "You are the execution layer of a persistent Second Brain. Your job is to finish the user's task, not merely discuss it. "
        "Follow the execution plan, use the supplied memory/research before general knowledge, and do not ask questions already answered in recent dialogue. "
        "If the user delegated a choice, choose the strongest option and continue. "
        "Every concrete number, percentage, date-sensitive fact, market-size statement, demand claim, low-competition claim, and supply-gap claim derived from research MUST have an inline source marker [S#] in the same sentence or immediately after it. "
        "Never convert a weak inference into a fact. If the evidence cannot prove low competition or low supply, say that explicitly. "
        f"{evidence_rule} {privacy} "
        "Return a complete, coherent answer in the user's language. For long-form work, do not stop mid-section or mid-sentence."
    )


async def _draft(user_text: str, history: list, plan: dict, memories: list, research_packs: list, research_enough: bool):
    personal = bool(plan.get("needs_memory") or plan.get("mode") == "personal")
    messages = [
        {"role": "system", "content": _answer_system(plan, research_enough, personal)},
        {"role": "system", "content": "EXECUTION PLAN:\n" + json.dumps(plan, ensure_ascii=False)},
        {"role": "system", "content": _memory_context(memories)},
        {"role": "system", "content": _research_context(research_packs)},
        *history,
        {"role": "user", "content": user_text},
    ]
    route = "local" if personal and not UNOROUTER_PRIVATE_CHAT else "fast_cloud"
    tokens = EXECUTIVE_LONG_MAX_TOKENS if plan.get("long_form") else EXECUTIVE_SIMPLE_MAX_TOKENS
    return await chat(messages, temperature=0.18, num_predict=tokens, route=route)


def _deterministic_research_audit(draft: str, refs: list, needs_research: bool):
    if not needs_research:
        return []
    valid = {str(r.get("ref")) for r in refs if r.get("ref")}
    issues = []
    cited = set(re.findall(r"\[(S\d+)\]", draft or ""))
    if refs and not (cited & valid):
        issues.append("research_answer_has_no_valid_inline_citations")

    for raw in re.split(r"(?<=[。！？!?\n])", draft or ""):
        sentence = raw.strip()
        if not sentence:
            continue
        has_number = bool(re.search(r"\d(?:[\d,.]*\d)?\s*(?:%|％|万人|億円|兆円|人|件|年|倍)", sentence))
        strong_claim = any(marker.lower() in sentence.lower() for marker in _STRONG_RESEARCH_CLAIM_MARKERS)
        if not (has_number or strong_claim):
            continue
        markers = set(re.findall(r"\[(S\d+)\]", sentence))
        if not (markers & valid):
            issues.append("uncited_material_claim: " + sentence[:180])
        if len(issues) >= 8:
            break
    return issues


async def _critique(user_text: str, plan: dict, draft: str, research_packs: list, research_enough: bool, refs: list):
    deterministic = _deterministic_research_audit(draft, refs, bool(plan.get("needs_research")))
    if deterministic:
        return {"pass": False, "issues": deterministic, "missing": [], "confidence": 1.0, "deterministic": True}

    if not EXECUTIVE_REVIEW_ENABLED or plan.get("complexity") != "complex":
        return {"pass": True, "issues": [], "missing": [], "confidence": 1.0, "skipped": True}

    evidence_summary = _research_context(research_packs)[:12000]
    prompt = f"""Audit the proposed answer against the user's request and the execution plan.
Return one strict JSON object only:
{{"pass": true|false, "issues": ["..."], "missing": ["..."], "confidence": 0.0}}

User request:
{user_text}

Plan:
{json.dumps(plan, ensure_ascii=False)}

Research sufficient: {research_enough}
Evidence:
{evidence_summary}

Draft:
{draft[:16000]}

Fail the draft if any of these occur:
- it ignores an explicit part of the request;
- it asks an unnecessary clarification after the user delegated choices;
- it presents unsupported demand/supply/competition/current-fact claims as proven;
- it gives a concrete statistic or percentage without a matching source marker;
- it contradicts itself;
- it stops before completing the requested deliverable;
- it claims to have researched evidence that is not in the evidence pack.
Do not fail merely for style preferences.
"""
    try:
        raw = await chat(
            [
                {"role": "system", "content": "You are a strict answer auditor. JSON only. When uncertain, fail closed."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            num_predict=420,
            route="cloud",
        )
        result = _extract_json_object(raw)
    except Exception as e:
        print(f"[EXECUTIVE] critic failed: {e}")
        return {"pass": False, "issues": ["critic_unavailable"], "missing": [], "confidence": 0.0}

    if not result:
        return {"pass": False, "issues": ["critic_parse_failed"], "missing": [], "confidence": 0.0}
    result["pass"] = bool(result.get("pass", False))
    result["issues"] = [str(x)[:400] for x in result.get("issues", [])][:8]
    result["missing"] = [str(x)[:400] for x in result.get("missing", [])][:8]
    try:
        result["confidence"] = float(result.get("confidence", 0.0))
    except Exception:
        result["confidence"] = 0.0
    return result


async def _revise(user_text: str, plan: dict, draft: str, critique: dict, memories: list, research_packs: list, research_enough: bool):
    personal = bool(plan.get("needs_memory") or plan.get("mode") == "personal")
    prompt = f"""Revise the draft so it fully satisfies the user's request.

User request:
{user_text}

Plan:
{json.dumps(plan, ensure_ascii=False)}

Critique:
{json.dumps(critique, ensure_ascii=False)}

Original draft:
{draft}

Rules:
- Fix every substantive critique issue.
- Remove any number, percentage, market-size, demand, supply-gap, or low-competition claim that cannot be cited from the supplied research.
- Every retained material research claim must carry an inline [S#] citation.
- If research is insufficient, label uncertain conclusions as hypotheses instead of pretending certainty.
- Complete the deliverable in this response.
- Answer in the user's language.
"""
    messages = [
        {"role": "system", "content": _answer_system(plan, research_enough, personal)},
        {"role": "system", "content": _memory_context(memories)},
        {"role": "system", "content": _research_context(research_packs)},
        {"role": "user", "content": prompt},
    ]
    route = "local" if personal and not UNOROUTER_PRIVATE_CHAT else "fast_cloud"
    tokens = EXECUTIVE_LONG_MAX_TOKENS if plan.get("long_form") else EXECUTIVE_SIMPLE_MAX_TOKENS
    return await chat(messages, temperature=0.08, num_predict=tokens, route=route)


def _append_sources(response: str, refs: list):
    if not refs:
        return response
    used = []
    for ref in refs:
        marker = f"[{ref.get('ref')}]"
        if marker in response:
            used.append(ref)
    if not used:
        return response
    lines = []
    seen = set()
    for ref in used[:8]:
        url = ref.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        lines.append(f"[{ref.get('ref')}] {ref.get('source') or ''} — {ref.get('title') or ''}\n{url}")
    return response if not lines else response + "\n\n参照した情報源:\n" + "\n".join(lines)


def _safe_research_failure(user_text: str, critique: dict):
    issues = critique.get("issues") or []
    detail = " / ".join(str(x)[:140] for x in issues[:3])
    return (
        "第2の脳で調査・生成・監査まで実行しましたが、根拠監査を通過できなかったため、記事をそのまま出すのを止めました。\n\n"
        "現時点では、需要・供給不足・競合の少なさ・市場規模などを同時に裏付ける証拠が足りないか、生成文に出典のない断定が残っています。"
        "ここでそれっぽい数字や『未開拓』『競合が少ない』を作るより、未証明として止めるのが正しい挙動です。"
        + (f"\n\n監査で残った問題: {detail}" if detail else "")
    )


async def _store_conversation_later(user_text: str, response: str, run_id=None):
    try:
        metadata = {"agent_run_id": run_id} if run_id else None
        await store_memory("conversation", "user", user_text, 0.55, metadata)
        await store_memory("conversation", "assistant", response, 0.20, metadata)
    except Exception as e:
        print(f"[EXECUTIVE] memory save failed: {e}")
    try:
        await consolidate_user_turn(user_text, run_id)
    except Exception as e:
        print(f"[EXECUTIVE] consolidation failed: {e}")


async def run(user_text: str):
    if not EXECUTIVE_ENABLED:
        response = await chat([{"role": "user", "content": user_text}], temperature=0.25, num_predict=EXECUTIVE_SIMPLE_MAX_TOKENS, route="fast_cloud")
        asyncio.create_task(_store_conversation_later(user_text, response))
        return ExecutiveResult(response, [], None, "disabled", {}, {})

    history = _history()
    if _is_greeting(user_text):
        response = await chat(
            [{"role": "system", "content": "Reply naturally and briefly in the user's language."}, *history[-4:], {"role": "user", "content": user_text}],
            num_predict=100,
            route="local",
        )
        asyncio.create_task(_store_conversation_later(user_text, response))
        return ExecutiveResult(response, [], None, "greeting", {}, {})

    plan_start = _now_ms()
    plan = await _make_plan(user_text, history)
    mode = plan.get("mode", "work")
    run_id = start_agent_run(user_text, plan.get("goal", ""), mode, plan)
    step_no = 1
    add_agent_step(run_id, step_no, "plan", "Create execution plan", output_data=plan, duration_ms=_elapsed_ms(plan_start))
    step_no += 1

    try:
        memory_start = _now_ms()
        memories = await _gather_memory(user_text, bool(plan.get("needs_memory")))
        add_agent_step(run_id, step_no, "memory", "Retrieve relevant long-term memory", output_data={"count": len(memories), "ids": [m.get("id") for m in memories[:8]]}, duration_ms=_elapsed_ms(memory_start))
        step_no += 1

        research_start = _now_ms()
        research_packs, refs, research_enough = await _gather_research(plan, user_text)
        add_agent_step(run_id, step_no, "research", "Retrieve and research external evidence", input_data={"queries": plan.get("subquestions", [])}, output_data={"packs": len(research_packs), "references": len(refs), "sufficient": research_enough}, duration_ms=_elapsed_ms(research_start))
        step_no += 1

        draft_start = _now_ms()
        response = await _draft(user_text, history, plan, memories, research_packs, research_enough)
        add_agent_step(run_id, step_no, "draft", "Produce answer or deliverable", output_data={"characters": len(response)}, duration_ms=_elapsed_ms(draft_start))
        step_no += 1

        review_start = _now_ms()
        critique = await _critique(user_text, plan, response, research_packs, research_enough, refs)
        add_agent_step(run_id, step_no, "review", "Audit answer against goal and evidence", output_data=critique, duration_ms=_elapsed_ms(review_start))
        step_no += 1

        revisions = 0
        while not critique.get("pass", True) and revisions < max(0, EXECUTIVE_MAX_REVISIONS):
            revise_start = _now_ms()
            response = await _revise(user_text, plan, response, critique, memories, research_packs, research_enough)
            revisions += 1
            add_agent_step(run_id, step_no, "revise", "Revise failed draft", input_data={"issues": critique.get("issues", []), "missing": critique.get("missing", [])}, output_data={"characters": len(response), "revision": revisions}, duration_ms=_elapsed_ms(revise_start))
            step_no += 1
            critique = await _critique(user_text, plan, response, research_packs, research_enough, refs)

        if plan.get("needs_research") and not critique.get("pass", False):
            response = _safe_research_failure(user_text, critique)
        else:
            response = _append_sources(response, refs)

        finish_agent_run(
            run_id,
            "completed",
            final_response=response,
            critique=critique,
            metadata={"research_sufficient": research_enough, "references": len(refs), "revisions": revisions, "memory_count": len(memories)},
        )
        asyncio.create_task(_store_conversation_later(user_text, response, run_id))
        return ExecutiveResult(response, memories, run_id, mode, plan, critique)
    except Exception as e:
        add_agent_step(run_id, step_no, "error", "Executive failure", status="error", error=e)
        finish_agent_run(run_id, "failed", error=e, metadata={"mode": mode})
        raise


def status():
    runs = recent_agent_runs(10, include_steps=False)
    completed = sum(1 for r in runs if r.get("status") == "completed")
    failed = sum(1 for r in runs if r.get("status") == "failed")
    return {
        "enabled": EXECUTIVE_ENABLED,
        "review_enabled": EXECUTIVE_REVIEW_ENABLED,
        "max_research_queries": EXECUTIVE_MAX_RESEARCH_QUERIES,
        "max_revisions": EXECUTIVE_MAX_REVISIONS,
        "recent_runs": len(runs),
        "recent_completed": completed,
        "recent_failed": failed,
        "latest_run_id": runs[0]["id"] if runs else None,
    }
