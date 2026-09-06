import asyncio
import json
import hashlib
import os
import re
import time
from dataclasses import dataclass

from .job_context import checkpoint, current as job_context, gather_owned


async def _step(key, function, *args, kind="checkpoint"):
    return await checkpoint(key, list(args), lambda: function(*args), kind)


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
from .context_manager import format_conversation_summary, maybe_update_conversation_summary
from .evidence_auditor import assess_research_coverage
from .ollama_client import chat
from .orchestrator import gather_research_context, is_research_task, is_current_task

EXECUTIVE_MAX_RESEARCH_ROUNDS = int(os.getenv("EXECUTIVE_MAX_RESEARCH_ROUNDS", "2"))


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
    "考えて", "作って", "書いて", "実行して", "進めて", "完成させ", "直して",
)

# Only concrete, externally verifiable assertions belong here. Generic headings
# such as '需要はあるが供給が少ないジャンルの例' must not trip the audit.
_STRONG_RESEARCH_ASSERTIONS = (
    "需要が高い", "需要が増", "需要は高", "需要は増", "供給が少ない", "供給不足で", "供給が不足",
    "競合が少ない", "競合は少ない", "競合がほとんど", "未開拓で", "未開拓市場", "市場規模は",
    "不足率は", "成長率は", "シェアは", "圧倒的に", "確実に伸", "急増して", "急拡大して",
    "low competition", "undersupplied", "market size is", "growth rate is", "demand is high",
)


def _conversation_state():
    context = job_context.get()
    if context is not None and hasattr(context, "conversation_snapshot"):
        return context.conversation_snapshot
    return format_conversation_summary()


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
        obj = json.loads(cleaned[start:end + 1])
        return obj if isinstance(obj, dict) else {}
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
    complexity = "complex" if research or long_form or delegated or len(user_text) > 180 else "simple"
    required = []
    if research:
        required = ["claims needed to satisfy the user's requested decision or deliverable"]
    return {
        "goal": user_text.strip()[:700],
        "mode": "personal" if personal else ("research" if research else "work"),
        "complexity": complexity,
        "needs_research": bool(research),
        "needs_memory": bool(personal or "前" in user_text or "続き" in user_text),
        "subquestions": [user_text.strip()] if research else [],
        "required_evidence": required,
        "success_criteria": ["ユーザーの依頼に直接答える", "不要な確認質問をしない", "成果物を最後まで完成させる"],
        "deliverable": "complete answer",
        "long_form": bool(long_form),
    }


async def _make_plan(user_text: str, history: list):
    fallback = _fallback_plan(user_text)
    if fallback["complexity"] == "simple":
        return fallback

    summary = _conversation_state()
    prompt = f"""Create a compact execution plan for a persistent Second Brain.
Return one strict JSON object only.

User request:
{user_text}

Older conversation context:
{summary}

Return exactly these fields:
{{
  "goal": "...",
  "mode": "research|work|personal",
  "complexity": "simple|complex",
  "needs_research": true|false,
  "needs_memory": true|false,
  "subquestions": ["..."],
  "required_evidence": ["..."],
  "success_criteria": ["..."],
  "deliverable": "...",
  "long_form": true|false
}}

Rules:
- If the user delegated choices, do not plan a clarification question; choose and proceed.
- Research/current/market/comparison requests need research.
- If prior preferences, decisions, goals or unfinished work can materially change the answer, set needs_memory=true.
- Decompose research into independently searchable questions; at most {EXECUTIVE_MAX_RESEARCH_QUERIES} initial queries.
- required_evidence must name what would actually prove the requested conclusion. Example: labor shortage does not prove low SEO competition.
- Keep success criteria concrete and testable.
"""
    try:
        raw = await chat(
            [
                {"role": "system", "content": "You are the planning controller of a persistent agent. JSON only."},
                *history[-4:],
                {"role": "user", "content": prompt},
            ],
            temperature=0.03,
            num_predict=650,
            route="fast_cloud",
        )
        plan = _extract_json_object(raw)
    except Exception as e:
        print(f"[EXECUTIVE] planner failed: {e}")
        return fallback

    if not plan:
        return fallback
    plan["goal"] = str(plan.get("goal") or fallback["goal"])[:1000]
    plan["mode"] = plan.get("mode") if plan.get("mode") in {"research", "work", "personal"} else fallback["mode"]
    plan["complexity"] = plan.get("complexity") if plan.get("complexity") in {"simple", "complex"} else fallback["complexity"]
    plan["needs_research"] = bool(plan.get("needs_research", fallback["needs_research"]))
    plan["needs_memory"] = bool(plan.get("needs_memory", fallback["needs_memory"]))
    plan["long_form"] = bool(plan.get("long_form", fallback["long_form"]))
    plan["deliverable"] = str(plan.get("deliverable") or fallback["deliverable"])[:600]
    subq = [str(x).strip() for x in plan.get("subquestions", []) if str(x).strip()]
    if plan["needs_research"] and not subq:
        subq = [user_text]
    plan["subquestions"] = subq[:max(1, EXECUTIVE_MAX_RESEARCH_QUERIES)]
    required = [str(x).strip() for x in plan.get("required_evidence", []) if str(x).strip()]
    plan["required_evidence"] = required[:8] or fallback["required_evidence"]
    criteria = [str(x).strip() for x in plan.get("success_criteria", []) if str(x).strip()]
    plan["success_criteria"] = criteria[:8] or fallback["success_criteria"]
    return plan


async def _gather_memory(user_text: str, needs_memory: bool):
    memories=[]
    if needs_memory:
        try:
            memories=await recall(user_text, top_k=8)
        except Exception as e:
            print(f"[EXECUTIVE] memory recall failed: {e}")
    try:
        from . import knowledge as k
        k.init_knowledge()
        notes=k.related({'id':-1,'title':user_text,'content':user_text,'topics':[]},limit=4)
        for n in notes:
            memories.append({'id':'knowledge:'+str(n['id']),'kind':'knowledge','source':n['source_url'] or n['origin'],
              'importance':0.6,'content':'[資料の解釈・未検証] '+n['content'][:1800]})
    except Exception as e:
        print(f"[EXECUTIVE] knowledge recall failed: {e}")
    return memories


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


async def _research_queries(queries):
    async def one(q):
        try:
            context, refs, enough = await gather_research_context(q, on_demand=True)
            return {"query": q, "context": context, "refs": refs, "enough": enough}
        except Exception as e:
            return {"query": q, "context": f"RESEARCH ERROR: {e}", "refs": [], "enough": False}
    return await gather_owned(*(_step("query:" + hashlib.sha256(q.encode()).hexdigest()[:20], one, q, kind="research_pack") for q in queries)) if queries else []


def _merge_research(raw_packs, packs, all_refs, next_no, seen_urls):
    for raw_pack in raw_packs:
        pack, next_no = _renumber_pack(raw_pack, next_no)
        filtered = []
        for ref in pack.get("refs", []):
            url = ref.get("url") or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            filtered.append(ref)
            all_refs.append(ref)
        pack["refs"] = filtered
        packs.append(pack)
    return next_no


async def _gather_research(plan: dict, user_text: str):
    if not plan.get("needs_research"):
        coverage = {"sufficient": True, "missing_facets": [], "followup_queries": [], "supported_conclusions": [], "confidence": 1.0}
        return [], [], True, coverage, 0

    initial = plan.get("subquestions") or [user_text]
    initial = initial[:max(1, EXECUTIVE_MAX_RESEARCH_QUERIES)]
    packs, all_refs, seen_urls = [], [], set()
    next_no = 1
    attempted = set()
    rounds = 0
    queries = initial
    coverage = {"sufficient": False, "missing_facets": ["not_assessed"], "followup_queries": [], "supported_conclusions": [], "confidence": 0.0}

    while queries and rounds < max(1, EXECUTIVE_MAX_RESEARCH_ROUNDS):
        clean = []
        for q in queries:
            q = str(q or "").strip()
            key = q.lower()
            if not q or key in attempted:
                continue
            attempted.add(key)
            clean.append(q)
        if not clean:
            break
        rounds += 1
        raw = await _research_queries(clean[:max(1, EXECUTIVE_MAX_RESEARCH_QUERIES)])
        next_no = _merge_research(raw, packs, all_refs, next_no, seen_urls)
        coverage = await _step(f"coverage:{rounds}", assess_research_coverage, user_text, plan, packs, all_refs, kind="evidence_snapshot")
        if coverage.get("sufficient"):
            break
        queries = coverage.get("followup_queries") or []

    structural = bool(packs) and any(p.get("enough") for p in packs)
    research_enough = bool(coverage.get("sufficient")) and structural
    return packs, all_refs, research_enough, coverage, rounds


def _research_context(packs: list, coverage=None):
    if not packs:
        return "SECOND BRAIN RESEARCH: not required"
    chunks = ["SECOND BRAIN RESEARCH PACKS:"]
    for idx, pack in enumerate(packs, 1):
        chunks.append(f"\n--- RESEARCH {idx}: {pack['query']} ---\n{str(pack.get('context') or '')[:14000]}")
    if coverage:
        chunks.append("\n--- EVIDENCE COVERAGE ---\n" + json.dumps(coverage, ensure_ascii=False))
    return "\n".join(chunks)


def _answer_system(plan: dict, research_enough: bool, personal: bool):
    evidence_rule = (
        "The evidence-gap controller found the requested facets sufficiently covered; still distinguish direct facts from inference."
        if research_enough else
        "Evidence remains incomplete. Do not manufacture missing facts. Produce the strongest useful answer possible while explicitly labeling unproven parts."
    )
    privacy = (
        "The memory block contains user-specific context. Treat explicit user memories as authoritative when relevant."
        if personal else
        "Do not invent personal facts about the user."
    )
    return (
        "You are the execution layer of a persistent Second Brain. Finish the user's task rather than merely discussing it. "
        "Use the plan, rolling context, relevant memory and research evidence before general knowledge. "
        "Do not repeat questions already answered. If choices were delegated, make the strongest reasonable choice and continue. "
        "Every concrete statistic, date-sensitive factual claim, market-size statement, demand claim, low-competition claim, or supply-gap claim based on research must carry a valid inline [S#] citation. "
        "Never use evidence of one kind as proof of another (for example labor shortage != low SEO competition). "
        f"{evidence_rule} {privacy} "
        "Return a complete coherent answer in the user's language. Do not stop mid-section."
    )


def _draft_route(plan: dict):
    personal = bool(plan.get("needs_memory") or plan.get("mode") == "personal")
    if personal and not UNOROUTER_PRIVATE_CHAT:
        return "local"
    if plan.get("needs_research") and plan.get("complexity") == "complex":
        return "reasoning"
    return "fast_cloud"


async def _draft(user_text: str, history: list, plan: dict, memories: list, research_packs: list, research_enough: bool, coverage: dict):
    personal = bool(plan.get("needs_memory") or plan.get("mode") == "personal")
    messages = [
        {"role": "system", "content": _answer_system(plan, research_enough, personal)},
        {"role": "system", "content": "EXECUTION PLAN:\n" + json.dumps(plan, ensure_ascii=False)},
        {"role": "system", "content": _conversation_state()},
        {"role": "system", "content": _memory_context(memories)},
        {"role": "system", "content": _research_context(research_packs, coverage)},
        *history,
        {"role": "user", "content": user_text},
    ]
    tokens = EXECUTIVE_LONG_MAX_TOKENS if plan.get("long_form") else EXECUTIVE_SIMPLE_MAX_TOKENS
    return await chat(messages, temperature=0.16, num_predict=tokens, route=_draft_route(plan))


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
        if not sentence or sentence.startswith("#") or sentence.startswith(("- ", "* ")) and len(sentence) < 45:
            continue
        # Skip labels/headings that merely name the user's requested concept.
        if len(sentence) < 55 and any(x in sentence for x in ("ジャンルの例", "選定基準", "調査方法", "記事構成")):
            continue
        has_number = bool(re.search(r"\d(?:[\d,.]*\d)?\s*(?:%|％|万人|億円|兆円|人|件|年|倍|ドル|円)", sentence))
        strong_claim = any(marker.lower() in sentence.lower() for marker in _STRONG_RESEARCH_ASSERTIONS)
        if not (has_number or strong_claim):
            continue
        markers = set(re.findall(r"\[(S\d+)\]", sentence))
        if not (markers & valid):
            issues.append("uncited_material_claim: " + sentence[:220])
        if len(issues) >= 8:
            break
    return issues


async def _critique(user_text: str, plan: dict, draft: str, research_packs: list, research_enough: bool, refs: list, coverage: dict):
    deterministic = _deterministic_research_audit(draft, refs, bool(plan.get("needs_research")))
    if deterministic:
        return {"pass": False, "issues": deterministic, "missing": [], "confidence": 1.0, "deterministic": True}

    if not EXECUTIVE_REVIEW_ENABLED or plan.get("complexity") != "complex":
        return {"pass": True, "issues": [], "missing": [], "confidence": 1.0, "skipped": True}

    prompt = f"""Audit the proposed answer against the user's request, plan and evidence.
Return one strict JSON object only:
{{"pass": true|false, "issues": ["..."], "missing": ["..."], "confidence": 0.0}}

User request:
{user_text}

Plan:
{json.dumps(plan, ensure_ascii=False)}

Evidence coverage:
{json.dumps(coverage, ensure_ascii=False)}

Evidence:
{_research_context(research_packs)[:16000]}

Draft:
{draft[:18000]}

Fail for substantive problems only:
- an explicit part of the request is ignored;
- an unnecessary clarification is asked after delegation;
- unsupported demand/supply/competition/current facts are asserted;
- a concrete externally verifiable statistic lacks a valid source marker;
- evidence of A is misrepresented as proof of B;
- internal contradiction;
- requested deliverable is incomplete;
- source ids are fabricated.
Do NOT fail a generic heading, a description of the user's strategy, or a clearly labeled hypothesis merely because it lacks a citation.
"""
    try:
        raw = await chat(
            [
                {"role": "system", "content": "You are a strict evidence-aware answer auditor. JSON only. Fail closed on substantive uncertainty."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            num_predict=500,
            route="local" if plan.get("needs_memory") and not UNOROUTER_PRIVATE_CHAT else "verify",
        )
        result = _extract_json_object(raw)
    except Exception as e:
        print(f"[EXECUTIVE] critic failed: {e}")
        return {"pass": False, "issues": ["critic_unavailable"], "missing": [], "confidence": 0.0}
    if not result:
        return {"pass": False, "issues": ["critic_parse_failed"], "missing": [], "confidence": 0.0}
    result["pass"] = bool(result.get("pass", False))
    result["issues"] = [str(x)[:500] for x in result.get("issues", [])][:8]
    result["missing"] = [str(x)[:500] for x in result.get("missing", [])][:8]
    try:
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    except Exception:
        result["confidence"] = 0.0
    return result


async def _revise(user_text: str, plan: dict, draft: str, critique: dict, memories: list, research_packs: list, research_enough: bool, coverage: dict):
    personal = bool(plan.get("needs_memory") or plan.get("mode") == "personal")
    prompt = f"""Repair the draft so it fully satisfies the user's request and passes the audit.

User request:
{user_text}

Plan:
{json.dumps(plan, ensure_ascii=False)}

Evidence coverage:
{json.dumps(coverage, ensure_ascii=False)}

Critique:
{json.dumps(critique, ensure_ascii=False)}

Original draft:
{draft}

Rules:
- Fix every substantive issue.
- Remove or weaken any number or external claim that cannot be cited.
- Every retained material research claim must carry a valid inline [S#].
- Do not invent evidence or source ids.
- If a requested conclusion remains unproven, state the precise limitation but still complete all parts that can be completed honestly.
- Answer in the user's language and finish the deliverable.
"""
    messages = [
        {"role": "system", "content": _answer_system(plan, research_enough, personal)},
        {"role": "system", "content": _conversation_state()},
        {"role": "system", "content": _memory_context(memories)},
        {"role": "system", "content": _research_context(research_packs, coverage)},
        {"role": "user", "content": prompt},
    ]
    tokens = EXECUTIVE_LONG_MAX_TOKENS if plan.get("long_form") else EXECUTIVE_SIMPLE_MAX_TOKENS
    return await chat(messages, temperature=0.06, num_predict=tokens, route=_draft_route(plan))


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
    for ref in used[:10]:
        url = ref.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        domain = ref.get("domain") or ""
        lines.append(f"[{ref.get('ref')}] {ref.get('source') or domain} — {ref.get('title') or ''}\n{url}")
    return response if not lines else response + "\n\n参照した情報源:\n" + "\n".join(lines)


def _safe_research_failure(coverage: dict, critique: dict):
    missing = coverage.get("missing_facets") or []
    issues = critique.get("issues") or []
    parts = [
        "第2の脳は追加探索と再生成まで行いましたが、根拠監査を通せない主張が残ったため、未検証の完成品を出すのを止めました。"
    ]
    if missing:
        parts.append("不足している証拠: " + " / ".join(str(x)[:180] for x in missing[:5]))
    if issues:
        parts.append("最終監査: " + " / ".join(str(x)[:180] for x in issues[:4]))
    parts.append("これは『調べずに止まった』のではなく、探索を繰り返しても今回の情報源では証明できなかった、という状態です。")
    return "\n\n".join(parts)


async def _store_conversation_later(user_text: str, response: str, run_id=None):
    try:
        metadata = {"agent_run_id": run_id} if run_id else None
        await _step("persist:user", store_memory, "conversation", "user", user_text, 0.55, metadata)
        await _step("persist:assistant", store_memory, "conversation", "assistant", response, 0.20, metadata)
    except Exception as e:
        print(f"[EXECUTIVE] memory save failed: {e}")
    try:
        await _step("persist:consolidation", consolidate_user_turn, user_text, run_id)
    except Exception as e:
        print(f"[EXECUTIVE] consolidation failed: {e}")
    try:
        await _step("persist:summary", maybe_update_conversation_summary, user_text, response, run_id)
    except Exception as e:
        print(f"[EXECUTIVE] context summary failed: {e}")


async def run(user_text: str):
    if not EXECUTIVE_ENABLED:
        response = await chat(
            [{"role": "user", "content": user_text}],
            temperature=0.25,
            num_predict=EXECUTIVE_SIMPLE_MAX_TOKENS,
            route="fast_cloud",
        )
        if job_context.get() is not None:
            await _store_conversation_later(user_text, response)
        else:
            asyncio.create_task(_store_conversation_later(user_text, response))
        return ExecutiveResult(response, [], None, "disabled", {}, {})

    history = await checkpoint("history", {"request": user_text}, _history)
    if _is_greeting(user_text):
        response = await chat(
            [{"role": "system", "content": "Reply naturally and briefly in the user's language."}, *history[-4:], {"role": "user", "content": user_text}],
            num_predict=100,
            route="local",
        )
        if job_context.get() is not None:
            await _store_conversation_later(user_text, response)
        else:
            asyncio.create_task(_store_conversation_later(user_text, response))
        return ExecutiveResult(response, [], None, "greeting", {}, {})

    if job_context.get() is not None:
        job_context.get().conversation_snapshot = await checkpoint("conversation_state", {}, format_conversation_summary)
    plan_start = _now_ms()
    plan = await _step("plan", _make_plan, user_text, history, kind="plan")
    mode = plan.get("mode", "work")
    run_id = await checkpoint("agent_run", {"request": user_text, "plan": plan}, lambda: start_agent_run(user_text, plan.get("goal", ""), mode, plan))
    step_no = 1
    add_agent_step(run_id, step_no, "plan", "Create execution plan", output_data=plan, duration_ms=_elapsed_ms(plan_start))
    step_no += 1

    try:
        memory_start = _now_ms()
        memories = await _step("memory", _gather_memory, user_text, bool(plan.get("needs_memory")))
        if memories:
            plan["needs_memory"] = True  # Retrieved notes may be private; keep generation local.
        add_agent_step(
            run_id, step_no, "memory", "Retrieve relevant long-term memory",
            output_data={"count": len(memories), "ids": [m.get("id") for m in memories[:8]]},
            duration_ms=_elapsed_ms(memory_start),
        )
        step_no += 1

        research_start = _now_ms()
        research_packs, refs, research_enough, coverage, research_rounds = await _step("research", _gather_research, plan, user_text, kind="research_pack")
        add_agent_step(
            run_id, step_no, "research", "Iterative evidence collection and gap analysis",
            input_data={"initial_queries": plan.get("subquestions", []), "required_evidence": plan.get("required_evidence", [])},
            output_data={
                "packs": len(research_packs),
                "references": len(refs),
                "sufficient": research_enough,
                "rounds": research_rounds,
                "coverage": coverage,
            },
            duration_ms=_elapsed_ms(research_start),
        )
        step_no += 1

        draft_start = _now_ms()
        response = await _step("draft", _draft, user_text, history, plan, memories, research_packs, research_enough, coverage, kind="draft")
        add_agent_step(
            run_id, step_no, "draft", "Produce answer or deliverable",
            output_data={"characters": len(response), "route": _draft_route(plan)},
            duration_ms=_elapsed_ms(draft_start),
        )
        step_no += 1

        review_start = _now_ms()
        critique = await _step("review:0", _critique, user_text, plan, response, research_packs, research_enough, refs, coverage, kind="critique")
        add_agent_step(
            run_id, step_no, "review", "Audit answer against goal and evidence",
            output_data=critique,
            duration_ms=_elapsed_ms(review_start),
        )
        step_no += 1

        revisions = 0
        while not critique.get("pass", True) and revisions < max(0, EXECUTIVE_MAX_REVISIONS):
            revise_start = _now_ms()
            response = await _step(f"revise:{revisions+1}", _revise, user_text, plan, response, critique, memories, research_packs, research_enough, coverage, kind="revised_draft")
            revisions += 1
            add_agent_step(
                run_id, step_no, "revise", "Repair failed draft",
                input_data={"issues": critique.get("issues", []), "missing": critique.get("missing", [])},
                output_data={"characters": len(response), "revision": revisions},
                duration_ms=_elapsed_ms(revise_start),
            )
            step_no += 1
            critique = await _step(f"review:{revisions}", _critique, user_text, plan, response, research_packs, research_enough, refs, coverage, kind="critique")

        if plan.get("needs_research") and not critique.get("pass", False):
            response = _safe_research_failure(coverage, critique)
        else:
            response = _append_sources(response, refs)

        await checkpoint("final", {"response": response, "critique": critique}, lambda: response, "answer")
        finish_agent_run(
            run_id,
            "completed",
            final_response=response,
            critique=critique,
            metadata={
                "research_sufficient": research_enough,
                "references": len(refs),
                "research_rounds": research_rounds,
                "coverage": coverage,
                "revisions": revisions,
                "memory_count": len(memories),
                "draft_route": _draft_route(plan),
            },
        )
        if job_context.get() is not None:
            await _store_conversation_later(user_text, response, run_id)
        else:
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
        "max_initial_research_queries": EXECUTIVE_MAX_RESEARCH_QUERIES,
        "max_research_rounds": EXECUTIVE_MAX_RESEARCH_ROUNDS,
        "max_revisions": EXECUTIVE_MAX_REVISIONS,
        "recent_runs": len(runs),
        "recent_completed": completed,
        "recent_failed": failed,
        "latest_run_id": runs[0]["id"] if runs else None,
        "pipeline": ["plan", "memory", "research", "gap_check", "draft", "review", "revise", "persist"],
    }
