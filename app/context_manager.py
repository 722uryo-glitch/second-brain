import json
import re

from .ollama_client import chat
from .runtime_state import get_state, set_state

_STATE_KEY = "conversation_rolling_summary_v1"


def get_conversation_summary():
    state = get_state(_STATE_KEY, {}) or {}
    if not isinstance(state, dict):
        return {}
    return state


def format_conversation_summary():
    state = get_conversation_summary()
    if not state:
        return "ROLLING CONVERSATION CONTEXT: none"
    return "ROLLING CONVERSATION CONTEXT:\n" + json.dumps(state, ensure_ascii=False)


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


async def maybe_update_conversation_summary(user_text: str, assistant_text: str, run_id=None):
    """Refresh durable conversation context periodically, off the response path.

    Recent raw turns remain the short-term context. This compact state preserves
    older goals, decisions and unresolved threads after those turns fall out of
    the prompt window.
    """
    if run_id is not None:
        try:
            # Updating every four completed runs avoids constant local-model load.
            if int(run_id) % 4 != 0:
                return {"updated": False, "reason": "interval"}
        except Exception:
            pass

    previous = get_conversation_summary()
    prompt = f"""Update the rolling conversation state using ONLY the new USER and ASSISTANT turn.
Return one strict JSON object only.

Previous state:
{json.dumps(previous, ensure_ascii=False)}

New USER turn:
{str(user_text or '')[:5000]}

New ASSISTANT turn:
{str(assistant_text or '')[:7000]}

Return:
{{
  "summary": "compact neutral summary, max 900 Japanese characters",
  "active_goals": ["..."],
  "decisions": ["..."],
  "open_threads": ["..."],
  "constraints": ["..."]
}}

Rules:
- Preserve still-relevant prior state and update it rather than rewriting history.
- User statements outrank assistant statements.
- Assistant proposals are not user decisions unless the user accepted them.
- Do not infer sensitive traits or hidden personal facts.
- Remove resolved open threads.
- Keep each list at at most 8 items.
"""
    try:
        raw = await chat(
            [
                {"role": "system", "content": "You are a private local conversation-state compressor. JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            num_predict=700,
            route="local",
        )
        state = _extract_json(raw)
    except Exception as e:
        return {"updated": False, "error": str(e)}

    if not state:
        return {"updated": False, "reason": "parse_failed"}
    normalized = {
        "summary": str(state.get("summary") or "")[:1800],
        "active_goals": [str(x)[:400] for x in state.get("active_goals", []) if str(x).strip()][:8],
        "decisions": [str(x)[:400] for x in state.get("decisions", []) if str(x).strip()][:8],
        "open_threads": [str(x)[:400] for x in state.get("open_threads", []) if str(x).strip()][:8],
        "constraints": [str(x)[:400] for x in state.get("constraints", []) if str(x).strip()][:8],
    }
    set_state(_STATE_KEY, normalized)
    print("[CONTEXT] rolling summary updated")
    return {"updated": True, "state": normalized}
