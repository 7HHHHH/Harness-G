import json
import re
from typing import Dict, Optional


ACTION_RE = re.compile(r"^(A\d+)(?:\s*\|\|\s*(.*))?$")


def parse_tool_query(text: str) -> Optional[str]:
    match = re.search(r"<query>(.*?)</query>", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
    except Exception:
        return None
    if "query" not in data:
        return None
    return str(data["query"])


def parse_harness_g_action(
    raw: str,
    action_map: Optional[Dict[str, dict]] = None,
    initialized: bool = True,
    qh_max_words: int = 16,
) -> dict:
    action_map = action_map or {}
    action = (raw or "").strip()
    result = {
        "raw": action,
        "action_id": None,
        "hop_query": None,
        "is_valid": False,
        "invalid_reason": None,
        "is_natural_query": False,
        "semantic_action": None,
    }

    upper = action.upper()
    if upper == "INIT":
        result["semantic_action"] = "INIT"
        result["is_valid"] = not initialized
        if not result["is_valid"]:
            result["invalid_reason"] = "INIT is only valid before the episode is initialized"
        return result

    match = ACTION_RE.match(action)
    if not match:
        result["is_natural_query"] = bool(action and not action.startswith("<"))
        result["invalid_reason"] = "expected INIT or an available action id such as A0"
        return result

    action_id, hop_query = match.group(1), match.group(2)
    result["action_id"] = action_id
    hop_query = hop_query.strip() if hop_query is not None else None
    if hop_query == "":
        hop_query = None
    result["hop_query"] = hop_query

    mapped = action_map.get(action_id)
    if mapped is None:
        result["invalid_reason"] = f"unknown or unavailable action id: {action_id}"
        return result

    semantic_action = mapped.get("type") or mapped.get("semantic_action")
    result["semantic_action"] = semantic_action

    # Actions never accept a model-written ``|| short query``: the environment
    # builds the retrieval query (mixquery = question + selected evidence text)
    # itself, so any ``A_k || ...`` is rejected.
    if hop_query:
        result["invalid_reason"] = "actions do not accept a short query; use a plain action id such as A0"
        return result

    result["is_valid"] = True
    return result
