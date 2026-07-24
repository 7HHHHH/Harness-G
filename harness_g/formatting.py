import os
from typing import Iterable, List


def raw_score_observation_enabled() -> bool:
    return os.environ.get("HARNESS_G_SHOW_RAW_SCORE", "").lower() in {"1", "true", "yes", "on"}


def score_text(score: object) -> str:
    try:
        return f"{float(score):.4f}"
    except Exception:
        return "0.0000"


def observation_block(lines: Iterable[str]) -> str:
    return "\n".join(["[HARNESS_G_OBS]", *lines, "[/HARNESS_G_OBS]"])


def _clean_display_text(text: object) -> str:
    return str(text or "").strip().strip('"').strip()


def _title_text(title: object, text: object) -> str:
    """Render ``title: text`` but avoid duplicating the title when the body
    already begins with it (Wikipedia bodies usually do), and drop bare quotes."""
    text = _clean_display_text(text)
    title = str(title or "").strip()
    if not title:
        return text
    if text.lower().startswith(title.lower()):
        return text
    return f"{title}: {text}"


def sentence_line(display_id: str, sentence: dict, include_score: bool = False) -> str:
    prefix = display_id
    if (include_score or raw_score_observation_enabled()) and sentence.get("score") is not None:
        prefix += f" | score: {score_text(sentence.get('score'))}"
    if sentence.get("source"):
        prefix += f" | source: {sentence.get('source')}"
    return f"{prefix} | {_title_text(sentence.get('title', ''), sentence.get('text', ''))}"


def selected_line(display_id: str, sentence: dict) -> str:
    return f"{display_id} | {_title_text(sentence.get('title', ''), sentence.get('text', ''))}"


def entity_line(display_id: str, entity: dict) -> str:
    surface = entity.get("surface_forms", [entity.get("canonical", "")])
    if isinstance(surface, list):
        surface = surface[0] if surface else entity.get("canonical", "")
    suffix = ""
    if raw_score_observation_enabled() and entity.get("score") is not None:
        suffix += f" | score: {score_text(entity.get('score'))}"
    if entity.get("source"):
        suffix += f" | source: {entity.get('source')}"
    return f"{display_id} | {surface} | {entity.get('label', 'ENTITY')}{suffix}"


def actions_lines(action_map: dict) -> List[str]:
    lines = []
    for action_id, action in action_map.items():
        action_type = action.get("type")
        if action_type == "SELECT":
            suffix = _action_suffix(action)
            lines.append(f"{action_id} = SELECT {action.get('display_sid')}{suffix}")
        elif action_type == "LOOKUP":
            lines.append(f"{action_id} = LOOKUP {action.get('display_eid')}{_lookup_suffix(action)}")
        elif action_type == "ANSWER_WITH":
            display_sids = action.get("display_sids") or ([action.get("display_sid")] if action.get("display_sid") else [])
            sid_text = ",".join(str(s) for s in display_sids if s)
            line = f"{action_id} = ANSWER_WITH {sid_text}".rstrip()
            evidence = action.get("evidence_preview")
            if evidence:
                line += f" | evidence: {evidence}"
            lines.append(line)
        elif action_type == "ANSWER":
            lines.append(f"{action_id} = ANSWER")
        else:
            lines.append(f"{action_id} = {action_type}")
    return lines


def _lookup_suffix(action: dict) -> str:
    pieces = []
    if action.get("entity_surface"):
        pieces.append(f"entity: {action.get('entity_surface')}")
    if action.get("via"):
        pieces.append(f"via: {action.get('via')}")
    if raw_score_observation_enabled() and action.get("score") is not None:
        pieces.append(f"score: {score_text(action.get('score'))}")
    suffix = ""
    if pieces:
        suffix = " | " + " | ".join(pieces)
    if action.get("low_priority"):
        suffix += " | low priority"
    return suffix


def _action_suffix(action: dict) -> str:
    pieces = []
    if action.get("source"):
        pieces.append(f"source: {action.get('source')}")
    if raw_score_observation_enabled() and action.get("score") is not None:
        pieces.append(f"score: {score_text(action.get('score'))}")
    if action.get("expanded_entity"):
        pieces.append(f"expanded_entity: {action.get('expanded_entity')}")
    if not pieces:
        return ""
    return " | " + " | ".join(pieces)
