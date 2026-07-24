"""Read-only SNC candidate previews.

These helpers surface the evidence that an action would reveal without applying
the action, committing an episode transition, or mutating graph-index state.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from harness_g.text_utils import compact_keywords


def preview_candidate(
    episode: Any,
    action_id: Any,
    *,
    hop_query_override: Any = None,
    bridge_query_override: Any = None,
) -> dict[str, Any]:
    """Return the evidence/entity preview for one current frontier action.

    ``hop_query_override`` / ``bridge_query_override`` carry the parsed
    ``|| query`` refinement of the *taken* action so the taken preview matches
    what ``env.step`` actually executes. Frontier candidates are previewed
    without overrides (they have no model-generated query). Overrides use
    ``is not None`` so an explicit empty string is preserved rather than
    collapsing to the action-map default.
    """
    try:
        action = episode.current_action_map[action_id]
        action_type = _get_value(action, "type")
        graph_index = episode.graph_index

        surfaced_sids: list[str]
        used_entity_ids: frozenset[str]

        if action_type == "SELECT":
            surfaced_sids = [_get_required_value(action, "sid")]
            used_entity_ids = frozenset()
        elif action_type == "LOOKUP":
            eid = _get_required_value(action, "eid")
            anchor_eid = _get_value(action, "anchor_eid")
            # LOOKUP does not take a model-written ``|| need_query``. The
            # retrieval query is the environment-built mixquery (question +
            # selected evidence text), so ``hop_query_override`` is intentionally
            # ignored here — the taken preview must match ``env._lookup``, which
            # also ignores any parsed hop_query for LOOKUP.
            mixquery = _lookup_mixquery(episode)
            ranked = graph_index.lookup_entity(
                eid,
                mixquery,
                topk=episode.expanded_visible_sentence_k,
                anchor_eid=anchor_eid,
            )
            surfaced_sids = _ranked_sids(ranked)
            used = {eid}
            if anchor_eid:
                used.add(str(anchor_eid))
            used_entity_ids = frozenset(used)
        elif action_type == "ANSWER_WITH":
            sids = _get_value(action, "sids")
            if not sids:
                single = _get_value(action, "sid")
                sids = [single] if single else []
            surfaced_sids = [str(sid) for sid in sids if sid]
            used_entity_ids = frozenset()
        else:
            return _empty_preview()

        surfaced_entity_ids = _surfaced_entity_ids(graph_index, surfaced_sids)
        return {
            "surfaced_sids": surfaced_sids,
            "surfaced_entity_ids": surfaced_entity_ids,
            "used_entity_ids": used_entity_ids,
        }
    except Exception:
        return _empty_preview()


def preview_frontier(episode: Any) -> dict[str, dict[str, Any]]:
    """Preview every action currently available on the episode frontier."""
    previews: dict[str, dict[str, Any]] = {}
    for action_id in list(getattr(episode, "current_action_map", {})):
        try:
            previews[action_id] = preview_candidate(episode, action_id)
        except Exception:
            previews[action_id] = _empty_preview()
    return previews


def _empty_preview() -> dict[str, Any]:
    return {
        "surfaced_sids": [],
        "surfaced_entity_ids": frozenset(),
        "used_entity_ids": frozenset(),
    }


def _get_value(item: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(item, Mapping):
            if key in item:
                return item[key]
        elif hasattr(item, key):
            return getattr(item, key)
    return default


def _get_required_value(item: Any, *keys: str) -> str:
    value = _get_value(item, *keys)
    if value is None:
        raise KeyError(keys[0])
    return str(value)


def _ranked_sids(ranked: Any) -> list[str]:
    sids: list[str] = []
    for row in ranked or []:
        sid = _get_value(row, "sid")
        if sid is not None:
            sids.append(str(sid))
    return sids


def _surfaced_entity_ids(graph_index: Any, surfaced_sids: list[str]) -> frozenset[str]:
    entity_ids: set[str] = set()
    for sid in surfaced_sids:
        for entity in graph_index.get_entities_for_sentence(sid) or []:
            eid = _get_value(entity, "eid")
            if eid is not None:
                entity_ids.add(str(eid))
    return frozenset(entity_ids)



def _lookup_mixquery(episode: Any) -> str:
    """Build the LOOKUP retrieval query, matching ``env._lookup``.

    ``mixquery`` = the full question (never truncated) + text of every
    SELECTed evidence sentence filling the rest of the ``mixquery_max_words``
    budget. When nothing has been SELECTed yet it falls back to the question
    alone. This must stay in lock-step with ``env._lookup_mixquery`` so the
    taken preview retrieves the same sentences as ``env.step``.
    """
    question = str(getattr(episode, "question", "") or "").strip()
    selected_text = _selected_text(episode)
    budget = getattr(episode, "mixquery_max_words", None)
    if budget is None:
        budget = 64
    budget = max(int(budget), 1)
    question_words = question.split()
    evidence_budget = max(budget - len(question_words), 0)
    evidence_words = selected_text.split()[:evidence_budget]
    return " ".join(question_words + evidence_words)


def _selected_text(episode: Any) -> str:
    """Join the text of all SELECTed evidence sentences.

    ``episode.selected_evidence`` is a list of sids in the real environment;
    some test doubles store sentence dicts instead. Both shapes are handled so
    the preview works against the real episode and against fixtures.
    """
    graph_index = getattr(episode, "graph_index", None)
    sentences = getattr(graph_index, "sentences", None)
    parts: list[str] = []
    for item in getattr(episode, "selected_evidence", []) or []:
        text = ""
        if isinstance(item, str):
            sentence = sentences.get(item) if isinstance(sentences, dict) else None
            if sentence:
                text = str(sentence.get("text", "") or "").strip()
        elif isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _cap_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    return " ".join(str(text).split()[:max_words])
