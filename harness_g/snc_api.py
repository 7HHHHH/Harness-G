"""SNC side-channel payload helpers for the Harness-G API server.

The API side-channel is intentionally opt-in. When disabled, callers should
continue returning the legacy observation-only response.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .answer_scoring import build_answer_scoring_prompt
from .protocol import parse_harness_g_action
from .snc_preview import preview_candidate


_TRUTHY = {"1", "true", "t", "yes", "y", "on"}
_INFORMATION_ACTION_TYPES = frozenset({"LOOKUP"})
_COMMIT_ACTION_TYPES = frozenset({"SELECT", "ANSWER_WITH"})


def snc_enabled() -> bool:
    """The API SNC side-channel is always enabled."""

    return True


def frontier_topk() -> int:
    """Return the number of candidate previews to include in the SNC frontier."""

    return 4


def _frontier_stratify_enabled() -> bool:
    """Return whether action-type stratification is on (default on).

    This only controls whether ``_choose_frontier_action_ids`` picks a
    best-effort one-per-type sample before filling with ranked order. It does
    NOT control the taken-exclusion invariant (that is always on and is a
    Fix B correctness requirement, not an ablation knob).
    """
    return True


def build_snc_step_payload(episode: Any, raw_action: str) -> dict[str, Any] | None:
    """Build the pre-step SNC payload for one API action.

    This must be called before ``episode.step(...)`` mutates the episode. The
    frontier is capped by ``frontier_topk()`` to keep API payloads
    bounded.

    Fix B invariants (always on, not flag-gated):
      * ``taken`` is previewed separately with the parsed ``hop_query`` /
        ``bridge_query`` override so its ``answer_prompt`` reflects what
        ``env.step`` actually executes, not the action-map default query.
      * ``taken_action_id`` is NEVER a key in ``frontier``. If it were, the
        trainer's ``_collect_score_requests`` (snc_trainer.py:246-251) would
        reuse the frontier's default-query prompt score for the taken ref and
        silently undo Fix B. The frontier therefore excludes taken regardless
        of stratification.
    """
    try:
        if not snc_enabled():
            return None
        if not getattr(episode, "initialized", False):
            return None

        action_map = getattr(episode, "current_action_map", None)
        if not action_map:
            return None

        selected_before = list(getattr(episode, "selected_evidence", []))
        seen_before = _seen_sids_before(episode)
        parsed = _resolve_taken_action(episode, raw_action, action_map)
        if parsed is None:
            return None
        taken_action_id = parsed["action_id"]
        hop_query = parsed.get("hop_query")

        baseline_prompt = build_answer_scoring_prompt(
            getattr(episode, "question", ""),
            seen_before,
            getattr(episode, "graph_index"),
        )

        taken_preview = preview_candidate(
            episode,
            taken_action_id,
            hop_query_override=hop_query,
            bridge_query_override=hop_query,
        )
        taken = _build_candidate_payload(
            episode, taken_action_id, selected_before, seen_before, taken_preview,
        )
        taken["action_id"] = taken_action_id
        taken["action_type"] = _action_type(action_map.get(taken_action_id))

        # SNC's immediate frontier term compares information-acquisition
        # actions only. SELECT/ANSWER_WITH/ANSWER do not reveal new evidence;
        # SELECT can still receive downstream enabling credit through explicit
        # provenance, while terminal quality remains an outcome signal.
        frontier: dict[str, dict[str, Any]] = {}
        duplicate_contexts = 0
        if taken.get("is_information_action"):
            context_keys: set[tuple[str, ...]] = set()
            for action_id in _ordered_frontier_action_ids(action_map, taken_action_id):
                candidate = _build_candidate_payload(
                    episode,
                    action_id,
                    selected_before,
                    seen_before,
                    preview_candidate(episode, action_id),
                )
                context_key = tuple(str(sid) for sid in candidate["context_sids"])
                if context_key in context_keys:
                    duplicate_contexts += 1
                    continue
                context_keys.add(context_key)
                frontier[action_id] = candidate
                if len(frontier) >= frontier_topk():
                    break

        return {
            "selected_sids_before": selected_before,
            "seen_sids_before": seen_before,
            "baseline_prompt": baseline_prompt,
            "taken_action_id": taken_action_id,
            "frontier": frontier,
            "taken": taken,
            "frontier_duplicate_contexts_skipped": duplicate_contexts,
        }
    except Exception:
        raise


def _build_candidate_payload(
    episode: Any,
    action_id: str,
    selected_before: list[str],
    seen_before: list[str],
    preview: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one frontier/taken candidate dict from a preview.

    Used for both frontier candidates (no override) and the taken action
    (with override). The shape must stay consistent so the trainer can treat
    them uniformly; the taken dict additionally gets ``action_id`` /
    ``action_type`` appended by the caller.
    """
    surfaced_sids = _json_list(preview.get("surfaced_sids", []))
    surfaced_entity_ids = _sorted_list(preview.get("surfaced_entity_ids", []))
    used_entity_ids = _sorted_list(preview.get("used_entity_ids", []))
    seen_set = set(seen_before)
    new_sids = [sid for sid in surfaced_sids if sid not in seen_set]
    ctx_sids = list(seen_before) + new_sids
    action = getattr(episode, "current_action_map", {}).get(action_id)
    action_type = (_action_type(action) or "").upper()
    action_sids = _action_sids(action)
    if action_type in _COMMIT_ACTION_TYPES:
        produced_sids = action_sids
        consumed_sids = action_sids
    elif action_type in _INFORMATION_ACTION_TYPES:
        produced_sids = new_sids
        consumed_sids = list(selected_before)
    else:
        produced_sids = []
        consumed_sids = []
    produced_entity_ids = _entity_ids_for_sids(episode, produced_sids)
    answer_prompt = build_answer_scoring_prompt(
        getattr(episode, "question", ""),
        ctx_sids,
        getattr(episode, "graph_index"),
    )
    return {
        "surfaced_sids": surfaced_sids,
        "new_sids": new_sids,
        "context_sids": ctx_sids,
        "surfaced_entity_ids": surfaced_entity_ids,
        "produced_sids": produced_sids,
        "produced_entity_ids": produced_entity_ids,
        "consumed_sids": consumed_sids,
        "used_entity_ids": used_entity_ids,
        "is_information_action": action_type in _INFORMATION_ACTION_TYPES,
        "answer_prompt": answer_prompt,
    }


def _choose_frontier_action_ids(
    action_map: Mapping[str, Any],
    taken_action_id: str | None,
    limit: int,
) -> list[str]:
    """Choose the frontier action ids for the SNC baseline.

    Invariants:
      * ``taken_action_id`` is NEVER in the returned list. This is a Fix B
        correctness requirement (prevents the trainer from reusing the
        frontier's default-query prompt score for the taken ref). It holds
        whether stratification is on or off.
      * With action-type stratification (always on), the first
        pass takes at most one action per eligible information-action type so
        the baseline is not dominated by the first menu entries. Missing types
        do not consume quota. The second pass fills in ranked order.
      * When the flag is off, the list is pure ranked order (still excluding
        taken) — the ablation path.
    """
    if limit <= 0:
        return []
    return _ordered_frontier_action_ids(action_map, taken_action_id)[:limit]


def _ordered_frontier_action_ids(
    action_map: Mapping[str, Any], taken_action_id: str | None
) -> list[str]:
    """Return all eligible information actions in frontier priority order.

    Prompt/context equivalence requires previews, so final deduplication and
    top-k truncation happen in :func:`build_snc_step_payload`.
    """

    ranked = [
        (str(aid), action)
        for aid, action in _ranked_action_items(action_map)
        if str(aid) != str(taken_action_id)
        and (_action_type(action) or "").upper() in _INFORMATION_ACTION_TYPES
    ]

    if not _frontier_stratify_enabled():
        return [aid for aid, _ in ranked]

    by_type: dict[str, list[tuple[str, Any]]] = {}
    for aid, action in ranked:
        by_type.setdefault(_action_type(action) or "", []).append((aid, action))

    # Commit and terminal actions are excluded because the evidence scorer
    # cannot distinguish their transition semantics. They receive structural
    # or outcome credit instead of contaminating the evidence frontier.
    priority = ["LOOKUP"]

    chosen: list[str] = []
    chosen_set: set[str] = set()

    for typ in priority:
        bucket = by_type.get(typ) or []
        if not bucket:
            continue
        aid = bucket[0][0]
        chosen.append(str(aid))
        chosen_set.add(str(aid))

    for aid, _ in ranked:
        if str(aid) in chosen_set:
            continue
        chosen.append(str(aid))
        chosen_set.add(str(aid))

    return chosen


def _ranked_action_items(action_map: Mapping[str, Any]) -> list[tuple[str, Any]]:
    indexed_items = list(action_map.items())

    def sort_key(item_with_index: tuple[int, tuple[str, Any]]) -> tuple[int, float, int]:
        index, (_, action) = item_with_index
        score = _action_score(action)
        if score is None:
            return (1, 0.0, index)
        return (0, -score, index)

    return [
        item
        for _, item in sorted(
            enumerate(indexed_items),
            key=sort_key,
        )
    ]


def _action_score(action: Any) -> float | None:
    value: Any
    if isinstance(action, Mapping):
        value = action.get("score")
    else:
        value = getattr(action, "score", None)

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_taken_action(
    episode: Any,
    raw_action: str,
    action_map: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Parse ``raw_action`` and return a normalized taken-action dict.

    Returns ``{"action_id", "hop_query", "semantic_action"}`` or None if the
    action is invalid / unknown. Normalization is required because callers
    (and tests) may feed ``parse_harness_g_action`` results as a dict or a
    SimpleNamespace; downstream code should not have to guess. ``hop_query``
    is the Fix B payload — it must reach ``preview_candidate`` for the taken
    preview to match ``env.step``.

    Note: ``_parsed_action_id`` retains a legacy tuple branch for defensive
    compatibility, but ``parse_harness_g_action`` always returns a dict, so
    in practice the tuple path is dead. ``_parsed_hop_query`` therefore
    returns None for tuple shapes (a tuple has no defined hop_query position);
    this is acceptable because no real parser produces tuples.
    """
    try:
        parsed = parse_harness_g_action(
            raw_action,
            action_map=action_map,
            initialized=getattr(episode, "initialized", False),
            qh_max_words=getattr(episode, "qh_max_words", None),
        )
    except Exception:
        return None

    if _parsed_invalid(parsed):
        return None

    action_id = _parsed_action_id(parsed)
    if not action_id or action_id not in action_map:
        raw_action_id = raw_action.strip() if isinstance(raw_action, str) else None
        if raw_action_id in action_map and not _parsed_invalid(parsed):
            action_id = raw_action_id
        else:
            return None

    return {
        "action_id": str(action_id),
        "hop_query": _parsed_hop_query(parsed),
        "semantic_action": _parsed_semantic_action(parsed),
    }


def _parsed_hop_query(parsed: Any) -> str | None:
    # Tuples have no defined hop_query position; return None rather than guess.
    # parse_harness_g_action always returns a dict, so this only matters for
    # defensive compatibility with hypothetical tuple-returning parsers.
    if isinstance(parsed, Mapping):
        value = parsed.get("hop_query")
        return str(value) if value else None
    if isinstance(parsed, tuple):
        return None
    value = getattr(parsed, "hop_query", None)
    return str(value) if value else None


def _parsed_semantic_action(parsed: Any) -> str | None:
    if isinstance(parsed, Mapping):
        value = parsed.get("semantic_action")
        return str(value) if value else None
    if isinstance(parsed, tuple):
        return None
    value = getattr(parsed, "semantic_action", None)
    return str(value) if value else None


def _parsed_action_id(parsed: Any) -> str | None:
    if parsed is None:
        return None

    if isinstance(parsed, str):
        return parsed

    if isinstance(parsed, Mapping):
        if parsed.get("is_valid") is False or parsed.get("invalid_reason"):
            return None
        action_id = parsed.get("action_id") or parsed.get("id")
        return str(action_id) if action_id is not None else None

    valid = getattr(parsed, "valid", None)
    if valid is False:
        return None

    action_id = getattr(parsed, "action_id", None)
    if action_id is None:
        action_id = getattr(parsed, "id", None)
    if action_id is not None:
        return str(action_id)

    if isinstance(parsed, tuple) and parsed:
        first = parsed[0]
        return str(first) if first is not None else None

    return None


def _parsed_invalid(parsed: Any) -> bool:
    if parsed is None:
        return True
    if isinstance(parsed, Mapping):
        return parsed.get("is_valid") is False or bool(parsed.get("invalid_reason"))
    return getattr(parsed, "valid", None) is False


def _action_type(action: Any) -> str | None:
    if isinstance(action, Mapping):
        value = action.get("type") or action.get("action_type")
    else:
        value = getattr(action, "type", None) or getattr(action, "action_type", None)
    return str(value) if value is not None else None


def _seen_sids_before(episode: Any) -> list[str]:
    """Return SNC's stable cumulative evidence state before the action."""

    explicit = getattr(episode, "snc_seen_sids", None)
    if explicit is not None:
        return list(dict.fromkeys(str(sid) for sid in explicit if sid is not None))

    # Compatibility for older serialized/test episodes. Prefer stable display
    # insertion order when available; otherwise use selected + current visible.
    real_to_display = getattr(episode, "_real_sid_to_display", None)
    if isinstance(real_to_display, Mapping) and real_to_display:
        return list(dict.fromkeys(str(sid) for sid in real_to_display))
    selected = list(getattr(episode, "selected_evidence", []) or [])
    visible = list(getattr(episode, "current_visible_sids", []) or [])
    return list(dict.fromkeys(str(sid) for sid in selected + visible if sid is not None))


def _action_sids(action: Any) -> list[str]:
    if action is None:
        return []
    if isinstance(action, Mapping):
        values = action.get("sids")
        if not values:
            sid = action.get("sid")
            values = [sid] if sid else []
    else:
        values = getattr(action, "sids", None)
        if not values:
            sid = getattr(action, "sid", None)
            values = [sid] if sid else []
    return list(dict.fromkeys(str(sid) for sid in values if sid is not None))


def _entity_ids_for_sids(episode: Any, sids: list[str]) -> list[str]:
    graph_index = getattr(episode, "graph_index", None)
    getter = getattr(graph_index, "get_entities_for_sentence", None)
    if not callable(getter):
        return []
    entity_ids: set[str] = set()
    for sid in sids:
        for entity in getter(sid) or []:
            if isinstance(entity, Mapping):
                eid = entity.get("eid")
            else:
                eid = getattr(entity, "eid", entity)
            if eid is not None:
                entity_ids.add(str(eid))
    return sorted(entity_ids)


def _sorted_list(values: Any) -> list[Any]:
    if values is None:
        return []
    try:
        return sorted(list(values))
    except TypeError:
        return sorted(list(values), key=str)


def _json_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (set, frozenset)):
        return _sorted_list(values)
    return list(values)
