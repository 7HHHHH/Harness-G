"""Trainer-side SNC token reward construction.

This module is intentionally model-agnostic.  It receives a scoring callable
instead of importing a torch model, verl worker, or rollout implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
import os
import re

import torch

from .snc import SncConfig, SncStep, compute_snc_credit
from .answer_scoring import normalize_answer_aliases

ScoreFn = Callable[[list[str], list[list[str]]], Sequence[float]]

_BASELINE_KEY = "baseline"


class MockScoreFn:
    """Deterministic prompt-to-float scorer for trainer tests."""

    def __init__(self, scores: Mapping[str, float] | None = None) -> None:
        self.scores = dict(scores or {})
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompts: list[str], golds: list[list[str]]) -> list[float]:
        prompts = list(prompts)
        golds = list(golds)
        self.calls.append({"prompts": prompts, "golds": golds})
        return [float(self.scores.get(prompt, self._fallback_score(prompt))) for prompt in prompts]

    @staticmethod
    def _fallback_score(prompt: str) -> float:
        return float(sum(ord(ch) for ch in prompt) % 1000) / 1000.0


def compute_snc_token_rewards(
    envs: Sequence[Any],
    responses: torch.Tensor,
    gold_answers: Sequence[Any],
    score_fn: ScoreFn,
    tokenizer: Any,
    cfg: SncConfig,
    *,
    igpo_reproduce: bool = False,
    query_end: str = "</query>",
    query_start: str = "<query>",
    placement: str | None = None,
) -> torch.Tensor:
    """Return per-token SNC rewards for final response tensors.

    Credits are placed according to ``placement`` (default reads
    ``placement`` argument, default ``"span"``):

    * ``"span"``: the step credit is distributed uniformly across the tokens of
      that step's ``<query>...</query>`` action block (span-sum == credit).
      This keeps SNC credit on model-generated action tokens only — it never
      lands on ``<knowledge>`` tool-return tokens — so the span-local advantage
      is not silently zeroed by ``loss_mask``.
    * ``"anchor"`` (legacy/debug): the full step credit is placed at the final
      token of the step's ``</query>`` subsequence.

    If a response has fewer query blocks than navigation steps, remaining
    non-zero credits fall back to the last non-pad token.
    """

    rewards, _ = compute_snc_token_rewards_with_diagnostics(
        envs,
        responses,
        gold_answers,
        score_fn,
        tokenizer,
        cfg,
        igpo_reproduce=igpo_reproduce,
        query_end=query_end,
        query_start=query_start,
        placement=placement,
    )
    return rewards


def compute_snc_token_rewards_with_diagnostics(
    envs: Sequence[Any],
    responses: torch.Tensor,
    gold_answers: Sequence[Any],
    score_fn: ScoreFn,
    tokenizer: Any,
    cfg: SncConfig,
    *,
    igpo_reproduce: bool = False,
    query_end: str = "</query>",
    query_start: str = "<query>",
    placement: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute SNC rewards and return lightweight diagnostics for tests."""

    _validate_batch_inputs(envs, responses, gold_answers)
    query_end_ids = _encode_query_end(tokenizer, query_end)
    query_start_ids = _encode_query_start(tokenizer, query_start)
    if placement is None:
        placement = "span"
    snc_payloads_by_sample = [_snc_steps_for_env(env) for env in envs]

    all_prompts, all_golds, request_index_by_ref, request_meta = _collect_score_requests(
        snc_payloads_by_sample, gold_answers
    )
    request_scores, score_fn_calls = _score_once(score_fn, all_prompts, all_golds)
    score_by_ref = {
        ref: request_scores[request_index]
        for ref, request_index in request_index_by_ref.items()
    }

    snc_steps_by_sample, credits_by_sample, structure_by_sample = _build_steps_and_credits(
        snc_payloads_by_sample,
        score_by_ref,
        cfg,
        igpo_reproduce=igpo_reproduce,
        gold_answers=gold_answers,
    )
    rewards, placement_diagnostics, skipped_credits, span_stats = _place_token_rewards(
        responses,
        credits_by_sample,
        query_end_ids,
        query_start_ids,
        getattr(tokenizer, "pad_token_id", None),
        placement=placement,
        tokenizer=tokenizer,
        query_start=query_start,
        query_end=query_end,
    )

    diagnostics = {
        "score_prompts": all_prompts,
        "score_golds": all_golds,
        "score_requests": request_meta,
        "score_values": request_scores,
        "score_fn_calls": score_fn_calls,
        "snc_steps": snc_steps_by_sample,
        "step_credits": credits_by_sample,
        "structure": structure_by_sample,
        "placements": placement_diagnostics,
        "skipped_credits": skipped_credits,
        "placement_mode": placement,
        "span_stats": span_stats,
    }
    return rewards, diagnostics


def _validate_batch_inputs(
    envs: Sequence[Any], responses: torch.Tensor, gold_answers: Sequence[Any]
) -> None:
    if not isinstance(responses, torch.Tensor):
        raise TypeError("responses must be a torch.Tensor")
    if responses.dim() != 2:
        raise ValueError("responses must have shape [batch, resp_len]")
    if len(envs) != responses.shape[0]:
        raise ValueError("envs length must match responses batch size")
    if len(gold_answers) != responses.shape[0]:
        raise ValueError("gold_answers length must match responses batch size")


def _snc_steps_for_env(env: Any) -> list[dict[str, Any] | None]:
    return list(getattr(env, "_snc_steps", None) or [])


def _encode_query_end(tokenizer: Any, query_end: str) -> list[int]:
    token_ids = tokenizer.encode(query_end, add_special_tokens=False)
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    token_ids = [int(token_id) for token_id in token_ids]
    if not token_ids:
        raise ValueError("query_end must encode to at least one token")
    return token_ids


def _encode_query_start(tokenizer: Any, query_start: str) -> list[int]:
    token_ids = tokenizer.encode(query_start, add_special_tokens=False)
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    token_ids = [int(token_id) for token_id in token_ids]
    if not token_ids:
        raise ValueError("query_start must encode to at least one token")
    return token_ids


def _collect_score_requests(
    snc_payloads_by_sample: Sequence[Sequence[dict[str, Any] | None]],
    gold_answers: Sequence[Any],
) -> tuple[
    list[str],
    list[list[str]],
    dict[tuple[int, int, str], int],
    list[dict[str, Any]],
]:
    all_prompts: list[str] = []
    all_golds: list[list[str]] = []
    request_meta: list[dict[str, Any]] = []
    request_index_by_sample_prompt: dict[tuple[int, str], int] = {}
    request_index_by_ref: dict[tuple[int, int, str], int] = {}

    def add_prompt(sample_index: int, prompt: str) -> int:
        sample_prompt_key = (sample_index, prompt)
        if sample_prompt_key not in request_index_by_sample_prompt:
            request_index_by_sample_prompt[sample_prompt_key] = len(all_prompts)
            all_prompts.append(prompt)
            aliases = normalize_answer_aliases(gold_answers[sample_index])
            if not aliases:
                raise ValueError(f"SNC sample {sample_index} has no valid answer aliases")
            all_golds.append(aliases)
            request_meta.append(
                {
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "gold_aliases": aliases,
                }
            )
        return request_index_by_sample_prompt[sample_prompt_key]

    for sample_index, snc_payloads in enumerate(snc_payloads_by_sample):
        for step_index, payload in enumerate(snc_payloads):
            if payload is None:
                continue

            taken = payload.get("taken") or {}
            taken_action_id = payload.get("taken_action_id", taken.get("action_id"))
            if taken_action_id is None:
                continue

            baseline_prompt = payload.get("baseline_prompt")
            if not isinstance(baseline_prompt, str):
                raise ValueError("SNC payload missing string baseline_prompt")
            request_index_by_ref[(sample_index, step_index, _BASELINE_KEY)] = add_prompt(
                sample_index, baseline_prompt
            )

            frontier = payload.get("frontier") or {}
            for action_id, candidate in frontier.items():
                answer_prompt = candidate.get("answer_prompt")
                if not isinstance(answer_prompt, str):
                    raise ValueError("SNC frontier candidate missing string answer_prompt")
                request_index_by_ref[(sample_index, step_index, str(action_id))] = add_prompt(
                    sample_index, answer_prompt
                )

            taken_ref = (sample_index, step_index, str(taken_action_id))
            if taken_ref not in request_index_by_ref:
                answer_prompt = taken.get("answer_prompt")
                if not isinstance(answer_prompt, str):
                    raise ValueError("SNC taken action missing string answer_prompt")
                request_index_by_ref[taken_ref] = add_prompt(sample_index, answer_prompt)

    return all_prompts, all_golds, request_index_by_ref, request_meta


def _score_once(
    score_fn: ScoreFn, all_prompts: list[str], all_golds: list[list[str]]
) -> tuple[list[float], int]:
    if not all_prompts:
        return [], 0

    raw_scores = score_fn(all_prompts, all_golds)
    if isinstance(raw_scores, torch.Tensor):
        raw_scores = raw_scores.detach().cpu().tolist()
    scores = [float(score) for score in raw_scores]
    if len(scores) != len(all_prompts):
        raise ValueError("score_fn must return one score per prompt")
    return scores, 1


def _is_harvest_step(payload: Mapping[str, Any], gold_answers: Sequence[str]) -> bool:
    """True when this step's SELECT (or V3 ANSWER_WITH) first brings the gold
    answer text into the selected-evidence context (answer in the post-select
    scoring prompt but not in the pre-select baseline prompt)."""
    from .text_utils import contains_any_answer

    taken = payload.get("taken") or {}
    if str(taken.get("action_type") or "").upper() not in {"SELECT", "ANSWER_WITH"}:
        return False
    answer_prompt = taken.get("answer_prompt")
    baseline_prompt = payload.get("baseline_prompt")
    if not isinstance(answer_prompt, str) or not isinstance(baseline_prompt, str):
        return False
    return contains_any_answer(answer_prompt, gold_answers) and not contains_any_answer(
        baseline_prompt, gold_answers
    )


def _build_steps_and_credits(
    snc_payloads_by_sample: Sequence[Sequence[dict[str, Any] | None]],
    score_by_ref: Mapping[tuple[int, int, str], float],
    cfg: SncConfig,
    *,
    igpo_reproduce: bool,
    gold_answers: Sequence[Any] = (),
) -> tuple[list[list[SncStep | None]], list[list[float]], list[dict[str, Any]]]:
    snc_steps_by_sample: list[list[SncStep | None]] = []
    credits_by_sample: list[list[float]] = []
    # Per-sample structural decomposition so callers can measure whether the
    # N-step *enabling* term (r_en, the SNC innovation) is actually active or
    # whether credit collapses onto the myopic frontier-relative term (r_fr).
    structure_by_sample: list[dict[str, Any]] = []
    harvest_bonus_steps = 0

    for sample_index, snc_payloads in enumerate(snc_payloads_by_sample):
        snc_steps_by_turn: list[SncStep | None] = [None] * len(snc_payloads)
        concrete_steps: list[SncStep] = []
        concrete_turn_indices: list[int] = []
        concrete_step_records: list[dict[str, Any]] = []

        for step_index, payload in enumerate(snc_payloads):
            if payload is None:
                continue

            taken = payload.get("taken") or {}
            taken_action_id = payload.get("taken_action_id", taken.get("action_id"))
            taken_key = None if taken_action_id is None else str(taken_action_id)
            if taken_key is None:
                continue

            baseline_ref = (sample_index, step_index, _BASELINE_KEY)
            if baseline_ref not in score_by_ref:
                raise ValueError("missing baseline score for SNC step")
            baseline_g = score_by_ref[baseline_ref]

            frontier_ig: dict[str, float] = {}
            frontier_scores: dict[str, float] = {}
            frontier = payload.get("frontier") or {}
            for action_id in frontier:
                action_key = str(action_id)
                action_ref = (sample_index, step_index, action_key)
                if action_ref not in score_by_ref:
                    raise ValueError("missing frontier score for SNC step")
                frontier_scores[action_key] = float(score_by_ref[action_ref])
                frontier_ig[action_key] = frontier_scores[action_key] - baseline_g

            taken_ref = (sample_index, step_index, taken_key)
            if taken_ref not in score_by_ref:
                raise ValueError("missing taken action score for SNC step")
            taken_g = float(score_by_ref[taken_ref])
            taken_ig = taken_g - baseline_g

            # Dependency-bearing entity IDs for the taken action. Both sources
            # now flow through the API payload (see build_snc_step_payload):
            #   - used_entity_ids: the entities a LOOKUP/EXPAND/BRIDGE queries;
            #   - surfaced_entity_ids: the entities in the sentences a
            #     SELECT/ANSWER_WITH surfaces (these are the enabling signal
            #     that lets r_en credit intermediate multi-hop steps whose own
            #     taken_ig is ~0).
            dependency_ids = set(taken.get("used_entity_ids") or ())
            dependency_ids.update(taken.get("surfaced_entity_ids") or ())

            snc_step = SncStep(
                taken_action_id=taken_key,
                taken_ig=taken_ig,
                frontier_ig=frontier_ig,
                surfaced_entity_ids=frozenset(dependency_ids),
                action_type=str(taken.get("action_type") or "").upper(),
                is_information_action=bool(taken.get("is_information_action", True)),
                produced_sids=frozenset(str(x) for x in (taken.get("produced_sids") or ())),
                consumed_sids=frozenset(str(x) for x in (taken.get("consumed_sids") or ())),
                produced_entity_ids=frozenset(
                    str(x) for x in (taken.get("produced_entity_ids") or ())
                ),
                used_entity_ids=frozenset(str(x) for x in (taken.get("used_entity_ids") or ())),
            )
            snc_steps_by_turn[step_index] = snc_step
            concrete_steps.append(snc_step)
            concrete_turn_indices.append(step_index)
            deadzone = max(float(getattr(cfg, "ig_deadzone", 0.0) or 0.0), 0.0)
            concrete_step_records.append(
                {
                    "step_index": step_index,
                    "taken_action_id": taken_key,
                    "action_type": snc_step.action_type,
                    "is_information_action": snc_step.is_information_action,
                    "baseline_g": float(baseline_g),
                    "taken_g": taken_g,
                    "taken_ig_raw": float(taken_ig),
                    "taken_ig": 0.0 if abs(taken_ig) < deadzone else float(taken_ig),
                    "frontier": {
                        key: {
                            "g": frontier_scores[key],
                            "ig_raw": float(value),
                            "ig": 0.0 if abs(value) < deadzone else float(value),
                        }
                        for key, value in frontier_ig.items()
                    },
                    "seen_sids_before": list(payload.get("seen_sids_before") or ()),
                    "new_sids": list(taken.get("new_sids") or ()),
                    "produced_sids": sorted(snc_step.produced_sids),
                    "consumed_sids": sorted(snc_step.consumed_sids),
                    "produced_entity_ids": sorted(snc_step.produced_entity_ids),
                    "used_entity_ids": sorted(snc_step.used_entity_ids),
                }
            )

        credits_by_turn = [0.0] * len(snc_payloads)
        sample_structure: dict[str, Any] = {
            "num_concrete_steps": len(concrete_steps),
            "num_dependency_edges": 0,
            "r_fr_abssum": 0.0,
            "r_en_abssum": 0.0,
            "r_total_abssum": 0.0,
            "mode": "igpo" if igpo_reproduce else "snc",
            "dependency_edges": [],
            "steps": concrete_step_records,
        }
        if concrete_steps:
            if igpo_reproduce:
                concrete_credits = [float(step.taken_ig) for step in concrete_steps]
                credit_result = None
            else:
                credit_result = compute_snc_credit(concrete_steps, cfg)
                concrete_credits = _coerce_float_list(
                    credit_result.r_total, "compute_snc_credit(...).r_total"
                )
            if len(concrete_credits) != len(concrete_steps):
                raise ValueError("SNC credit result length must match concrete step count")
            for turn_index, credit in zip(concrete_turn_indices, concrete_credits):
                credits_by_turn[turn_index] = float(credit)
            # Structure diagnostics — computed AFTER credit assignment and fully
            # isolated in its own guard, so a malformed credit_result can never
            # alter or skip the credit path above (true zero-behavioural-change).
            try:
                sample_structure["r_total_abssum"] = float(
                    sum(abs(float(c)) for c in concrete_credits)
                )
                if credit_result is not None:
                    sample_structure["num_dependency_edges"] = len(
                        getattr(credit_result, "dependency_edges", []) or []
                    )
                    sample_structure["r_fr_abssum"] = float(
                        sum(abs(float(x)) for x in (getattr(credit_result, "r_fr", []) or []))
                    )
                    sample_structure["r_en_abssum"] = float(
                        sum(abs(float(x)) for x in (getattr(credit_result, "r_en", []) or []))
                    )
                    _cr_diag = getattr(credit_result, "diagnostics", {}) or {}
                    sample_structure["dependency_edges"] = [
                        [int(a), int(b)]
                        for a, b in (getattr(credit_result, "dependency_edges", []) or [])
                    ]
                    for record, r_fr, r_en, r_total in zip(
                        concrete_step_records,
                        getattr(credit_result, "r_fr", []) or [],
                        getattr(credit_result, "r_en", []) or [],
                        getattr(credit_result, "r_total", []) or [],
                    ):
                        record["r_fr"] = float(r_fr)
                        record["r_en"] = float(r_en)
                        record["r_total"] = float(r_total)
                    sample_structure["fixc_safe"] = bool(_cr_diag.get("fixc_safe", False))
                    sample_structure["fixc_protected_steps"] = int(_cr_diag.get("fixc_protected_steps", 0))
            except Exception:
                pass

        harvest_bonus = float(getattr(cfg, "harvest_bonus", 0.0) or 0.0)
        if harvest_bonus > 0 and sample_index < len(gold_answers):
            gold = normalize_answer_aliases(gold_answers[sample_index])
            for step_index, payload in enumerate(snc_payloads):
                if payload is None:
                    continue
                if _is_harvest_step(payload, gold):
                    credits_by_turn[step_index] += harvest_bonus
                    harvest_bonus_steps += 1

        snc_steps_by_sample.append(snc_steps_by_turn)
        credits_by_sample.append(credits_by_turn)
        structure_by_sample.append(sample_structure)

    if float(getattr(cfg, "harvest_bonus", 0.0) or 0.0) > 0:
        print(
            f"[Harness-G SNC] harvest bonus: {harvest_bonus_steps} steps credited "
            f"(+{getattr(cfg, 'harvest_bonus', 0.0)} each) across {len(snc_payloads_by_sample)} samples"
        )

    return snc_steps_by_sample, credits_by_sample, structure_by_sample


def _coerce_float_list(values: Any, name: str) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    try:
        return [float(value) for value in values]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of floats") from exc


def _query_block_spans_from_decode(
    tokenizer: Any,
    token_ids: Sequence[int],
    query_start: str,
    query_end: str,
) -> list[tuple[int, int] | None] | None:
    """Find ``<query>...</query>`` block token spans by decoding + offset mapping.

    This is robust to token-boundary merging: the Qwen tokenizer merges ``<`` with
    the preceding char and ``>`` with the following char (e.g. ``<query>`` becomes
    ``['<', 'query', '>{"']``), so naive token-subsequence matching fails.  By
    decoding to text, locating the literal ``<query>``/``</query>`` substrings, and
    mapping char offsets back to token indices via ``return_offsets_mapping``, we
    recover the true block boundaries.

    Returns a list of ``(start_tok, end_tok)`` inclusive spans (or None entries for
    malformed blocks), or None if the tokenizer does not support decode/__call__
    with offset mapping (caller falls back to subsequence matching).
    """
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        return None
    offs = enc.get("offset_mapping")
    # Fetch without boolean `or`: a multi-element torch.Tensor is truthy-ambiguous
    # and would raise before the tensor-safe conversion below.
    enc_ids = enc.get("input_ids", None)
    if enc_ids is None:
        enc_ids = enc.get("ids", None)
    if offs is None or enc_ids is None:
        return None
    # Only trust the mapping when decode->re-tokenize reproduces the EXACT
    # token id sequence; equal length alone is insufficient (same length but
    # different ids would misalign char->token offsets). Falls back to
    # subsequence matching otherwise.
    enc_ids_list = enc_ids.detach().cpu().tolist() if isinstance(enc_ids, torch.Tensor) else list(enc_ids)
    if enc_ids_list != list(token_ids):
        return None

    starts = [m.start() for m in re.finditer(re.escape(query_start), text)]
    ends = [m.start() + len(query_end) for m in re.finditer(re.escape(query_end), text)]
    n = min(len(starts), len(ends))
    spans: list[tuple[int, int] | None] = []
    for k in range(n):
        ts = _char_to_tok_start(offs, starts[k])
        te = _char_to_tok_end(offs, ends[k])
        if ts is not None and te is not None and te >= ts:
            spans.append((ts, te))
        else:
            spans.append(None)
    return spans


def _char_to_tok_start(offs: Sequence[tuple[int, int]], char_pos: int) -> int | None:
    for i, (s, e) in enumerate(offs):
        if s == char_pos:
            return i
        if s < char_pos < e:
            return i
    return None


def _char_to_tok_end(offs: Sequence[tuple[int, int]], char_pos: int) -> int | None:
    """Token index whose span contains ``char_pos - 1`` (i.e. ``s < char_pos <= e``).

    A boundary-merging tokenizer (Qwen BPE) can merge ``</query>``'s ``>``
    with the following char into one token, so the end char offset of
    ``</query>`` may fall *inside* a token rather than on its end edge.
    Matching on ``char_pos - 1`` containment is robust to that: we return the
    token that holds the last real char of the block.
    """
    for i, (s, e) in enumerate(offs):
        if e == char_pos or s < char_pos <= e:
            return i
    return None


def _place_token_rewards(
    responses: torch.Tensor,
    credits_by_sample: Sequence[Sequence[float]],
    query_end_ids: Sequence[int],
    query_start_ids: Sequence[int],
    pad_token_id: int | None,
    *,
    placement: str = "span",
    tokenizer: Any = None,
    query_start: str = "<query>",
    query_end: str = "</query>",
) -> tuple[torch.Tensor, list[dict[str, Any]], int, dict[str, float]]:
    """Place per-step SNC credit onto response tokens.

    ``placement="span"`` (default): distribute step k's credit uniformly across
    the tokens of the k-th ``<query>...</query>`` action block so the span sums
    to the step credit.  This is per-token span-local credit (NOT strict
    action-level normalization — span-internal normalization is left for future
    work).  Credit only lands on model-generated action tokens, never on
    ``<knowledge>`` tool-return tokens or padding.

    ``placement="anchor"`` (legacy/debug): place the full step credit at the
    k-th ``</query>`` end token.

    Block boundaries are found by decode+offset_mapping when a tokenizer is
    supplied (robust to Qwen's boundary-merging tokenization); otherwise the
    caller-supplied ``query_start_ids``/``query_end_ids`` subsequences are used
    (unit-test path).
    """
    rewards = torch.zeros_like(responses, dtype=torch.float32)
    placement_diagnostics: list[dict[str, Any]] = []
    skipped_credits = 0
    mode = (placement or "span").strip().lower()
    if mode not in {"span", "anchor"}:
        mode = "span"

    span_placed = 0
    anchor_placed = 0
    span_lens: list[int] = []

    for sample_index, credits_by_turn in enumerate(credits_by_sample):
        token_ids = [int(token_id) for token_id in responses[sample_index].detach().cpu().tolist()]
        last_non_pad_index = _last_non_pad_index(token_ids, pad_token_id)

        # Prefer decode+offset_mapping (real tokenizer); fall back to subsequence.
        block_spans: list[tuple[int, int] | None] | None = None
        end_anchors: list[int | None] = []
        if tokenizer is not None:
            block_spans = _query_block_spans_from_decode(tokenizer, token_ids, query_start, query_end)
        if block_spans is None:
            qe = _query_end_anchor_indices(token_ids, query_end_ids, last_non_pad_index)
            qs = _query_start_anchor_indices(token_ids, query_start_ids, last_non_pad_index)
            block_spans = []
            for k in range(max(len(qs), len(qe))):
                s = qs[k] if k < len(qs) else None
                e = qe[k] if k < len(qe) else None
                block_spans.append((s, e) if (s is not None and e is not None and e >= s) else None)
            end_anchors = list(qe)
        else:
            end_anchors = [sp[1] if sp is not None else None for sp in block_spans]

        sample_placements: list[dict[str, Any]] = []

        for step_index, credit in enumerate(credits_by_turn):
            credit = float(credit)
            if credit == 0.0:
                continue

            placed = False
            if mode == "span" and step_index < len(block_spans):
                span = block_spans[step_index]
                if span is not None:
                    s, e = span
                    span_idx = [
                        i for i in range(s, e + 1)
                        if pad_token_id is None or token_ids[i] != pad_token_id
                    ]
                    if span_idx:
                        per_tok = credit / len(span_idx)
                        for i in span_idx:
                            rewards[sample_index, i] += per_tok
                        sample_placements.append(
                            {
                                "step_index": step_index,
                                "span_start": s,
                                "span_end": e,
                                "span_len": len(span_idx),
                                "credit": credit,
                                "reason": "span",
                            }
                        )
                        span_placed += 1
                        span_lens.append(len(span_idx))
                        placed = True

            if not placed:
                # anchor fallback (also the default for placement="anchor").
                # Use the </query> end token when available (even if the full
                # <query> start was missing); only fall back to last_non_pad when
                # there is no query-end anchor for this step.
                if step_index < len(end_anchors) and end_anchors[step_index] is not None:
                    anchor_index = end_anchors[step_index]
                    reason = "query_end"
                else:
                    anchor_index = last_non_pad_index
                    reason = "last_non_pad"

                if anchor_index is None:
                    skipped_credits += 1
                    sample_placements.append(
                        {
                            "step_index": step_index,
                            "anchor_index": None,
                            "credit": credit,
                            "reason": "skipped",
                        }
                    )
                    continue

                rewards[sample_index, anchor_index] += credit
                if mode == "anchor":
                    placed_reason = reason
                elif reason == "last_non_pad":
                    placed_reason = "span_fallback_last"
                else:
                    placed_reason = "span_fallback_anchor"
                sample_placements.append(
                    {
                        "step_index": step_index,
                        "anchor_index": anchor_index,
                        "credit": credit,
                        "reason": placed_reason,
                    }
                )
                anchor_placed += 1

        placement_diagnostics.append(
            {
                "block_spans": block_spans,
                "end_anchors": end_anchors,
                "last_non_pad_index": last_non_pad_index,
                "placements": sample_placements,
            }
        )

    total_placed = span_placed + anchor_placed
    span_stats = {
        "span_placed": span_placed,
        "anchor_placed": anchor_placed,
        "span_match_rate": (span_placed / total_placed) if total_placed > 0 else 0.0,
        "span_len_mean": (sum(span_lens) / len(span_lens)) if span_lens else 0.0,
        "skipped_credit_count": skipped_credits,
    }

    return rewards, placement_diagnostics, skipped_credits, span_stats


def _last_non_pad_index(token_ids: Sequence[int], pad_token_id: int | None) -> int | None:
    if not token_ids:
        return None
    if pad_token_id is None:
        return len(token_ids) - 1
    for index in range(len(token_ids) - 1, -1, -1):
        if token_ids[index] != pad_token_id:
            return index
    return None


def _query_end_anchor_indices(
    token_ids: Sequence[int],
    query_end_ids: Sequence[int],
    last_non_pad_index: int | None,
) -> list[int]:
    if last_non_pad_index is None:
        return []

    needle = list(query_end_ids)
    needle_len = len(needle)
    anchors: list[int] = []
    start = 0
    stop = last_non_pad_index - needle_len + 1
    while start <= stop:
        if list(token_ids[start : start + needle_len]) == needle:
            anchors.append(start + needle_len - 1)
            start += needle_len
        else:
            start += 1
    return anchors


def _query_start_anchor_indices(
    token_ids: Sequence[int],
    query_start_ids: Sequence[int],
    last_non_pad_index: int | None,
) -> list[int]:
    """Return the first-token index of each ``<query>`` occurrence."""
    if last_non_pad_index is None:
        return []

    needle = list(query_start_ids)
    needle_len = len(needle)
    anchors: list[int] = []
    start = 0
    stop = last_non_pad_index - needle_len + 1
    while start <= stop:
        if list(token_ids[start : start + needle_len]) == needle:
            anchors.append(start)
            start += needle_len
        else:
            start += 1
    return anchors
