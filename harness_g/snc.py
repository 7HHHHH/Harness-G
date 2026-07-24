"""Structural non-myopic credit assignment utilities.

This module is intentionally pure Python.  It only operates on scalar
information-gain values and lightweight step metadata supplied by callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Literal, Optional, Sequence, Tuple
import hashlib
import itertools
import math
import statistics


BaselineMode = Literal["mean", "max"]
DependencyRule = Literal["provenance", "shared_entity", "structural_edge", "bridge_chain"]
ComplementarityMode = Literal["propagate", "shapley_exact"]
WeightScheme = Literal["uniform"]


@dataclass(frozen=True)
class SncStep:
    taken_action_id: str
    taken_ig: float
    frontier_ig: Dict[str, float]
    surfaced_entity_ids: FrozenSet[str]
    action_type: str = ""
    is_information_action: bool = True
    produced_sids: FrozenSet[str] = frozenset()
    consumed_sids: FrozenSet[str] = frozenset()
    produced_entity_ids: FrozenSet[str] = frozenset()
    used_entity_ids: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class SncConfig:
    alpha: float = 1.0
    beta: float = 1.0
    baseline: BaselineMode = "mean"
    dependency_rule: DependencyRule = "provenance"
    complementarity_mode: ComplementarityMode = "propagate"
    weight_scheme: WeightScheme = "uniform"
    propagation_gamma: float = 1.0
    ig_deadzone: float = 1e-4
    shuffle_dependency_edges: bool = False
    coalition_value_fn: Optional[Callable[[FrozenSet[int]], float]] = None
    shapley_max_steps: int = 6
    # Per-step bonus added to the credit of a SELECT action that first brings
    # the gold answer text into selected evidence. The trajectory-level harvest
    # gate failed because GRPO gets no gradient when no rollout in a group
    # harvests; delivering the same signal as step-level credit attaches it to
    # the exact SELECT tokens instead. 0.0 = off.
    harvest_bonus: float = 0.0
    # Fix C-safe: when a step's enabling credit r_en is strictly positive, do
    # not let a negative myopic r_fr cancel it. Only the myopic term is
    # zeroed (effective_r_fr = 0) on protected steps; alpha/beta still apply.
    # r_en <= 0 steps keep their original (possibly negative) r_fr so that
    # genuinely useless/wrong actions are still penalised. Default off
    # preserves legacy credit composition.
    fixc_safe: bool = False


@dataclass(frozen=True)
class SncCreditResult:
    r_fr: List[float]
    r_en: List[float]
    r_total: List[float]
    dependency_edges: List[Tuple[int, int]]
    diagnostics: Dict[str, object]


def frontier_relative_credit(step: SncStep, cfg: SncConfig) -> float:
    """Return the taken action's frontier-relative immediate credit."""
    if not step.is_information_action:
        return 0.0
    other_igs = [
        _apply_deadzone(float(ig), cfg.ig_deadzone)
        for action_id, ig in step.frontier_ig.items()
        if action_id != step.taken_action_id
    ]
    if not other_igs:
        return 0.0

    if cfg.baseline == "mean":
        baseline = statistics.mean(other_igs)
    elif cfg.baseline == "max":
        baseline = max(other_igs)
    else:
        raise ValueError("baseline must be 'mean' or 'max'")

    taken_ig = _apply_deadzone(float(step.taken_ig), cfg.ig_deadzone)
    return _apply_deadzone(taken_ig - float(baseline), cfg.ig_deadzone)


def build_dependency_edges(
    steps: Sequence[SncStep], cfg: SncConfig
) -> List[Tuple[int, int]]:
    """Return edges ``(t, u)`` where later step ``u`` depends on step ``t``."""
    if cfg.dependency_rule == "provenance":
        edges: set[Tuple[int, int]] = set()
        for u, current_step in enumerate(steps):
            # Direct provenance only: a consumed SID/entity depends on its most
            # recent producer. This avoids transitive shortcut edges; recursive
            # propagation is responsible for carrying credit further back.
            for sid in current_step.consumed_sids:
                producer = _latest_producer(steps, u, sid, "produced_sids")
                if producer is not None:
                    edges.add((producer, u))
            for eid in current_step.used_entity_ids:
                producer = _latest_producer(steps, u, eid, "produced_entity_ids")
                if producer is not None:
                    edges.add((producer, u))
        return sorted(edges)

    if cfg.dependency_rule == "shared_entity":
        edges: List[Tuple[int, int]] = []
        for t, previous_step in enumerate(steps):
            previous_entities = set(previous_step.surfaced_entity_ids)
            if not previous_entities:
                continue

            for u in range(t + 1, len(steps)):
                # M1 has no separate "entities used by action u" field.  Use
                # u.surfaced_entity_ids as the proxy for involved entities.
                current_entities = set(steps[u].surfaced_entity_ids)
                if previous_entities.intersection(current_entities):
                    edges.append((t, u))
        return edges

    if cfg.dependency_rule == "structural_edge":
        raise NotImplementedError(
            "dependency_rule='structural_edge' requires graph-structural "
            "signals that are not part of pure M1."
        )

    if cfg.dependency_rule == "bridge_chain":
        raise NotImplementedError(
            "dependency_rule='bridge_chain' requires bridge-chain metadata "
            "that is not part of pure M1."
        )

    raise ValueError(
        "dependency_rule must be 'provenance', 'shared_entity', "
        "'structural_edge', or 'bridge_chain'"
    )


def complementarity_credit(
    steps: Sequence[SncStep],
    igs: Sequence[float],
    edges: Sequence[Tuple[int, int]],
    cfg: SncConfig,
) -> List[float]:
    """Return enabling/complementarity credit for each step."""
    if len(steps) != len(igs):
        raise ValueError("steps and igs must have the same length")

    if cfg.complementarity_mode == "propagate":
        return _propagate_credit(steps, igs, edges, cfg)

    if cfg.complementarity_mode == "shapley_exact":
        if cfg.coalition_value_fn is None:
            raise ValueError("shapley_exact requires coalition_value_fn; refusing silent fallback")
        if len(steps) > cfg.shapley_max_steps:
            raise ValueError(
                f"shapley_exact supports at most {cfg.shapley_max_steps} steps, got {len(steps)}"
            )
        return _shapley_exact_credit(len(steps), cfg.coalition_value_fn)

    raise ValueError("complementarity_mode must be 'propagate' or 'shapley_exact'")


def compute_snc_credit(steps: Sequence[SncStep], cfg: SncConfig) -> SncCreditResult:
    """Compute SNC credit vectors and lightweight diagnostics for an episode."""
    r_fr = [frontier_relative_credit(step, cfg) for step in steps]
    igs = [_apply_deadzone(float(step.taken_ig), cfg.ig_deadzone) for step in steps]
    dependency_edges = build_dependency_edges(steps, cfg)
    if cfg.shuffle_dependency_edges:
        dependency_edges = _shuffle_edges(steps, dependency_edges)
    r_en = complementarity_credit(steps, igs, dependency_edges, cfg)

    # Fix C-safe: protect enabling steps from being cancelled by a negative
    # myopic r_fr. Only applied when r_en is *strictly* positive; r_en <= 0
    # steps keep their original (possibly negative) r_fr so useless/wrong
    # actions remain penalised. alpha/beta still apply to the (possibly
    # zeroed) effective r_fr and to r_en unchanged.
    fixc_protected_steps = 0
    r_total = []
    for immediate, enabling in zip(r_fr, r_en):
        if cfg.fixc_safe and enabling > cfg.ig_deadzone and immediate < 0.0:
            effective_r_fr = 0.0
            fixc_protected_steps += 1
        else:
            effective_r_fr = immediate
        r_total.append(
            float(cfg.alpha) * effective_r_fr + float(cfg.beta) * enabling
        )

    diagnostics = {
        "baseline": cfg.baseline,
        "num_steps": len(steps),
        "num_dependency_edges": len(dependency_edges),
        "dependency_rule": cfg.dependency_rule,
        "complementarity_mode_requested": cfg.complementarity_mode,
        "complementarity_mode_effective": _effective_complementarity_mode(steps, cfg),
        "weight_scheme": cfg.weight_scheme,
        "propagation_gamma": float(cfg.propagation_gamma),
        "ig_deadzone": float(cfg.ig_deadzone),
        "shuffle_dependency_edges": bool(cfg.shuffle_dependency_edges),
        "fixc_safe": bool(cfg.fixc_safe),
        "fixc_protected_steps": fixc_protected_steps,
    }

    return SncCreditResult(
        r_fr=r_fr,
        r_en=r_en,
        r_total=r_total,
        dependency_edges=dependency_edges,
        diagnostics=diagnostics,
    )


def _propagate_credit(
    steps: Sequence[SncStep],
    igs: Sequence[float],
    edges: Sequence[Tuple[int, int]],
    cfg: SncConfig,
) -> List[float]:
    if cfg.weight_scheme != "uniform":
        raise ValueError("weight_scheme must be 'uniform'")

    predecessors_by_u: List[List[int]] = [[] for _ in steps]
    for t, u in edges:
        if t < 0 or u < 0 or t >= len(steps) or u >= len(steps):
            raise ValueError("dependency edge index out of range")
        if t >= u:
            raise ValueError("dependency edges must satisfy t < u")
        predecessors_by_u[u].append(t)

    r_en = [0.0 for _ in steps]
    gamma = float(cfg.propagation_gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("propagation_gamma must be in [0, 1]")
    for u in range(len(steps) - 1, -1, -1):
        predecessors = predecessors_by_u[u]
        if not predecessors:
            continue
        downstream_return = float(igs[u]) + gamma * float(r_en[u])
        share = downstream_return / float(len(predecessors))
        for t in predecessors:
            r_en[t] += share

    return r_en


def _shapley_exact_credit(
    num_steps: int, value_fn: Callable[[FrozenSet[int]], float]
) -> List[float]:
    if num_steps == 0:
        return []

    all_indices = tuple(range(num_steps))
    factorial_n = float(math.factorial(num_steps))
    credits = [0.0 for _ in all_indices]

    for i in all_indices:
        others = tuple(j for j in all_indices if j != i)
        for coalition_size in range(num_steps):
            weight = (
                math.factorial(coalition_size)
                * math.factorial(num_steps - coalition_size - 1)
                / factorial_n
            )
            for coalition_tuple in itertools.combinations(others, coalition_size):
                coalition = frozenset(coalition_tuple)
                with_i = frozenset(coalition_tuple + (i,))
                marginal = float(value_fn(with_i)) - float(value_fn(coalition))
                credits[i] += weight * marginal

    return credits


def _effective_complementarity_mode(
    steps: Sequence[SncStep], cfg: SncConfig
) -> str:
    if cfg.complementarity_mode != "shapley_exact":
        return cfg.complementarity_mode
    return "shapley_exact"


def _latest_producer(
    steps: Sequence[SncStep], before: int, value: str, field: str
) -> int | None:
    for index in range(before - 1, -1, -1):
        if value in getattr(steps[index], field):
            return index
    return None


def _apply_deadzone(value: float, threshold: float) -> float:
    threshold = max(float(threshold), 0.0)
    return 0.0 if abs(value) < threshold else float(value)


def _shuffle_edges(
    steps: Sequence[SncStep], edges: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Deterministically permute predecessors while preserving each indegree."""

    by_target: dict[int, list[int]] = {}
    for source, target in edges:
        by_target.setdefault(target, []).append(source)
    identity = "|".join(step.taken_action_id for step in steps).encode("utf-8")
    shuffled: list[Tuple[int, int]] = []
    for target, true_sources in sorted(by_target.items()):
        candidates = sorted(
            range(target),
            key=lambda source: hashlib.sha256(
                identity + f"|{target}|{source}".encode("ascii")
            ).digest(),
        )
        count = min(len(true_sources), len(candidates))
        if count == len(candidates):
            sampled = candidates
        else:
            true_set = set(true_sources)
            sampled = candidates[:count]
            for offset in range(1, len(candidates)):
                if set(sampled) != true_set:
                    break
                sampled = [
                    candidates[(offset + index) % len(candidates)]
                    for index in range(count)
                ]
        shuffled.extend((source, target) for source in sorted(sampled))
    return shuffled
