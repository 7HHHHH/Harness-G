import math

from harness_g.snc import (
    SncConfig,
    SncStep,
    build_dependency_edges,
    complementarity_credit,
    compute_snc_credit,
    frontier_relative_credit,
)


def test_frontier_relative_equal_candidates_is_zero():
    step = SncStep(
        taken_action_id="a",
        taken_ig=1.0,
        frontier_ig={"a": 1.0, "b": 1.0},
        surfaced_entity_ids=frozenset(),
    )

    credit = frontier_relative_credit(step, SncConfig(baseline="mean"))

    assert math.isclose(credit, 0.0)


def test_frontier_relative_uniquely_high_taken_candidate_is_positive():
    step = SncStep(
        taken_action_id="a",
        taken_ig=3.0,
        frontier_ig={"a": 3.0, "b": 1.0, "c": 1.0},
        surfaced_entity_ids=frozenset(),
    )

    credit = frontier_relative_credit(step, SncConfig(baseline="mean"))

    assert credit > 0.0


def test_dependency_propagation_credits_zero_ig_enabling_step():
    steps = [
        SncStep(
            taken_action_id="surface_x",
            taken_ig=0.0,
            frontier_ig={"surface_x": 0.0, "other": 0.0},
            surfaced_entity_ids=frozenset({"x"}),
            produced_entity_ids=frozenset({"x"}),
        ),
        SncStep(
            taken_action_id="use_x",
            taken_ig=5.0,
            frontier_ig={"use_x": 5.0, "other": 1.0},
            surfaced_entity_ids=frozenset({"x"}),
            used_entity_ids=frozenset({"x"}),
        ),
    ]

    result = compute_snc_credit(steps, SncConfig(complementarity_mode="propagate"))

    assert result.dependency_edges == [(0, 1)]
    assert result.r_en[0] > 0.0
    assert math.isclose(result.r_en[1], 0.0)


def test_shapley_exact_matches_propagate_and_rejects_missing_value_fn():
    steps = [
        SncStep(
            taken_action_id="surface_x",
            taken_ig=0.0,
            frontier_ig={"surface_x": 0.0},
            surfaced_entity_ids=frozenset({"x"}),
            produced_entity_ids=frozenset({"x"}),
        ),
        SncStep(
            taken_action_id="use_x",
            taken_ig=4.0,
            frontier_ig={"use_x": 4.0},
            surfaced_entity_ids=frozenset({"x"}),
            used_entity_ids=frozenset({"x"}),
        ),
    ]
    igs = [0.0, 4.0]
    edges = [(0, 1)]
    propagate_cfg = SncConfig(complementarity_mode="propagate")
    propagated = complementarity_credit(steps, igs, edges, propagate_cfg)

    additive_weights = {0: propagated[0], 1: propagated[1]}

    def value_fn(coalition):
        return sum(additive_weights[i] for i in coalition)

    shapley_cfg = SncConfig(
        complementarity_mode="shapley_exact",
        coalition_value_fn=value_fn,
    )
    shapley = complementarity_credit(steps, igs, edges, shapley_cfg)

    assert all(math.isclose(a, b) for a, b in zip(shapley, propagated))

    import pytest

    with pytest.raises(ValueError, match="coalition_value_fn"):
        compute_snc_credit(steps, SncConfig(complementarity_mode="shapley_exact"))


def test_shared_entity_dependency_edges_on_handmade_example():
    steps = [
        SncStep(
            taken_action_id="a",
            taken_ig=1.0,
            frontier_ig={"a": 1.0},
            surfaced_entity_ids=frozenset({"x", "y"}),
        ),
        SncStep(
            taken_action_id="b",
            taken_ig=1.0,
            frontier_ig={"b": 1.0},
            surfaced_entity_ids=frozenset({"z"}),
        ),
        SncStep(
            taken_action_id="c",
            taken_ig=1.0,
            frontier_ig={"c": 1.0},
            surfaced_entity_ids=frozenset({"x"}),
            produced_entity_ids=frozenset({"x"}),
        ),
        SncStep(
            taken_action_id="d",
            taken_ig=1.0,
            frontier_ig={"d": 1.0},
            surfaced_entity_ids=frozenset({"z", "q"}),
        ),
    ]

    edges = build_dependency_edges(steps, SncConfig(dependency_rule="shared_entity"))

    assert edges == [(0, 2), (1, 3)]


def test_commit_action_has_no_immediate_frontier_credit():
    step = SncStep(
        taken_action_id="select",
        taken_ig=0.5,
        frontier_ig={"lookup": 0.9},
        surfaced_entity_ids=frozenset(),
        action_type="SELECT",
        is_information_action=False,
    )

    assert frontier_relative_credit(step, SncConfig()) == 0.0


def test_provenance_uses_latest_direct_producer():
    steps = [
        SncStep("lookup", 0.0, {}, frozenset(), produced_sids=frozenset({"S1"})),
        SncStep(
            "select",
            0.0,
            {},
            frozenset(),
            produced_sids=frozenset({"S1"}),
            consumed_sids=frozenset({"S1"}),
        ),
        SncStep("next_lookup", 1.0, {}, frozenset(), consumed_sids=frozenset({"S1"})),
    ]

    assert build_dependency_edges(steps, SncConfig()) == [(0, 1), (1, 2)]


def test_recursive_propagation_reaches_earliest_enabler():
    steps = [
        SncStep("t0", 0.0, {}, frozenset(), produced_sids=frozenset({"S1"})),
        SncStep(
            "t1",
            0.0,
            {},
            frozenset(),
            produced_sids=frozenset({"S2"}),
            consumed_sids=frozenset({"S1"}),
        ),
        SncStep("t2", 1.0, {}, frozenset(), consumed_sids=frozenset({"S2"})),
    ]

    result = compute_snc_credit(steps, SncConfig(alpha=0.0, beta=1.0))

    assert result.dependency_edges == [(0, 1), (1, 2)]
    assert result.r_en == [1.0, 1.0, 0.0]


def test_ig_deadzone_removes_subthreshold_deltas():
    step = SncStep("lookup", 5e-5, {"other": -5e-5}, frozenset())

    assert frontier_relative_credit(step, SncConfig(ig_deadzone=1e-4)) == 0.0


def test_ig_deadzone_removes_tiny_frontier_relative_difference():
    step = SncStep("lookup", 1.1e-4, {"other": 1.0e-4}, frozenset())

    assert frontier_relative_credit(step, SncConfig(ig_deadzone=1e-4)) == 0.0


# ---------------------------------------------------------------------------
# Fix C-safe: protect enabling steps (r_en > 0) from negative myopic r_fr.
# ---------------------------------------------------------------------------

def _two_step_enabling(steps_r_en_positive=True):
    """Step 0 is the enabler (surfaced x), step 1 uses x with high IG.

    r_en[0] > 0 (it enabled step 1's gain); r_en[1] == 0.
    Step 0's frontier_relative r_fr is negative (taken_ig below frontier mean)
    to exercise the cancellation-protection path.
    """
    return [
        SncStep(
            taken_action_id="surface_x",
            taken_ig=0.2,
            frontier_ig={"surface_x": 0.2, "other": 1.0},
            surfaced_entity_ids=frozenset({"x"}),
            produced_entity_ids=frozenset({"x"}),
        ),
        SncStep(
            taken_action_id="use_x",
            taken_ig=5.0,
            frontier_ig={"use_x": 5.0, "other": 1.0},
            surfaced_entity_ids=frozenset({"x"}),
            used_entity_ids=frozenset({"x"}),
        ),
    ]


def test_fixc_safe_off_preserves_legacy_composition():
    steps = _two_step_enabling()

    legacy = compute_snc_credit(steps, SncConfig(fixc_safe=False))
    fixc = compute_snc_credit(steps, SncConfig(fixc_safe=False))

    assert legacy.r_total == fixc.r_total
    # legacy: alpha*r_fr + beta*r_en for each step
    expected = [
        1.0 * legacy.r_fr[i] + 1.0 * legacy.r_en[i] for i in range(len(steps))
    ]
    assert all(math.isclose(a, b) for a, b in zip(legacy.r_total, expected))
    assert legacy.diagnostics["fixc_safe"] is False
    assert legacy.diagnostics["fixc_protected_steps"] == 0


def test_fixc_safe_on_protects_enabling_step_with_negative_r_fr():
    steps = _two_step_enabling()
    cfg = SncConfig(fixc_safe=True)

    result = compute_snc_credit(steps, cfg)

    # step 0: enabler, r_en > 0 and r_fr < 0 -> protected (r_fr zeroed)
    assert result.r_en[0] > 0.0
    assert result.r_fr[0] < 0.0
    assert math.isclose(result.r_total[0], 1.0 * 0.0 + 1.0 * result.r_en[0])
    # step 1: r_en == 0 -> NOT protected, keeps original r_fr
    assert math.isclose(result.r_en[1], 0.0)
    assert math.isclose(result.r_total[1], 1.0 * result.r_fr[1] + 1.0 * result.r_en[1])
    assert result.diagnostics["fixc_safe"] is True
    assert result.diagnostics["fixc_protected_steps"] == 1


def test_fixc_safe_does_not_protect_when_r_en_is_zero():
    # No dependency edge -> r_en all zero -> nothing protected even with fixc on.
    steps = [
        SncStep(
            taken_action_id="a",
            taken_ig=0.1,
            frontier_ig={"a": 0.1, "b": 1.0},  # r_fr negative
            surfaced_entity_ids=frozenset(),  # no entities -> no edges
        ),
        SncStep(
            taken_action_id="b",
            taken_ig=0.1,
            frontier_ig={"b": 0.1, "a": 1.0},  # r_fr negative
            surfaced_entity_ids=frozenset(),
        ),
    ]

    legacy = compute_snc_credit(steps, SncConfig(fixc_safe=False))
    fixc = compute_snc_credit(steps, SncConfig(fixc_safe=True))

    assert all(math.isclose(a, 0.0) for a in fixc.r_en)
    # r_en == 0 (not strictly > 0) -> negative r_fr preserved
    assert fixc.r_total == legacy.r_total
    assert fixc.diagnostics["fixc_protected_steps"] == 0
    assert all(rf < 0.0 for rf in fixc.r_fr)


def test_fixc_safe_does_not_protect_when_r_fr_nonnegative():
    # Enabling step with POSITIVE r_fr: protection condition (r_fr < 0) is
    # false -> effective_r_fr == r_fr, total unchanged regardless of fixc.
    steps = [
        SncStep(
            taken_action_id="surface_x",
            taken_ig=3.0,
            frontier_ig={"surface_x": 3.0, "other": 1.0},  # r_fr positive
            surfaced_entity_ids=frozenset({"x"}),
            produced_entity_ids=frozenset({"x"}),
        ),
        SncStep(
            taken_action_id="use_x",
            taken_ig=5.0,
            frontier_ig={"use_x": 5.0, "other": 1.0},
            surfaced_entity_ids=frozenset({"x"}),
            used_entity_ids=frozenset({"x"}),
        ),
    ]

    legacy = compute_snc_credit(steps, SncConfig(fixc_safe=False))
    fixc = compute_snc_credit(steps, SncConfig(fixc_safe=True))

    assert fixc.r_total == legacy.r_total
    assert fixc.r_fr[0] > 0.0
    assert fixc.diagnostics["fixc_protected_steps"] == 0


def test_fixc_safe_respects_alpha_beta_scaling():
    # alpha != beta: protection zeroes r_fr, then alpha*0 + beta*r_en.
    # Non-protected step keeps alpha*r_fr + beta*r_en.
    steps = _two_step_enabling()
    cfg = SncConfig(alpha=0.5, beta=2.0, fixc_safe=True)

    result = compute_snc_credit(steps, cfg)

    # step 0 protected: alpha*0 + beta*r_en[0]
    assert math.isclose(result.r_total[0], 0.5 * 0.0 + 2.0 * result.r_en[0])
    # step 1 not protected: alpha*r_fr[1] + beta*r_en[1]
    assert math.isclose(
        result.r_total[1], 0.5 * result.r_fr[1] + 2.0 * result.r_en[1]
    )
    assert result.diagnostics["fixc_protected_steps"] == 1
