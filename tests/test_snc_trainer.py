from types import SimpleNamespace
import os

import pytest
import torch

from harness_g import snc_trainer
from harness_g.snc_trainer import MockScoreFn


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "</query>":
            return [7, 8]
        if text == "<query>":
            return [6, 7]
        raise AssertionError(f"unexpected encode text: {text!r}")


def _candidate(prompt, *, used_entity_ids=None, surfaced_sids=None, surfaced_entity_ids=None):
    return {
        "surfaced_sids": list(surfaced_sids or []),
        "surfaced_entity_ids": list(surfaced_entity_ids or []),
        "used_entity_ids": list(used_entity_ids or []),
        "answer_prompt": prompt,
    }


def _fake_envs():
    sample0_step0 = {
        "selected_sids_before": [],
        "baseline_prompt": "s0_base",
        "taken_action_id": "a0",
        "frontier": {
            "a0": _candidate(
                "s0_a0", used_entity_ids=["E0"], surfaced_sids=["S0"]
            ),
            "a1": _candidate(
                "s0_a1", used_entity_ids=["E1"], surfaced_sids=["S1"]
            ),
        },
        "taken": {
            "action_id": "a0",
            "action_type": "query",
            "surfaced_sids": ["S0"],
            "used_entity_ids": ["E0"],
            "answer_prompt": "s0_a0",
        },
    }
    sample1_step0 = {
        "selected_sids_before": ["S9"],
        "baseline_prompt": "s1_base0",
        "taken_action_id": "b0",
        "frontier": {
            "b0": _candidate("shared_prompt", used_entity_ids=["E2"]),
            "b1": _candidate("shared_prompt", used_entity_ids=["E3"]),
        },
        "taken": {
            "action_id": "b0",
            "action_type": "query",
            "surfaced_sids": ["S2"],
            "used_entity_ids": ["E2"],
            "answer_prompt": "shared_prompt",
        },
    }
    sample1_step1 = {
        "selected_sids_before": ["S9", "S2"],
        "baseline_prompt": "s1_base1",
        "taken_action_id": "c0",
        "frontier": {
            "c0": _candidate("s1_c0", used_entity_ids=["E4"]),
            "c1": _candidate("s1_c1", used_entity_ids=["E5"]),
        },
        "taken": {
            "action_id": "c0",
            "action_type": "query",
            "surfaced_sids": ["S4"],
            "used_entity_ids": ["E4"],
            "answer_prompt": "s1_c0",
        },
    }
    return [
        SimpleNamespace(
            _snc_steps=[sample0_step0, None],
            _actions=["<query>one</query>", "<query>ignored</query>"],
        ),
        SimpleNamespace(
            _snc_steps=[sample1_step0, sample1_step1],
            _actions=["<query>two</query>", "<query>three</query>"],
        ),
    ]


def _scores():
    return {
        "s0_base": 0.5,
        "s0_a0": 2.0,
        "s0_a1": 0.75,
        "s1_base0": 1.0,
        "shared_prompt": 4.0,
        "s1_base1": 10.0,
        "s1_c0": 6.0,
        "s1_c1": 15.0,
    }


def test_batches_dedupes_scores_and_places_igpo_credit():
    envs = _fake_envs()
    responses = torch.tensor(
        [
            [11, 7, 8, 12, 7, 8, 13, 0, 0],
            [21, 7, 8, 22, 23, 7, 8, 24, 0],
        ],
        dtype=torch.long,
    )
    score_fn = MockScoreFn(_scores())

    rewards, diagnostics = snc_trainer.compute_snc_token_rewards_with_diagnostics(
        envs,
        responses,
        ["gold0", "gold1"],
        score_fn,
        FakeTokenizer(),
        cfg=object(),
        igpo_reproduce=True,
        placement="anchor",
    )

    assert diagnostics["score_fn_calls"] == 1
    assert len(score_fn.calls) == 1
    assert score_fn.calls[0]["prompts"] == [
        "s0_base",
        "s0_a0",
        "s0_a1",
        "s1_base0",
        "shared_prompt",
        "s1_base1",
        "s1_c0",
        "s1_c1",
    ]
    assert score_fn.calls[0]["golds"] == [
        ["gold0"],
        ["gold0"],
        ["gold0"],
        ["gold1"],
        ["gold1"],
        ["gold1"],
        ["gold1"],
        ["gold1"],
    ]

    step00 = diagnostics["snc_steps"][0][0]
    assert step00.taken_action_id == "a0"
    assert step00.taken_ig == pytest.approx(1.5)
    assert step00.frontier_ig["a0"] == pytest.approx(1.5)
    assert step00.frontier_ig["a1"] == pytest.approx(0.25)
    assert step00.surfaced_entity_ids == frozenset({"E0"})
    assert diagnostics["snc_steps"][0][1] is None

    step10 = diagnostics["snc_steps"][1][0]
    assert step10.taken_ig == pytest.approx(3.0)
    assert step10.frontier_ig["b0"] == pytest.approx(3.0)
    assert step10.frontier_ig["b1"] == pytest.approx(3.0)

    step11 = diagnostics["snc_steps"][1][1]
    assert step11.taken_ig == pytest.approx(-4.0)
    assert step11.frontier_ig["c0"] == pytest.approx(-4.0)
    assert step11.frontier_ig["c1"] == pytest.approx(5.0)

    expected = torch.zeros_like(responses, dtype=torch.float32)
    expected[0, 2] = 1.5
    expected[1, 2] = 3.0
    expected[1, 6] = -4.0
    torch.testing.assert_close(rewards, expected)


def test_default_path_uses_m1_credit_and_preserves_turn_alignment(monkeypatch):
    envs = _fake_envs()
    responses = torch.tensor(
        [
            [11, 7, 8, 12, 7, 8, 13, 0, 0],
            [21, 7, 8, 22, 23, 7, 8, 24, 0],
        ],
        dtype=torch.long,
    )
    score_fn = MockScoreFn(_scores())
    credit_calls = []
    cfg = object()

    def fake_compute_snc_credit(steps, received_cfg):
        credit_calls.append((steps, received_cfg))
        if len(credit_calls) == 1:
            return SimpleNamespace(r_total=[7.0])
        return SimpleNamespace(r_total=[8.0, 9.0])

    monkeypatch.setattr(snc_trainer, "compute_snc_credit", fake_compute_snc_credit)

    rewards, diagnostics = snc_trainer.compute_snc_token_rewards_with_diagnostics(
        envs,
        responses,
        ["gold0", "gold1"],
        score_fn,
        FakeTokenizer(),
        cfg=cfg,
        placement="anchor",
    )

    assert len(credit_calls) == 2
    assert credit_calls[0][1] is cfg
    assert credit_calls[0][0][0].taken_ig == pytest.approx(1.5)
    assert credit_calls[1][0][0].taken_ig == pytest.approx(3.0)
    assert credit_calls[1][0][1].taken_ig == pytest.approx(-4.0)
    assert diagnostics["step_credits"] == [[7.0, 0.0], [8.0, 9.0]]

    expected = torch.zeros_like(responses, dtype=torch.float32)
    expected[0, 2] = 7.0
    expected[1, 2] = 8.0
    expected[1, 6] = 9.0
    torch.testing.assert_close(rewards, expected)


def test_invalid_payload_without_taken_action_id_is_skipped():
    invalid_step = {
        "selected_sids_before": [],
        "baseline_prompt": "invalid_base",
        "taken_action_id": None,
        "frontier": {
            "a0": _candidate("candidate_prompt", used_entity_ids=["E0"]),
        },
        "taken": {
            "action_id": None,
            "action_type": None,
        },
    }
    envs = [
        SimpleNamespace(
            _snc_steps=[invalid_step],
            _actions=["<query>A0 || invalid</query>"],
        )
    ]
    responses = torch.tensor([[11, 7, 8, 0]], dtype=torch.long)
    score_fn = MockScoreFn({"invalid_base": 10.0, "candidate_prompt": 20.0})

    rewards, diagnostics = snc_trainer.compute_snc_token_rewards_with_diagnostics(
        envs,
        responses,
        ["gold"],
        score_fn,
        FakeTokenizer(),
        cfg=object(),
        placement="anchor",
    )

    assert diagnostics["score_fn_calls"] == 0
    assert diagnostics["score_prompts"] == []
    assert diagnostics["snc_steps"] == [[None]]
    assert diagnostics["step_credits"] == [[0.0]]
    assert score_fn.calls == []
    torch.testing.assert_close(rewards, torch.zeros_like(responses, dtype=torch.float32))


def test_falls_back_to_last_non_pad_and_counts_unplaced_credit():
    placed_step0 = {
        "selected_sids_before": [],
        "baseline_prompt": "p_base0",
        "taken_action_id": "p0",
        "frontier": {"p0": _candidate("p_take0", used_entity_ids=["E0"])},
        "taken": {
            "action_id": "p0",
            "action_type": "query",
            "surfaced_sids": [],
            "used_entity_ids": ["E0"],
            "answer_prompt": "p_take0",
        },
    }
    placed_step1 = {
        "selected_sids_before": [],
        "baseline_prompt": "p_base1",
        "taken_action_id": "p1",
        "frontier": {"p1": _candidate("p_take1", used_entity_ids=["E1"])},
        "taken": {
            "action_id": "p1",
            "action_type": "query",
            "surfaced_sids": [],
            "used_entity_ids": ["E1"],
            "answer_prompt": "p_take1",
        },
    }
    skipped_step = {
        "selected_sids_before": [],
        "baseline_prompt": "s_base",
        "taken_action_id": "s0",
        "frontier": {"s0": _candidate("s_take", used_entity_ids=["E2"])},
        "taken": {
            "action_id": "s0",
            "action_type": "query",
            "surfaced_sids": [],
            "used_entity_ids": ["E2"],
            "answer_prompt": "s_take",
        },
    }
    envs = [
        SimpleNamespace(
            _snc_steps=[placed_step0, placed_step1],
            _actions=["<query>one</query>", "<query>two</query>"],
        ),
        SimpleNamespace(_snc_steps=[skipped_step], _actions=["<query>skip</query>"]),
    ]
    responses = torch.tensor(
        [
            [31, 7, 8, 32, 33, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    score_fn = MockScoreFn(
        {
            "p_base0": 1.0,
            "p_take0": 3.0,
            "p_base1": 10.0,
            "p_take1": 14.0,
            "s_base": 5.0,
            "s_take": 6.0,
        }
    )

    rewards, diagnostics = snc_trainer.compute_snc_token_rewards_with_diagnostics(
        envs,
        responses,
        ["gold0", "gold1"],
        score_fn,
        FakeTokenizer(),
        cfg=object(),
        igpo_reproduce=True,
        placement="anchor",
    )

    expected = torch.zeros_like(responses, dtype=torch.float32)
    expected[0, 2] = 2.0
    expected[0, 4] = 4.0
    torch.testing.assert_close(rewards, expected)

    assert diagnostics["placements"][0]["placements"][0]["reason"] == "query_end"
    assert diagnostics["placements"][0]["placements"][1]["reason"] == "last_non_pad"
    assert diagnostics["placements"][1]["placements"][0]["reason"] == "skipped"
    assert diagnostics["skipped_credits"] == 1


def _harvest_env(*, answer_prompt, baseline_prompt, action_type="SELECT"):
    step = {
        "selected_sids_before": [],
        "baseline_prompt": baseline_prompt,
        "taken_action_id": "a0",
        "frontier": {
            "a0": _candidate(answer_prompt, used_entity_ids=["E0"], surfaced_sids=["S0"]),
            "a1": _candidate("unrelated frontier prompt", used_entity_ids=["E1"], surfaced_sids=["S1"]),
        },
        "taken": {
            "action_id": "a0",
            "action_type": action_type,
            "surfaced_sids": ["S0"],
            "used_entity_ids": ["E0"],
            "answer_prompt": answer_prompt,
        },
    }
    return SimpleNamespace(_snc_steps=[step], _actions=["<query>one</query>"])


def _harvest_credits(env, gold, harvest_bonus):
    from harness_g.snc import SncConfig

    responses = torch.tensor([[11, 7, 8, 0]], dtype=torch.long)
    score_fn = MockScoreFn()
    _, diagnostics = snc_trainer.compute_snc_token_rewards_with_diagnostics(
        [env],
        responses,
        [gold],
        score_fn,
        FakeTokenizer(),
        cfg=SncConfig(harvest_bonus=harvest_bonus),
        igpo_reproduce=True,
        placement="anchor",
    )
    return diagnostics["step_credits"][0]


def test_harvest_bonus_credits_select_that_harvests_answer():
    env = _harvest_env(
        answer_prompt="question? evidence: Leslie was an American director.",
        baseline_prompt="question? evidence: none",
    )
    base = _harvest_credits(env, "American", harvest_bonus=0.0)
    boosted = _harvest_credits(env, "American", harvest_bonus=0.2)
    assert boosted[0] == pytest.approx(base[0] + 0.2)


def test_harvest_bonus_skips_non_select_actions():
    env = _harvest_env(
        answer_prompt="question? evidence: Leslie was an American director.",
        baseline_prompt="question? evidence: none",
        action_type="EXPAND_ENTITY",
    )
    base = _harvest_credits(env, "American", harvest_bonus=0.0)
    boosted = _harvest_credits(env, "American", harvest_bonus=0.2)
    assert boosted[0] == pytest.approx(base[0])


def test_harvest_bonus_skips_when_answer_already_selected():
    env = _harvest_env(
        answer_prompt="question? evidence: an American director. more text",
        baseline_prompt="question? evidence: already has American here",
    )
    base = _harvest_credits(env, "American", harvest_bonus=0.0)
    boosted = _harvest_credits(env, "American", harvest_bonus=0.2)
    assert boosted[0] == pytest.approx(base[0])


def test_select_enabling_step_builds_edge_and_r_en_via_surfaced_entity_ids():
    """Regression for Fix A: a SELECT whose used_entity_ids is empty but whose
    surfaced sentence mentions entity E must still carry E as a dependency-
    bearing ID. A later LOOKUP that harvests via the same entity E then forms a
    dependency edge, and r_en propagates the harvest IG back to the SELECT.

    Before Fix A, surfaced_entity_ids was dropped at the API payload boundary,
    so the SELECT carried no entities, no edge formed, and r_en stayed 0 even
    though the episode clearly had an enabling relationship.
    """
    from harness_g.snc import SncConfig, compute_snc_credit

    # Step 0: SELECT S1 (mentions entity E_bridge). SELECT queries no entity,
    # so used_entity_ids is empty — E_bridge only arrives via surfaced_entity_ids.
    select_step = {
        "selected_sids_before": [],
        "baseline_prompt": "sel_base",
        "taken_action_id": "a0",
        "frontier": {
            "a0": _candidate("sel_take", surfaced_sids=["S1"], surfaced_entity_ids=["E_bridge"]),
        },
        "taken": {
            "action_id": "a0",
            "action_type": "SELECT",
            "surfaced_sids": ["S1"],
            "surfaced_entity_ids": ["E_bridge"],
            "produced_sids": ["S1"],
            "produced_entity_ids": ["E_bridge"],
            "consumed_sids": ["S1"],
            "used_entity_ids": [],
            "is_information_action": False,
            "answer_prompt": "sel_take",
        },
    }
    # Step 1: LOOKUP E_bridge (harvests the answer sentence). LOOKUP queries
    # E_bridge, so used_entity_ids carries it too.
    lookup_step = {
        "selected_sids_before": ["S1"],
        "baseline_prompt": "look_base",
        "taken_action_id": "b0",
        "frontier": {
            "b0": _candidate("look_take", used_entity_ids=["E_bridge"], surfaced_sids=["S2"]),
        },
        "taken": {
            "action_id": "b0",
            "action_type": "LOOKUP",
            "surfaced_sids": ["S2"],
            "surfaced_entity_ids": ["E_bridge"],
            "produced_sids": ["S2"],
            "produced_entity_ids": ["E_bridge"],
            "consumed_sids": ["S1"],
            "used_entity_ids": ["E_bridge"],
            "is_information_action": True,
            "answer_prompt": "look_take",
        },
    }
    env = SimpleNamespace(
        _snc_steps=[select_step, lookup_step],
        _actions=["<query>one</query>", "<query>two</query>"],
    )
    responses = torch.tensor(
        [[11, 7, 8, 12, 7, 8, 13, 0]],
        dtype=torch.long,
    )
    # sel_base=0.2, sel_take=0.3 (SELECT itself barely moves gold prob — the
    # enabling case); look_base=0.3, look_take=0.9 (LOOKUP harvests the answer).
    score_fn = MockScoreFn(
        {"sel_base": 0.2, "sel_take": 0.3, "look_base": 0.3, "look_take": 0.9}
    )

    rewards, diagnostics = snc_trainer.compute_snc_token_rewards_with_diagnostics(
        [env],
        responses,
        ["gold"],
        score_fn,
        FakeTokenizer(),
        cfg=SncConfig(alpha=1.0, beta=1.0),
        placement="anchor",
    )

    steps = diagnostics["snc_steps"][0]
    # Fix A: the SELECT step carries E_bridge via surfaced_entity_ids even
    # though its used_entity_ids is empty.
    assert steps[0].surfaced_entity_ids == frozenset({"E_bridge"})
    assert steps[1].surfaced_entity_ids == frozenset({"E_bridge"})

    # Recompute credit directly to inspect r_en / edges without relying on the
    # real compute_snc_credit being the one invoked by the trainer (it is, but
    # this makes the assertion self-contained).
    credit = compute_snc_credit(steps, SncConfig(alpha=1.0, beta=1.0))
    assert credit.dependency_edges == [(0, 1)]
    # r_en on the SELECT (index 0) gets the LOOKUP's IG shared back.
    assert credit.r_en[0] == pytest.approx(steps[1].taken_ig)
    assert credit.r_en[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Span placement (HARNESS_G_SNC_PLACEMENT=span)
# FakeTokenizer: <query>=[6,7], </query>=[7,8], pad=0.
# ---------------------------------------------------------------------------

def test_span_placement_distributes_credit_over_query_block():
    # <query>INIT</query> <query>A0</query>  -> tokens below.
    # idx:  0 1 2  3 4  5   6 7 8  9 10 11  12 13
    resp = torch.tensor([[6, 7, 11, 7, 8, 12, 6, 7, 13, 7, 8, 14, 0, 0]], dtype=torch.long)
    rewards, _diags, skipped, span_stats = snc_trainer._place_token_rewards(
        resp, [[1.0, 2.0]], [7, 8], [6, 7], 0, placement="span")

    # step 0 span [0,4] -> 5 tokens -> 0.2 each; step 1 span [6,10] -> 0.4 each
    assert torch.allclose(rewards[0, 0:5], torch.full((5,), 0.2))
    assert torch.allclose(rewards[0, 6:11], torch.full((5,), 0.4))
    assert rewards[0, 5].item() == 0.0 and rewards[0, 11].item() == 0.0
    assert skipped == 0
    assert span_stats["span_match_rate"] == 1.0
    assert span_stats["span_len_mean"] == 5.0
    assert span_stats["skipped_credit_count"] == 0


def test_span_placement_excludes_padding_tokens():
    # span [0,4] contains a pad at idx 2 -> excluded from distribution.
    resp = torch.tensor([[6, 7, 0, 7, 8, 99, 0, 0]], dtype=torch.long)
    rewards, _diags, _skipped, span_stats = snc_trainer._place_token_rewards(
        resp, [[1.0]], [7, 8], [6, 7], 0, placement="span")

    # non-pad span tokens = idx 0,1,3,4 -> 4 tokens -> 0.25 each; idx 2 stays 0
    assert rewards[0, 2].item() == 0.0
    assert torch.allclose(rewards[0, 0], torch.tensor(0.25))
    assert torch.allclose(rewards[0, 4], torch.tensor(0.25))
    assert span_stats["span_len_mean"] == 4.0


def test_span_placement_falls_back_to_anchor_without_query_start():
    # No <query> ([6,7]) tokens, only </query> ([7,8]) -> span fallback to anchor.
    resp = torch.tensor([[11, 7, 8, 12, 7, 8, 13, 0, 0]], dtype=torch.long)
    rewards, diags, _skipped, span_stats = snc_trainer._place_token_rewards(
        resp, [[1.0, 1.0]], [7, 8], [6, 7], 0, placement="span")

    # fallback lands full credit on the </query> end tokens (idx 2 and 5)
    assert rewards[0, 2].item() == 1.0 and rewards[0, 5].item() == 1.0
    assert diags[0]["placements"][0]["reason"] == "span_fallback_anchor"
    assert span_stats["span_placed"] == 0
    assert span_stats["anchor_placed"] == 2
    assert span_stats["span_match_rate"] == 0.0


def test_span_placement_skips_when_no_anchor_available():
    # pad-only response -> no anchor -> credit skipped with diagnostics.
    resp = torch.tensor([[0, 0, 0]], dtype=torch.long)
    rewards, diags, skipped, span_stats = snc_trainer._place_token_rewards(
        resp, [[1.0]], [7, 8], [6, 7], 0, placement="span")

    assert rewards.abs().sum().item() == 0.0
    assert skipped == 1
    assert span_stats["skipped_credit_count"] == 1
    assert diags[0]["placements"][0]["reason"] == "skipped"


def test_anchor_placement_mode_preserves_legacy_behavior():
    # placement="anchor" places full credit at </query> end token (legacy).
    resp = torch.tensor([[6, 7, 11, 7, 8, 12, 6, 7, 13, 7, 8, 14, 0, 0]], dtype=torch.long)
    rewards, diags, _skipped, span_stats = snc_trainer._place_token_rewards(
        resp, [[1.0, 2.0]], [7, 8], [6, 7], 0, placement="anchor")

    # step 0 </query> end at idx 4; step 1 </query> end at idx 10
    assert rewards[0, 4].item() == 1.0 and rewards[0, 10].item() == 2.0
    assert rewards[0, 0:4].sum().item() == 0.0
    assert diags[0]["placements"][0]["reason"] == "query_end"
    assert span_stats["span_placed"] == 0
    assert span_stats["anchor_placed"] == 2


def test_production_snc_scorer_uses_reference_chat_template_and_alias_max(
    monkeypatch,
):
    from math import exp

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class ScoringTokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def __init__(self):
            self.chat_calls = []
            self.encode_calls = []

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt
        ):
            self.chat_calls.append((messages, tokenize, add_generation_prompt))
            return f"CHAT:{messages[0]['content']}|ASSISTANT:"

        def encode(self, text, *, add_special_tokens=False):
            self.encode_calls.append((text, add_special_tokens))
            alias_tokens = {"first": [101], "second": [102], "only": [103]}
            return alias_tokens.get(text, [11, 12])

    class ReferenceWorker:
        world_size = 1

        def __init__(self):
            self.calls = []

        def compute_ref_log_prob(self, data):
            self.calls.append(data)
            responses = data.batch["responses"]
            values = {101: -2.0, 102: -0.2, 103: -1.0}
            return torch.tensor(
                [[values[int(row[0])]] for row in responses], dtype=torch.float32
            )

    class ForbiddenActorWorker:
        def compute_log_prob(self, _):
            raise AssertionError("reference scoring must not call the live actor")

    tokenizer = ScoringTokenizer()
    reference = ReferenceWorker()
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.tokenizer = tokenizer
    trainer.use_reference_policy = True
    trainer.ref_policy_wg = reference
    trainer.actor_rollout_wg = ForbiddenActorWorker()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=2)
        )
    )
    monkeypatch.setenv("HARNESS_G_SNC_SCORER", "reference")
    monkeypatch.setenv("HARNESS_G_SNC_G_NORM", "mean_prob")

    score_fn = trainer._make_snc_score_fn()
    scores = score_fn(
        ["question 0", "question 1"], [["first", "second"], ["only"]]
    )

    assert scores == pytest.approx([exp(-0.2), exp(-1.0)])
    assert len(reference.calls) == 1
    assert [call[0][0]["content"] for call in tokenizer.chat_calls] == [
        "question 0",
        "question 1",
    ]
    assert all(call[1:] == (False, True) for call in tokenizer.chat_calls)
    encoded_texts = [text for text, _ in tokenizer.encode_calls]
    assert {"first", "second", "only"} <= set(encoded_texts)
    assert all("</answer>" not in text for text in encoded_texts)
    assert sum(text.endswith("<answer>") for text in encoded_texts) == 3


# ---------------------------------------------------------------------------
# Decode+offset-mapping span detection (real-tokenizer path).
# The Qwen tokenizer merges `<`/`>` with adjacent chars, so naive token-
# subsequence matching fails; the decode path must recover the block spans.
# ---------------------------------------------------------------------------

_QWEN_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
_has_qwen = os.path.isdir(_QWEN_PATH)


@pytest.mark.skipif(not _has_qwen, reason="Qwen tokenizer not available on this host")
def test_decode_based_span_placement_with_real_qwen_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_QWEN_PATH, trust_remote_code=True)
    # Two real query blocks; the `>`/`</` boundaries merge in Qwen's BPE, so
    # subsequence matching for <query>/</query> would FAIL on this input.
    text = ("<|im_start|>assistant\n<query>{\"query\": \"INIT\"}</query><|im_end|>\n"
            "<|im_start|>user\n<knowledge>obs</knowledge><|im_end|>\n"
            "<|im_start|>assistant\n<query>{\"query\": \"A0\"}</query><|im_end|>")
    ids = tok.encode(text, add_special_tokens=False)
    resp = torch.tensor([ids], dtype=torch.long)

    rewards, diags, skipped, span_stats = snc_trainer._place_token_rewards(
        resp, [[1.0, 2.0]], tok.encode("</query>", add_special_tokens=False),
        tok.encode("<query>", add_special_tokens=False), tok.pad_token_id,
        placement="span", tokenizer=tok)

    # Both credits placed via span (NOT falling back to anchor/last_non_pad).
    assert span_stats["span_match_rate"] == 1.0
    assert span_stats["span_placed"] == 2
    assert span_stats["anchor_placed"] == 0
    assert skipped == 0
    # span 0 and span 1 are distinct token ranges; each sums to its credit.
    p0 = diags[0]["placements"][0]
    p1 = diags[0]["placements"][1]
    assert p0["reason"] == "span" and p1["reason"] == "span"
    s0, e0 = p0["span_start"], p0["span_end"]
    s1, e1 = p1["span_start"], p1["span_end"]
    assert e0 < s1  # blocks don't overlap
    assert torch.allclose(rewards[0, s0:e0 + 1].sum(), torch.tensor(1.0))
    assert torch.allclose(rewards[0, s1:e1 + 1].sum(), torch.tensor(2.0))
    # nothing outside the two spans
    outside = rewards[0].clone()
    outside[s0:e0 + 1] = 0
    outside[s1:e1 + 1] = 0
    assert outside.abs().sum().item() == 0.0
