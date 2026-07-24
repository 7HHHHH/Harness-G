"""Unit tests for the grpo_snc advantage (GRPO outcome + lambda * SNC span-local).

These exercise the pure-torch math in ``verl.trainer.ppo.core_algos`` without
needing a trainer, actor, or env.
"""
from collections import defaultdict

import numpy as np
import pytest
import torch

from verl.trainer.ppo import core_algos


def _uid_index(uids):
    # Real trainer passes a numpy array (non_tensor_batch['uid']); numpy scalars
    # hash by value, unlike torch 0-dim tensors which hash by identity.
    return np.array(uids)


def test_compute_grpo_snc_advantage_lambda_zero_equals_grpo_outcome():
    # Outcome reward is a scalar at the last token; two rollouts per uid.
    outcome = torch.zeros(2, 6)
    outcome[0, 5] = 1.0
    outcome[1, 5] = 0.0
    snc = torch.zeros(2, 6)
    snc[0, 1] = 0.5
    snc[1, 1] = 0.2
    mask = torch.ones(2, 6, dtype=torch.bool)
    index = _uid_index([0, 0])

    adv_snc, ret_snc = core_algos.compute_grpo_snc_advantage(
        outcome, snc, mask, index, loss_mask=mask, snc_lambda=0.0)
    adv_grpo, ret_grpo = core_algos.compute_grpo_outcome_advantage(outcome, mask, index)

    assert torch.allclose(adv_snc, adv_grpo)
    assert torch.allclose(ret_snc, ret_grpo)


def test_snc_span_advantage_keeps_non_credit_tokens_zero():
    # SNC credit only on token 1 of each rollout; token 3 must stay 0.
    snc = torch.zeros(4, 6)
    snc[0, 1] = 1.0
    snc[1, 1] = 0.2
    snc[2, 1] = 0.9
    snc[3, 1] = 0.1
    mask = torch.ones(4, 6, dtype=torch.bool)
    index = _uid_index([0, 0, 1, 1])

    adv = core_algos.compute_snc_span_advantage(snc, (snc != 0) & mask, index)

    # non-credit tokens are exactly 0 (never negative from a subtracted mean)
    assert adv[:, 3].abs().sum().item() == 0.0
    assert adv[:, 0].abs().sum().item() == 0.0
    # credit tokens are non-zero where the group has variance
    assert adv[0, 1].item() != 0.0


def test_snc_span_advantage_all_zero_no_nan():
    snc = torch.zeros(3, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    index = _uid_index([0, 0, 1])

    adv = core_algos.compute_snc_span_advantage(snc, (snc != 0) & mask, index)

    assert torch.isfinite(adv).all()
    assert adv.abs().sum().item() == 0.0


def test_snc_span_advantage_uses_one_global_scale_across_uids():
    snc = torch.zeros(2, 5)
    snc[0, 1] = 0.7
    snc[1, 2] = 0.4
    mask = torch.ones(2, 5, dtype=torch.bool)
    index = _uid_index([0, 1])  # each its own group

    adv = core_algos.compute_snc_span_advantage(snc, (snc != 0) & mask, index)

    assert torch.isfinite(adv).all()
    # Population std([0.7, 0.4]) = 0.15. UID boundaries do not erase the
    # absolute difference between the two credits.
    assert adv[0, 1].item() == pytest.approx(0.7 / 0.15)
    assert adv[1, 2].item() == pytest.approx(0.4 / 0.15)
    # non-credit tokens stay 0
    assert adv[0, 0].item() == 0.0 and adv[1, 4].item() == 0.0


def test_snc_span_advantage_identical_group_uses_floor_and_clamp():
    snc = torch.zeros(3, 5)
    snc[:, 1] = 0.5
    mask = torch.ones(3, 5, dtype=torch.bool)
    index = _uid_index([7, 7, 7])

    adv = core_algos.compute_snc_span_advantage(snc, (snc != 0) & mask, index)

    assert torch.isfinite(adv).all()
    assert adv[:, 1].abs().max().item() <= 5.0  # within clamp
    assert adv[:, 1].abs().min().item() > 0.0   # signal survives (not zeroed)


def test_snc_span_advantage_noise_below_scale_floor_stays_small():
    snc = torch.zeros(2, 4)
    snc[0, 1] = 1e-6
    snc[1, 1] = -1e-6
    mask = torch.ones(2, 4, dtype=torch.bool)
    index = _uid_index([0, 0])

    adv = core_algos.compute_snc_span_advantage(
        snc,
        (snc != 0) & mask,
        index,
        scale_floor=5e-4,
    )

    assert adv[0, 1].item() == pytest.approx(0.002)
    assert adv[1, 1].item() == pytest.approx(-0.002)


def test_snc_span_advantage_different_spans_both_get_positive_signal():
    # Same uid, credit on different token positions with different magnitudes.
    # Scale-only: both get POSITIVE advantage (no mean centering -> no sign
    # flip), proportional to their credit. The larger credit gets larger adv.
    snc = torch.zeros(2, 6)
    snc[0, 1] = 0.8
    snc[1, 4] = 0.2
    mask = torch.ones(2, 6, dtype=torch.bool)
    index = _uid_index([0, 0])

    adv = core_algos.compute_snc_span_advantage(snc, (snc != 0) & mask, index)

    # advantage localized to each rollout's support token
    assert adv[0, 1].item() > 0.0
    assert adv[1, 4].item() > 0.0
    # both positive (NOT opposite signs — no mean centering)
    assert adv[0, 1].item() * adv[1, 4].item() > 0.0
    # larger credit -> larger advantage
    assert adv[0, 1].item() > adv[1, 4].item()
    # non-credit tokens stay exactly 0
    assert adv[0, 4].item() == 0.0 and adv[1, 1].item() == 0.0


def test_snc_span_advantage_same_total_different_position_both_get_signal():
    # THE canonical case the scale-only design exists to handle:
    # rollout A credits action 1, rollout B credits action 2, SAME total.
    # Old sum-then-center impl gave 0 advantage here (bug). Scale-only must
    # give both a positive local signal on their respective spans.
    snc = torch.zeros(2, 4)
    snc[0, 0] = 1.0   # A: action 1 good
    snc[1, 1] = 1.0   # B: action 2 good
    mask = torch.ones(2, 4, dtype=torch.bool)
    index = _uid_index([0, 0])

    adv = core_algos.compute_snc_span_advantage(snc, (snc != 0) & mask, index)

    assert torch.isfinite(adv).all()
    assert adv[0, 0].item() > 0.0, "rollout A's action-1 span must get credit"
    assert adv[1, 1].item() > 0.0, "rollout B's action-2 span must get credit"
    # both spans rewarded (the model can learn WHICH action was good)
    assert adv.abs().sum().item() > 0.0


def test_snc_span_advantage_respects_support_mask():
    # Mask out token 1 -> even if credit is there, support_mask drops it.
    snc = torch.zeros(2, 6)
    snc[0, 1] = 1.0
    snc[1, 1] = 0.2
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 1] = 0  # token 1 excluded from support
    index = _uid_index([0, 0])

    adv = core_algos.compute_snc_span_advantage(snc, (snc != 0) & mask, index)

    assert adv.abs().sum().item() == 0.0


def test_compute_grpo_snc_advantage_adds_lambda_times_snc():
    # lambda=1.0: advantages == outcome_adv + snc_adv (not outcome only).
    outcome = torch.zeros(2, 6)
    outcome[0, 5] = 1.0
    outcome[1, 5] = 0.0
    snc = torch.zeros(2, 6)
    snc[0, 1] = 1.0
    snc[1, 1] = 0.2
    mask = torch.ones(2, 6, dtype=torch.bool)
    index = _uid_index([0, 0])

    adv0, _ = core_algos.compute_grpo_snc_advantage(
        outcome, snc, mask, index, loss_mask=mask, snc_lambda=0.0)
    adv1, _ = core_algos.compute_grpo_snc_advantage(
        outcome, snc, mask, index, loss_mask=mask, snc_lambda=1.0)

    # The SNC channel only touches support tokens; outcome is broadcast.
    diff = (adv1 - adv0)
    nz = diff[diff != 0]
    assert nz.numel() > 0  # SNC added something
    assert torch.isfinite(adv1).all()


def test_snc_span_advantage_clamps_extreme_values():
    # Pathological credit magnitudes -> clamped to [-5, 5].
    snc = torch.zeros(2, 4)
    snc[0, 1] = 1e6
    snc[1, 1] = -1e6
    mask = torch.ones(2, 4, dtype=torch.bool)
    index = _uid_index([0, 0])

    adv = core_algos.compute_snc_span_advantage(snc, (snc != 0) & mask, index, clamp=5.0)

    assert adv.abs().max().item() <= 5.0
    assert torch.isfinite(adv).all()


def test_snc_span_advantage_mixed_sign_credits_preserved_not_cancelled():
    # Scale-only normalization does NOT sum per-sequence, so mixed-sign step
    # credits no longer cancel (the old sum-then-center impl zeroed these out).
    # Each span keeps its own sign: +0.5 -> positive adv, -0.5 -> negative adv.
    snc = torch.zeros(2, 6)
    snc[0, 1] = 0.5    # useful step
    snc[0, 4] = -0.5   # harmful step -> seq sum = 0 but local signals survive
    snc[1, 1] = 0.3
    mask = torch.ones(2, 6, dtype=torch.bool)
    index = _uid_index([0, 0])

    adv = core_algos.compute_snc_span_advantage(snc, (snc != 0) & mask, index)

    assert torch.isfinite(adv).all()
    # rollout 0: positive span gets +adv, negative span gets -adv (sign kept)
    assert adv[0, 1].item() > 0.0
    assert adv[0, 4].item() < 0.0
    # rollout 1 still gets a positive local signal
    assert adv[1, 1].item() > 0.0
    # the two spans of rollout 0 did NOT cancel to zero
    assert adv[0].abs().sum().item() > 0.0
