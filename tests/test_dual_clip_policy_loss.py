"""Dual-clip PPO regression tests.

The intro menu 3B run collapsed around step 85: negative-advantage tokens with
importance ratios up to e^20 produced pg_loss spikes (1e2-1e3) and grad_norm up
to 1e7. Grad clipping only rescales the noise direction, so the policy died.
compute_policy_loss(clip_ratio_c=...) must bound both the loss and the gradient
contributed by such tokens, while leaving healthy tokens untouched.
"""
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss


def _make_batch(log_ratio, advantage):
    """One-token batch with the given log(pi/pi_old) and advantage."""
    old_log_prob = torch.zeros(1, 1)
    log_prob = torch.full((1, 1), float(log_ratio), requires_grad=True)
    advantages = torch.full((1, 1), float(advantage))
    eos_mask = torch.ones(1, 1)
    return old_log_prob, log_prob, advantages, eos_mask


def test_negative_adv_huge_ratio_unbounded_without_dual_clip():
    # ratio = e^15, A = -0.5 -> vanilla PPO loss ~ 0.5 * e^15
    old, logp, adv, mask = _make_batch(15.0, -0.5)
    pg_loss, _, _ = compute_policy_loss(old, logp, adv, mask, cliprange=0.2)
    assert pg_loss.item() > 1e5  # documents the failure mode dual-clip removes


def test_dual_clip_bounds_loss_and_gradient():
    old, logp, adv, mask = _make_batch(15.0, -0.5)
    pg_loss, _, _ = compute_policy_loss(old, logp, adv, mask, cliprange=0.2,
                                        clip_ratio_c=3.0)
    # loss capped at c * |A| = 1.5
    assert abs(pg_loss.item() - 1.5) < 1e-6
    # ceiling is constant wrt theta -> zero gradient through the bad ratio
    pg_loss.backward()
    assert torch.all(logp.grad == 0)


def test_dual_clip_inactive_on_moderate_ratios():
    for log_ratio, advantage in [(0.05, 0.7), (-0.1, -0.4), (0.3, -0.2)]:
        old, logp, adv, mask = _make_batch(log_ratio, advantage)
        base, base_frac, base_kl = compute_policy_loss(old, logp, adv, mask,
                                                       cliprange=0.2)
        dual, dual_frac, dual_kl = compute_policy_loss(old, logp, adv, mask,
                                                       cliprange=0.2,
                                                       clip_ratio_c=3.0)
        assert torch.allclose(base, dual)
        assert torch.allclose(base_frac, dual_frac)
        assert torch.allclose(base_kl, dual_kl)


def test_dual_clip_positive_adv_untouched():
    # A > 0 with a huge ratio is already bounded by the vanilla clip term;
    # dual-clip must not alter it.
    old, logp, adv, mask = _make_batch(15.0, 0.5)
    base, _, _ = compute_policy_loss(old, logp, adv, mask, cliprange=0.2)
    dual, _, _ = compute_policy_loss(old, logp, adv, mask, cliprange=0.2,
                                     clip_ratio_c=3.0)
    assert torch.allclose(base, dual)
    assert abs(dual.item() - (-0.5 * 1.2)) < 1e-6


def test_dual_clip_none_matches_legacy():
    torch.manual_seed(0)
    old = torch.randn(4, 8)
    logp = old + 0.1 * torch.randn(4, 8)
    adv = torch.randn(4, 8)
    mask = (torch.rand(4, 8) > 0.3).float()
    base = compute_policy_loss(old, logp, adv, mask, cliprange=0.2)
    dual = compute_policy_loss(old, logp, adv, mask, cliprange=0.2,
                               clip_ratio_c=None)
    for a, b in zip(base, dual):
        assert torch.allclose(a, b)
