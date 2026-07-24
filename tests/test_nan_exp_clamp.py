"""Regression tests for the exp-overflow NaN fix in core_algos.

Root cause reproduced from runs like intro_menu_3b_2wiki_3ep (grad_norm NaN from
step 73 while every forward loss stayed finite): torch.exp() overflowed to inf on
an extreme log-prob difference, downstream masking/clamping handed that position a
zero grad_output, and exp backward computed 0 * inf = NaN.
"""
import torch

from verl.trainer.ppo import core_algos


def _masked_mean(values, mask):
    return (values * mask).sum() / mask.sum()


def test_low_var_kl_backward_finite_on_extreme_diff():
    # ref - logprob = 97 on a token that is masked out of the loss; pre-fix this
    # produced a NaN gradient even though the forward value was finite.
    logprob = torch.tensor([-100.0, -2.0], requires_grad=True)
    ref = torch.tensor([-3.0, -2.1])
    mask = torch.tensor([0.0, 1.0])

    kld = core_algos.kl_penalty(logprob, ref, kl_penalty="low_var_kl")
    loss = _masked_mean(kld, mask)
    assert torch.isfinite(loss), "forward kl loss must be finite"
    loss.backward()
    assert torch.isfinite(logprob.grad).all(), f"NaN/inf grad: {logprob.grad}"


def test_policy_loss_backward_finite_on_extreme_ratio():
    # log_prob - old_log_prob = +100 on a masked position with nonzero advantage.
    log_prob = torch.tensor([[0.0, -1.0]], requires_grad=True)
    old_log_prob = torch.tensor([[-100.0, -1.1]])
    advantages = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[0.0, 1.0]])

    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=mask,
        cliprange=0.2,
    )
    assert torch.isfinite(pg_loss)
    pg_loss.backward()
    assert torch.isfinite(log_prob.grad).all(), f"NaN/inf grad: {log_prob.grad}"


def test_low_var_kl_unchanged_in_normal_regime():
    torch.manual_seed(0)
    logprob = torch.randn(64) * 2.0
    ref = logprob + torch.empty(64).uniform_(-5.0, 5.0)

    kld = core_algos.kl_penalty(logprob, ref, kl_penalty="low_var_kl")
    d = ref - logprob
    expected = torch.clamp(torch.exp(d) - d - 1, min=-10, max=10)
    assert torch.allclose(kld, expected, atol=1e-6)


def test_policy_loss_unchanged_in_normal_regime():
    torch.manual_seed(1)
    bs, t = 4, 16
    log_prob = torch.randn(bs, t)
    old_log_prob = log_prob + torch.empty(bs, t).uniform_(-5.0, 5.0)
    advantages = torch.randn(bs, t)
    mask = (torch.rand(bs, t) > 0.3).float()
    mask[:, -1] = 1.0  # keep every row non-empty

    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=mask,
        cliprange=0.2,
    )

    # reference: the original unclamped computation
    nak = log_prob - old_log_prob
    ratio = torch.exp(nak)
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 0.8, 1.2)
    expected_pg = _masked_mean(torch.max(pg_losses, pg_losses2), mask)
    expected_kl = _masked_mean(-nak, mask)

    assert torch.allclose(pg_loss, expected_pg, atol=1e-6)
    assert torch.allclose(ppo_kl, expected_kl, atol=1e-6)
