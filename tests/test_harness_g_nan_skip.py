"""Test the V3 NaN skip-step guard (HARNESS_G_SKIP_NONFINITE_GRAD).

The actor's _optimizer_step should skip the optimizer update (and zero grads)
when the gradient norm is non-finite, instead of stepping NaN into the weights.
Uses a plain nn.Linear (non-FSDP branch) so no GPU/FSDP is needed. Skipped
gracefully if the heavy verl actor module cannot be imported in this env.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
dp_actor_mod = pytest.importorskip("verl.workers.actor.dp_actor")

DataParallelPPOActor = dp_actor_mod.DataParallelPPOActor


def _fake_actor():
    module = torch.nn.Linear(3, 3)
    stepped = {"count": 0}
    optimizer = SimpleNamespace(
        step=lambda: stepped.__setitem__("count", stepped["count"] + 1),
        zero_grad=lambda set_to_none=True: None,
    )
    fake = SimpleNamespace(
        actor_module=module,
        actor_optimizer=optimizer,
        config=SimpleNamespace(grad_clip=1.0),
    )
    return fake, module, stepped


def test_optimizer_step_skips_on_nonfinite_grad(monkeypatch):
    monkeypatch.setenv("HARNESS_G_SKIP_NONFINITE_GRAD", "1")
    fake, module, stepped = _fake_actor()
    for p in module.parameters():
        p.grad = torch.full_like(p, float("nan"))
    grad_norm = DataParallelPPOActor._optimizer_step(fake)
    assert not bool(torch.isfinite(grad_norm).all())
    assert stepped["count"] == 0  # step skipped, weights preserved


def test_optimizer_step_runs_on_finite_grad(monkeypatch):
    monkeypatch.setenv("HARNESS_G_SKIP_NONFINITE_GRAD", "1")
    fake, module, stepped = _fake_actor()
    for p in module.parameters():
        p.grad = torch.ones_like(p)
    DataParallelPPOActor._optimizer_step(fake)
    assert stepped["count"] == 1  # finite grad -> normal step


