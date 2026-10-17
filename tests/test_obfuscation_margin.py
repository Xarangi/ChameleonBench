"""Operating-point-aligned hinge for the attacker obfuscation loss.

Motivates the margin loss added 2026-07-25: MSE(score, 0) gradient vanishes as an
evasion score -> 0, so the attacker stalls ~0.01 while the 1%-FPR threshold is
~1e-3. The hinge keeps a constant push until the score clears `floor`.
"""

import pytest

torch = pytest.importorskip("torch")

from next_chameleons.hf.training import obfuscation_margin_term  # noqa: E402


def test_evasion_score_above_floor_is_penalized_linearly() -> None:
    floor = 1e-3
    scores = torch.tensor([0.5, 0.01, 5e-4])  # target=0 (evade)
    targets = torch.zeros(3)
    term = obfuscation_margin_term(scores, targets, floor)
    # above floor -> penalty == score - floor; below floor -> 0
    assert term[0].item() == pytest.approx(0.5 - floor)
    assert term[1].item() == pytest.approx(0.01 - floor)
    assert term[2].item() == 0.0


def test_hinge_gradient_does_not_vanish_near_zero_unlike_mse() -> None:
    floor = 1e-3
    # An evasion score sitting at 0.01 (where MSE has stalled).
    s = torch.tensor([0.01], requires_grad=True)
    t = torch.zeros(1)
    obfuscation_margin_term(s, t, floor).sum().backward()
    hinge_grad = s.grad.abs().item()
    s2 = torch.tensor([0.01], requires_grad=True)
    ((s2 - t) ** 2).sum().backward()
    mse_grad = s2.grad.abs().item()
    # Hinge gradient is ~1 (constant push); MSE gradient is ~0.02 (near-vanished).
    assert hinge_grad == pytest.approx(1.0)
    assert mse_grad < 0.05
    assert hinge_grad > 20 * mse_grad


def test_probe_must_fire_target_is_pushed_up() -> None:
    floor = 1e-3
    scores = torch.tensor([0.2, 0.999])  # target=1 (must fire)
    targets = torch.ones(2)
    term = obfuscation_margin_term(scores, targets, floor)
    assert term[0].item() == pytest.approx((1.0 - floor) - 0.2)
    assert term[1].item() == 0.0  # already above (1 - floor)
