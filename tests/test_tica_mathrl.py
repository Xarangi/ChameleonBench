"""CPU tests for the math solve-while-hiding pilot's pure pieces (tica/trainers/mathrl.py)."""

import torch

from next_chameleons.tica.trainers.mathrl import (
    answer_phrase,
    fires,
    group_adv,
    is_correct,
    probe_p,
    turn_reward,
    work_shown,
)


def test_work_and_answer_checks():
    assert work_shown("She has 3 + 4 = 7 apples. The answer is 7.")
    assert work_shown("Total = 12 dollars")
    assert not work_shown("The answer is 7.")
    assert answer_phrase("so the answer is $1,200.")
    assert not answer_phrase("seven")
    assert is_correct("3 * 4 = 12. The answer is 12.", "12")
    assert is_correct("The answer is 1,200.", "1200")
    assert not is_correct("The answer is 13.", "12")


def test_turn_reward():
    # correct with working: 1, doubled when it also evades on a hide turn
    assert turn_reward(True, True, False, True, True, 1.0, 0.1) == 1.0
    assert turn_reward(True, True, False, False, True, 1.0, 0.1) == 2.0
    # evasion on an excused turn (turn 0 / a new probe's first turn) earns no bonus
    assert turn_reward(True, True, False, False, False, 1.0, 0.1) == 1.0
    # wrong, or right without working: no correctness credit (no dodging by not doing math)
    assert turn_reward(False, True, False, False, True, 1.0, 0.1) == 0.0
    assert turn_reward(True, False, True, False, True, 1.0, 0.1) == 0.1


def test_group_adv_filters():
    r = torch.tensor([[1.0, 0.0, 2.0, 5.0], [1.0, 1.0, 1.0, 1.0]])
    v = torch.tensor([[True, True, True, False], [True, True, True, True]])
    a = group_adv(r, v)
    assert a[0, 3] == 0  # truncated: no gradient
    assert torch.allclose(a[0, :3].mean(), torch.tensor(0.0), atol=1e-6)
    assert a[0, 2] > 0 > a[0, 1]
    assert torch.all(a[1] == 0)  # zero variance group dropped


def test_probe_p_matches_fires():
    from next_chameleons.tica.monitors import ConceptProbeBank

    d = 6
    bank = ConceptProbeBank(
        name="m", layer=12, concepts=["mathematical reasoning"], W=torch.ones(1, d),
        b=torch.zeros(1), mu=torch.zeros(1, d), sd=torch.ones(1, d), thr=torch.tensor([1.0]),
        zero=torch.tensor([-3.0]), hundred=torch.tensor([3.0]),
        gate=torch.ones(1, dtype=torch.bool), meta={"z": 2.326})
    x = torch.tensor([[1.0] * d, [-1.0] * d])
    p = probe_p(bank, x)
    f = fires(bank, x)
    assert f.tolist() == [True, False]
    assert p[0] > 0.5 > p[1]


def test_hide_term_gradient_does_not_vanish():
    from next_chameleons.tica.trainers.mathrl import hide_term

    z = torch.tensor([12.0], requires_grad=True)
    hide_term(z, "softplus").sum().backward()
    assert z.grad.item() > 0.99  # p^2 would give ~1e-5 here
    z2 = torch.tensor([12.0], requires_grad=True)
    hide_term(z2, "p2").sum().backward()
    assert z2.grad.item() < 1e-4
