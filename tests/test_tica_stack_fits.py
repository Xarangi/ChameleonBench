"""adaptive.refit_stack_episodes (10-29): stacked rlobf fits across refits."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from next_chameleons.tica import probe_zoo as Z


def _data(seed, n_pos=24, n_neg=30, d=16):
    g = torch.Generator().manual_seed(seed)

    def seq(m):
        return torch.randn(int(torch.randint(3, 9, (1,), generator=g)), d, generator=g) + m

    pos, neg = [seq(0.6) for _ in range(n_pos)], [seq(-0.6) for _ in range(n_neg)]
    return pos, neg, [seq(0.6) for _ in range(8)]


def test_gate_fit_many_matches_gate_fit():
    cfg = SimpleNamespace(refit_cv_folds=3, refit_batched_rlobf=True, refit_fpr=0.05,
                          refit_min_auroc=0.5, refit_min_tpr_np=0.0, refit_min_tpr_ic=0.0,
                          refit_C=1.0, refit_batched_fits=False, refit_batched_logistic=False)
    items = []
    for s in (1, 2, 3):
        P, N, NP = _data(s, n_pos=20 + 4 * s)
        items.append({"P": P, "N": N, "NP": NP, "name": f"ep{s}", "layer": 3, "concept": "c"})
    torch.manual_seed(0)
    one = [Z.gate_fit("rlobf_attn", it["P"], it["N"], it["NP"], cfg, name=it["name"],
                      layer=it["layer"], concept=it["concept"]) for it in items]
    torch.manual_seed(0)
    many = Z.gate_fit_many("rlobf_attn", items, cfg)
    for (m1, i1), (m2, i2) in zip(one, many, strict=True):
        assert i1["kept"] == i2["kept"] and abs(i1["auroc"] - i2["auroc"]) < 1e-4
        assert abs(i1["tpr_np"] - i2["tpr_np"]) < 1e-6
        if m1 is not None:
            for p1, p2 in zip(m1.net.state_dict().values(), m2.net.state_dict().values(),
                              strict=True):
                assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-4)
