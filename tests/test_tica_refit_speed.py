"""Refit read speed-ups (REPLICATION 2026-10-15d) give the same reads as the original paths."""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "tests")

from next_chameleons.tica.adaptive import AdaptiveConfig


def _convs(n, prefix=""):
    return [[{"role": "user", "content": f"{prefix}req {k}"},
             {"role": "assistant", "content": "ab " * (k % 4 + 1) + f"x{k}"}] for k in range(n)]


def test_speedups_off_by_default():
    a = AdaptiveConfig()
    assert a.episode_fork_batch == 0 and a.refit_np_cache is False


def test_batched_pooled_tokens_match_incremental():
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    convs = _convs(7)
    t_inc, t_bat = {}, {}
    a = pol.pooled_incremental(convs, [1, 2], {}, token_out=t_inc, token_keep=3)
    b = pol.pooled(convs, [1, 2], token_out=t_bat, token_keep=3, batch=5)
    for L in (1, 2):
        assert torch.allclose(a[L], b[L], atol=1e-5)
        for x, y in zip(t_inc[L], t_bat[L], strict=True):
            assert torch.allclose(x.float(), y.float(), atol=1e-2)


def test_shared_prefix_fork_batch_matches_default():
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pre = "a long shared history " * 6
    convs = [[{"role": "user", "content": pre}, {"role": "assistant", "content": "ok"},
              {"role": "user", "content": f"next {k}"},
              {"role": "assistant", "content": "zz " * (k % 3 + 1)}] for k in range(9)]
    t1, t2 = {}, {}
    a = pol.pooled_shared_prefix(convs, [1], token_out=t1)
    b = pol.pooled_shared_prefix(convs, [1], token_out=t2, batch=9)
    assert torch.allclose(a[1], b[1], atol=1e-5)
    assert all(torch.allclose(x.float(), y.float(), atol=1e-2)
               for x, y in zip(t1[1], t2[1], strict=True))


def test_np_cache_matches_and_hits(tmp_path):
    from test_tica_kl_anchor import _setup

    tr, pol, _ = _setup(tmp_path, refit_np_cache=True, episode_fork_batch=4)
    convs = _convs(5)
    ref_t = {}
    ref = pol.pooled_incremental(convs, [1], {}, token_out=ref_t)
    t = {}
    got = tr._np_reads(None, convs, [1], t, None)
    assert torch.allclose(ref[1], got[1], atol=1e-5)
    t2 = {}
    again = tr._np_reads(None, convs[:3], [1], t2, None)  # all hits
    assert tr._np_hits == 3 and torch.allclose(again[1], got[1][:3])
    pol.weights_version = getattr(pol, "weights_version", 0) + 1  # new weights: cache dropped
    tr._np_reads(None, convs[:2], [1], {}, None)
    assert tr._np_hits == 3


def _tok_data(layers, n_pos=5, n_neg=6, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = [int(torch.randint(3, 9, (1,), generator=g)) for _ in range(n_pos + n_neg)]
    P = {L: [torch.randn(k, d, generator=g) + 0.5 for k in lens[:n_pos]] for L in layers}
    N = {L: [torch.randn(k, d, generator=g) - 0.5 for k in lens[n_pos:]] for L in layers}
    return P, N


def test_tokens_ens_batched_matches_loop():
    from next_chameleons.tica import probe_zoo as z

    for kind in ("nc_ens8_tok", "bailey_ens_tok"):
        P, N = _tok_data(z.TOKENS_ENS_LAYERS[kind])
        a = z._fit_tokens_ens(kind, P, N, torch.Generator().manual_seed(7), "cpu")
        b = z._fit_tokens_ens_loop(kind, P, N, torch.Generator().manual_seed(7), "cpu")
        for L in a:
            assert torch.allclose(a[L].weight, b[L].weight, atol=1e-5), (kind, L)
            assert torch.allclose(a[L].bias, b[L].bias, atol=1e-5), (kind, L)


def test_bailey_max_fires_if_any_layer_fires():
    from next_chameleons.tica import probe_zoo as z

    Ls = z.TOKENS_ENS_LAYERS["bailey_ens_tok"]
    P, N = _tok_data(Ls)
    lins = z._fit_tokens_ens("bailey_ens_tok", P, N, torch.Generator().manual_seed(1), "cpu")
    mean = z.TokensEnsMonitor("bailey_ens_tok", "m", 12, "c", lins)
    mx = z.TokensEnsMonitor("bailey_ens_tok_max", "x", 12, "c", lins)
    xs = {L: P[L][:2] for L in Ls}
    assert (mx.logits(xs) >= mean.logits(xs) - 1e-6).all()
    band = z.kind_layers("bailey_ens_tok_max", 12)
    assert min(band) == 4 and max(band) == 28 and 20 in band and 24 in band  # 10-19: all depths


def test_merged_reads_match_and_restore_exactly():
    import pytest

    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    with torch.no_grad():  # nonzero adapters
        g = torch.Generator().manual_seed(2)
        for n, p in pol.model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    before = {n: p.detach().clone() for n, p in pol.model.named_parameters()}
    convs = _convs(4)
    want = pol.pooled(convs, [1, 2])
    with pol.merged():
        got = pol.pooled(convs, [1, 2])
        with pytest.raises(RuntimeError), pol.reference():
            pass
    for L in (1, 2):
        assert torch.allclose(want[L], got[L], atol=1e-4)
    for n, p in pol.model.named_parameters():
        assert torch.equal(p, before[n]), n  # restored bit for bit
    with pol.reference():  # usable again after exit
        pol.pooled(convs[:1], [1])
    assert AdaptiveConfig().merged_reads is False


def test_truncated_reads_match_full_and_skip_deeper_layers():
    """truncate_reads (10-16c): reads at layers <= L are identical with the forward stopped after
    L, and deeper layers never run; the full depth is restored afterwards."""
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    assert AdaptiveConfig().truncate_reads is False and pol.truncate_reads is False
    long = "shared history " * 40
    forks = [[{"role": "user", "content": long},
              {"role": "assistant", "content": "ab cd"},
              {"role": "user", "content": f"next {k}"},
              {"role": "assistant", "content": "ef " * (k + 1)}] for k in range(5)]
    convs = _convs(6)
    calls = []
    h = pol.layers_module[2].register_forward_hook(lambda *a: calls.append(1))
    full_p = pol.pooled(convs, [0, 1])
    full_s = pol.pooled_shared_prefix(forks, [1])
    n_full = len(calls)
    pol.truncate_reads = True
    calls.clear()
    tr_p = pol.pooled(convs, [0, 1])
    tr_s = pol.pooled_shared_prefix(forks, [1])
    assert len(calls) == 0 and n_full > 0
    h.remove()
    pol.truncate_reads = False
    assert pol._depth_owner.config.num_hidden_layers == 3
    for L in (0, 1):
        assert torch.equal(full_p[L], tr_p[L])
    assert torch.allclose(full_s[1], tr_s[1], atol=1e-6)


def test_batched_mlp_ensemble_fits_match_sequential_gate():
    """refit_batched_fits (10-16c): the MLP-ensemble gate (5 CV fits + final, all layers in one
    stacked AdamW run) gives the same probe (logits to ~1e-4), AUROC and verdict as the per-fit
    path."""
    import dataclasses

    from next_chameleons.tica import probe_zoo as z

    g = torch.Generator().manual_seed(3)
    d = 24
    u = torch.randn(d, generator=g)
    mk = lambda n, s: torch.randn(n, d, generator=g) + s * 0.7 * u + 2.0  # noqa: E731
    seq = AdaptiveConfig(refit_cv_folds=5, refit_min_auroc=0.0, refit_min_tpr_np=0.0)
    bat = dataclasses.replace(seq, refit_batched_fits=True)
    for kind in ("ens4_mlp", "ens8_mlp"):
        Ls = z.kind_layers(kind, 12)
        P = {L: mk(40, 1) for L in Ls}
        N = {L: mk(60, -1) for L in Ls}
        NP = {L: mk(8, 1) for L in Ls}
        ma, qa = z.gate_fit(kind, P, N, NP, seq, name="e", layer=12, concept="c")
        mb, qb = z.gate_fit(kind, P, N, NP, bat, name="e", layer=12, concept="c")
        assert abs(qa["auroc"] - qb["auroc"]) < 1e-4, (kind, qa, qb)
        assert qa["tpr_np"] == qb["tpr_np"] and qa["kept"] == qb["kept"], kind
        xs = {L: torch.cat([P[L], N[L]]) for L in Ls}
        assert torch.allclose(ma.logits(xs), mb.logits(xs), atol=1e-4), kind
        assert abs(ma.thr - mb.thr) < 1e-4, kind


def test_batched_token_pool_fits_match_sequential_gate():
    """refit_batched_fits (10-16n): attnpool / maxpool gates (5 CV fits + final in one stacked run,
    standardisation folded into the parameters) match the per-fit path."""
    import dataclasses

    from next_chameleons.tica import probe_zoo as z

    g = torch.Generator().manual_seed(5)
    d = 12
    u = torch.randn(d, generator=g)

    def seqs(n, s):
        return [torch.randn(int(torch.randint(6, 14, (1,), generator=g)), d, generator=g)
                + s * 0.8 * u + 1.5 for _ in range(n)]

    seq = AdaptiveConfig(refit_cv_folds=5, refit_min_auroc=0.0, refit_min_tpr_np=0.0)
    bat = dataclasses.replace(seq, refit_batched_fits=True)
    P, N, NP = seqs(30, 1), seqs(40, -1), seqs(6, 1)
    for kind in ("attnpool", "maxpool"):
        ma, qa = z.gate_fit(kind, P, N, NP, seq, name="t", layer=12, concept="c")
        mb, qb = z.gate_fit(kind, P, N, NP, bat, name="t", layer=12, concept="c")
        assert abs(qa["auroc"] - qb["auroc"]) < 2e-3, (kind, qa, qb)
        assert qa["kept"] == qb["kept"], kind
        la, lb = ma.logits(P + N), mb.logits(P + N)
        assert torch.allclose(la, lb, atol=2e-3, rtol=2e-3), (kind, (la - lb).abs().max())
