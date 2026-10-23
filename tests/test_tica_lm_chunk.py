"""model.lm_chunk (TamIA 10-21): the chunked LM terms (policy._lm_terms, LM head input captured, no
full [B, n, V] logits) equal the full-logits path: per-span NLL, top-k reference targets, top-k KL
and every LoRA gradient, for forward and forward_multi (tiny fp32 model, many small chunks)."""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "tests")


def _convs():
    return [
        [{"role": "user", "content": "say cats"}, {"role": "assistant", "content": "meow meow"},
         {"role": "user", "content": "again"}, {"role": "assistant", "content": "purr and meow"}],
        [{"role": "user", "content": "dogs?"}, {"role": "assistant", "content": "woof"},
         {"role": "user", "content": "more"}, {"role": "assistant", "content": "bark bark woof"}],
    ]


def _run(pol, chunk, multi, kl_ref=None):
    pol.lm_chunk = chunk
    pol.model.zero_grad()
    convs = _convs()
    if multi:
        units = [pol.multi_view(c) for c in convs]
        fwd = pol.forward_multi
    else:
        units = [pol.view(c) for c in convs]
        fwd = pol.forward
    with torch.no_grad():  # reference targets (top-k of the same model)
        ex0 = {"topk": 4}
        fwd(units, [1], grad=False, with_nll=True, extra=ex0)
    ex = {"kl_ref": kl_ref if kl_ref is not None else ex0["ref"]}
    _, nll = fwd(units, [1], grad=True, with_nll=True, extra=ex)
    (nll.sum() + ex["kl"].sum()).backward()
    grads = {n: p.grad.detach().clone() for n, p in pol.model.named_parameters()
             if p.grad is not None}
    return nll.detach(), ex["kl"].detach(), ex0["ref"], grads


def test_chunked_lm_terms_match_full_logits():
    from next_chameleons.tica.config import ModelConfig
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    with torch.no_grad():  # nonzero adapters
        g = torch.Generator().manual_seed(5)
        for n, p in pol.model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        for n, p in pol.model.named_parameters():
            if "lora" in n:
                p.requires_grad_(True)
    pol.keep_think_history = True  # multi_view (as the v4 configs)
    for multi in (False, True):
        full = _run(pol, 0, multi)
        ref = full[2]
        perturbed = [(i, v - 0.1, t) for i, v, t in ref]  # a reference != the policy: KL > 0
        full = _run(pol, 0, multi, kl_ref=perturbed)
        chunk = _run(pol, 3, multi, kl_ref=perturbed)
        assert torch.allclose(full[0], chunk[0], atol=1e-5), multi
        assert torch.allclose(full[1], chunk[1], atol=1e-5), multi
        assert float(full[1].abs().sum()) > 0
        for (i1, v1, t1), (i2, v2, t2) in zip(full[2], chunk[2], strict=True):
            assert torch.equal(i1, i2) and torch.allclose(v1, v2, atol=1e-5)
            assert torch.allclose(t1, t2, atol=1e-5)
        assert full[3].keys() == chunk[3].keys() and full[3]
        for n in full[3]:
            assert torch.allclose(full[3][n], chunk[3][n], atol=1e-5), (multi, n)
    assert ModelConfig(path="x").lm_chunk == 0
