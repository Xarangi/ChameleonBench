"""crypsis.models == next_chameleons.tica.policy on tiny random Qwen3.5 / Gemma-4 models with the
real tokenizers (CPU, fp32, bit-equal). Skips when the tokenizer files are not on disk."""

from __future__ import annotations

import glob
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(1)  # tiny tensors; login-node cgroups cap CPU (64 threads thrash)

from crypsis.models.chat import SYS_CLOSE, SYS_OPEN, chat_prompt_ids  # noqa: E402
from crypsis.models.generation import HFGenerator, generate_batched  # noqa: E402
from crypsis.models.policy import Policy  # noqa: E402
from next_chameleons.tica import policy as old  # noqa: E402

HUB = "/scratch/s/sneheel/hf/hub"


def _snap(name):
    s = sorted(glob.glob(f"{HUB}/models--{name}/snapshots/*/tokenizer.json"))
    return os.path.dirname(s[0]) if s else None


QWEN, GEMMA = _snap("Qwen--Qwen3.5-2B"), _snap("google--gemma-4-12b-it")


def _tiny(kind, tok):
    torch.manual_seed(0)
    if kind == "qwen":
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        cfg = Qwen3_5TextConfig(
            vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            max_position_embeddings=4096)
        m = Qwen3_5ForCausalLM(cfg)
    else:
        from transformers.models.gemma4_unified.configuration_gemma4_unified import (
            Gemma4UnifiedTextConfig,
        )
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedForCausalLM,
        )

        cfg = Gemma4UnifiedTextConfig(
            vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=32,
            num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=6,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            final_logit_softcapping=30.0, pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
            max_position_embeddings=4096)
        m = Gemma4UnifiedForCausalLM(cfg)
    m = peft.get_peft_model(m, peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
    with torch.no_grad():
        g = torch.Generator().manual_seed(5)
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return m.eval()


_CACHE: dict = {}


def setup(kind, **kw):
    """(old Policy, new Policy) on ONE shared tiny model."""

    path = QWEN if kind == "qwen" else GEMMA
    if path is None:
        pytest.skip(f"{kind} tokenizer not on disk")
    if kind not in _CACHE:
        tok = transformers.AutoTokenizer.from_pretrained(path)
        _CACHE[kind] = (tok, _tiny(kind, tok))
    tok, m = _CACHE[kind]
    return old.Policy(m, tok, batch_size=2, **kw), Policy(m, tok, batch_size=2, **kw)


SYS = f"{SYS_OPEN}be brief{SYS_CLOSE}"
CONVS = [
    [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "hello friend"}],
    [{"role": "user", "content": SYS + "Monitor: ON.\n\nq one"},
     {"role": "assistant", "content": "answer one", "reasoning": "think a"},
     {"role": "user", "content": "q two"}, {"role": "assistant", "content": "answer two ok"}],
    [{"role": "system", "content": "sys"}, {"role": "user", "content": " spaced "},
     {"role": "assistant", "content": "x y z"}],
]
# keep-think reads need assistant turns without reasoning (multi_view raises otherwise, old and new)
PLAIN = [[{k: v for k, v in m.items() if k != "reasoning"} for m in c] for c in CONVS]
FORKS = [[*PLAIN[1][:3], {"role": "user", "content": f"fork {k}"},
          {"role": "assistant", "content": "w " * (k + 1)}] for k in range(3)]
LAYOUTS = [dict(), dict(enable_thinking=True), dict(history_reasoning=True),
           dict(keep_think_history=True)]


def _fields(fn, c):
    """View / MultiView fields (the classes differ between old and new), or the error raised."""

    try:
        v = fn(c)
    except ValueError as e:
        return ("raises", str(e))
    return (v.ids, v.spans) if hasattr(v, "spans") else (v.ids, v.start, v.n_tail)


def _eq(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
def test_render_and_views(kind):
    for lay in LAYOUTS:
        o, n = setup(kind, **lay)
        for c in CONVS:
            for gp in (False, True):
                assert n.chat.render(c, gp) == o.render(c, gp), (lay, c, gp)
            assert n.chat.prompt_ids(c[:-1]) == o.prompt_ids(c[:-1])
            assert _fields(n.chat.view, c) == _fields(o.view, c), (lay, c)
            if lay.get("keep_think_history"):
                assert _fields(n.chat.multi_view, c) == _fields(o.multi_view, c)
        assert n.chat_format.name == o.chat_format.name


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
def test_read_paths(kind):
    o, n = setup(kind, keep_think_history=True)
    L = [1, 3]
    for chunk in (0, 3):
        o.lm_chunk = chunk
        r = n.reader(lm_chunk=chunk)
        views = [o.view(c) for c in PLAIN]
        eo, en = {"topk": 4}, {"topk": 4}
        po, no = o.forward(views, L, grad=False, with_nll=True, extra=eo)
        pn, nn = r.forward(views, L, grad=False, with_nll=True, extra=en)
        _eq(po, pn)
        assert torch.equal(no, nn)
        for a, b in zip(eo["ref"], en["ref"], strict=True):
            assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        mvs = [o.multi_view(c) for c in PLAIN]
        to, tn = {}, {}
        po, no = o.forward_multi(mvs, L, grad=False, with_nll=True, token_out=to)
        pn, nn = r.forward_multi(mvs, L, grad=False, with_nll=True, token_out=tn)
        _eq(po, pn)
        assert torch.equal(no, nn)
        assert all(torch.equal(x, y) for k in L for x, y in zip(to[k], tn[k], strict=True))
    for trunc in (False, True):
        o.truncate_reads = trunc
        r = n.reader(truncate=trunc)
        _eq(o.pooled(PLAIN, L), r.pooled(PLAIN, L))
        _eq(o.pooled(PLAIN, [1]), r.pooled(PLAIN, [1]))
        for srt in (False, True):
            o.sort_reads = srt
            r.sort_reads = srt
            to, tn = {}, {}
            _eq(o.pooled_shared_prefix(FORKS, L, min_prefix=1, token_out=to, token_keep=2),
                r.pooled_shared_prefix(FORKS, L, min_prefix=1, token_out=tn, token_keep=2))
            assert all(torch.equal(x, y) for k in L for x, y in zip(to[k], tn[k], strict=True))
        so, sn = {}, {}
        for convs in ([f[:2] for f in FORKS], FORKS):  # growing: the 2nd call reuses the caches
            _eq(o.pooled_incremental(convs, L, so), r.pooled_incremental(convs, L, sn))
        assert so["hits"] == sn["hits"] == 3
    o.truncate_reads = False


def test_reference_merged_and_adapter_roundtrip(tmp_path):
    o, n = setup("qwen")
    r = n.reader()
    base = r.pooled(CONVS, [2])
    for ctx_o, ctx_n in ((o.reference, n.reference), (o.merged, n.merged)):
        with ctx_o():
            want = o.pooled(CONVS, [2])
        with ctx_n():
            got = r.pooled(CONVS, [2])
            if ctx_n == n.merged:
                with n.reference():
                    assert n.in_reference
                    ref_in = r.pooled(CONVS, [2])
        _eq(want, got)
    with n.reference():
        _eq(ref_in, r.pooled(CONVS, [2]))
    _eq(base, r.pooled(CONVS, [2]))  # weights restored after merged()
    n.save_adapter(tmp_path / "a")
    m2 = peft.PeftModel.from_pretrained(_tiny("qwen", n.tokenizer).get_base_model(),
                                        str(tmp_path / "a"))
    sd1 = {k: v for k, v in n.model.state_dict().items() if "lora" in k}
    sd2 = {k: v for k, v in m2.state_dict().items() if "lora" in k}
    assert sd1.keys() == sd2.keys() and all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_load_equiv(tmp_path):
    _, n = setup("qwen")
    n.model.get_base_model().save_pretrained(str(tmp_path / "m"))
    n.tokenizer.save_pretrained(str(tmp_path / "m"))
    lora = {"rank": 4, "alpha": 8}
    torch.manual_seed(1)
    po = old.Policy.from_pretrained(str(tmp_path / "m"), dtype="float32", lora=lora,
                                    keep_think_history=True)
    torch.manual_seed(1)
    pn = Policy.load(str(tmp_path / "m"), dtype="float32", lora=lora, keep_think_history=True)
    so, sn = po.model.state_dict(), pn.model.state_dict()
    assert so.keys() == sn.keys() and all(torch.equal(so[k], sn[k]) for k in so)
    assert len(pn.layers) == 4 and pn.dtype == torch.float32 and pn.has_adapter


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
def test_generation(kind):
    o, n = setup(kind)
    prompts = [CONVS[0][:1], CONVS[1][:3], CONVS[2][:2]]
    want = o.generate(prompts, 6)
    got = HFGenerator(n).generate(prompts, 6)
    assert [g.as_dict() for g in got] == want

    class NoThink:  # the eval scripts' proxy tokenizer
        def __init__(self, t):
            self._tok = t

        def __getattr__(self, k):
            return getattr(self._tok, k)

        def apply_chat_template(self, *a, **kw):
            kw.setdefault("enable_thinking", False)
            return self._tok.apply_chat_template(*a, **kw)

    from next_chameleons.benchmarks.capability import _chat_prompt_ids
    from next_chameleons.benchmarks.capability_heldout import generate_batched as old_gb

    users = ["what is 2+2", SYS + "q"]
    ids_o = [_chat_prompt_ids(NoThink(n.tokenizer), u) for u in users]
    ids_n = [chat_prompt_ids(n.tokenizer, u, enable_thinking=False) for u in users]
    assert ids_o == ids_n
    assert (generate_batched(n.model, n.tokenizer, ids_n, batch_size=2, max_new_tokens=5)
            == old_gb(n.model, n.tokenizer, ids_o, batch_size=2, max_new_tokens=5))
