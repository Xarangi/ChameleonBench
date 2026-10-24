"""Gemma-4 12B port (10-24; docs/tica/GEMMA4_12B.md), CPU only:
- the Qwen keep-think render is unchanged (golden string) and stays the default chat format;
- the Gemma-4 keep-think render: golden string, prefix property (generation prompt + content),
  token-for-token agreement with the snapshot's real chat template (when on disk);
- a tiny random Gemma4UnifiedForCausalLM (sliding + full attention, k = v, soft-capping): layer
  path, single pass == per-turn views in the Gemma format, cached reads (shared prefix /
  incremental, longer than the sliding window) == plain reads, truncated reads, chunked LM terms
  WITH soft-capping == the full-logits path;
- text-only loading of a multimodal-layout checkpoint (renamed keys, vision weights skipped,
  missing weights raise); with the real snapshot, text-only keys == the checkpoint's text keys;
- merged-weight / adapter key mapping for vLLM (model.language_model.*, layer_scalar, no lm_head);
- probe-zoo depth rescaling and the Gemma configs (depth checks, mirrors, tables restored).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
pytest.importorskip("transformers.models.gemma4_unified")

sys.path.insert(0, "tests")

from next_chameleons.tica.policy import (  # noqa: E402
    GEMMA4_TURNS,
    QWEN_CHATML,
    Policy,
    chat_format_for,
    find_decoder_layers,
    load_causal_lm,
    render_keep_reasoning,
    render_keep_reasoning_gemma4,
)

ROOT = Path(__file__).resolve().parents[1]
SNAP = Path("/scratch/s/sneheel/hf/hub/models--google--gemma-4-12b-it/snapshots/"
            "5926caa4ec0cac5cbfadaf4077420520de1d5205")
EOS, PAD, VOCAB = 257, 258, 260

@pytest.fixture(autouse=True)
def _zoo_tables_back_to_32():
    """Loading a depth-48 config rescales the module-level zoo tables: restore them after every
    test, so later test files see the 32-layer originals."""

    yield
    from next_chameleons.tica.probe_zoo import set_zoo_depth

    set_zoo_depth(32)


CONV = [{"role": "system", "content": "S"},
        {"role": "user", "content": " u0 "},
        {"role": "assistant", "content": "a0", "reasoning": ""},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1", "reasoning": "r1"}]


# ------------------------------------------------------------------------------------- renders
def test_qwen_render_unchanged_golden():
    want = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nu0<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\na0<|im_end|>\n"
            "<|im_start|>user\nu1<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nr1\n</think>\n\na1<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert render_keep_reasoning(CONV, True, False) == want
    assert QWEN_CHATML.render is render_keep_reasoning
    assert QWEN_CHATML.assistant_head == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert QWEN_CHATML.end_of_turn == "<|im_end|>"


def test_qwen_policy_render_is_the_old_function():
    from test_tica_framework import CharTok, tiny_policy

    assert chat_format_for(CharTok()) is QWEN_CHATML  # no template = the pre-10-24 behaviour
    assert chat_format_for(SimpleNamespace(chat_template="{{ '<|im_start|>' }}")) is QWEN_CHATML
    pol = tiny_policy()
    pol.keep_think_history = True
    assert pol.chat_format is QWEN_CHATML and pol.logit_softcap is None
    assert pol.render(CONV, True) == render_keep_reasoning(CONV, True, False)


def test_gemma4_render_golden_and_prefix():
    want = ("<bos><|turn>system\nS<turn|>\n<|turn>user\nu0<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>a0<turn|>\n"
            "<|turn>user\nu1<turn|>\n"
            "<|turn>model\n<|channel>thought\nr1\n<channel|>a1<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>")
    assert render_keep_reasoning_gemma4(CONV, True, False) == want
    # keep-think (every reasoning ""): each finished turn = its generation prompt + content
    conv = [dict(m, reasoning="") if m["role"] == "assistant" else m for m in CONV]
    full = render_keep_reasoning_gemma4(conv, False, False)
    for i, m in enumerate(conv):
        if m["role"] == "assistant":
            assert full.startswith(render_keep_reasoning_gemma4(conv[:i], True, False)
                                   + m["content"] + "<turn|>\n")
    # thinking on: <|think|> opens a system turn, the generation prompt has no empty channel
    s = render_keep_reasoning_gemma4([{"role": "user", "content": "q"}], True, True)
    assert s == "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nq<turn|>\n<|turn>model\n"
    with pytest.raises(ValueError):
        render_keep_reasoning_gemma4([{"role": "user", "content": "q"},
                                      {"role": "system", "content": "s"}], False, False)


@pytest.mark.skipif(not SNAP.is_dir(), reason="Gemma-4 snapshot not on disk")
def test_gemma4_render_matches_real_template():
    tok = transformers.AutoTokenizer.from_pretrained(str(SNAP))
    assert chat_format_for(tok) is GEMMA4_TURNS
    cases = [
        [{"role": "user", "content": "Hello there"}],
        [{"role": "system", "content": "You are terse."}, {"role": "user", "content": " Hi "},
         {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Again?"}],
    ]
    for conv in cases:
        for think in (False, True):
            ref = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=think)
            assert render_keep_reasoning_gemma4(conv, True, think) == ref, (conv, think)
            assert (tok(render_keep_reasoning_gemma4(conv, True, think),
                        add_special_tokens=False).input_ids
                    == tok(ref, add_special_tokens=False).input_ids)
    # the markers multi_view relies on are single special tokens
    for s in ("<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>"):
        assert len(tok(s, add_special_tokens=False).input_ids) == 1, s
    assert tok("<turn|>", add_special_tokens=False).input_ids[0] == 106  # an EOS id of the model


# ------------------------------------------------------------------------------- tiny Gemma
class GemmaCharTok:
    """Char-level tokenizer whose template marks it as Gemma-4 (chat_format_for -> GEMMA4_TURNS)."""

    eos_token_id, pad_token_id = EOS, PAD
    eos_token, pad_token = "<eos>", "<pad>"
    chat_template = "{{ '<|turn>' }} {{ '<|channel>' }}"

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        return render_keep_reasoning_gemma4(conv, add_generation_prompt, enable_thinking)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[min(ord(c), 255) for c in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i < 256)


def tiny_gemma_cfg(vocab=VOCAB, softcap=30.0, window=6):
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedTextConfig,
    )

    return Gemma4UnifiedTextConfig(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=32,
        num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=window,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        final_logit_softcapping=softcap, pad_token_id=PAD, eos_token_id=EOS, bos_token_id=1,
        max_position_embeddings=4096)


def tiny_gemma_model(cfg=None):
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedForCausalLM,
    )

    torch.manual_seed(0)
    m = Gemma4UnifiedForCausalLM(cfg or tiny_gemma_cfg())
    with torch.no_grad():  # non-trivial per-layer scalars (a missed load would show)
        for i, layer in enumerate(m.model.layers):
            layer.layer_scalar.fill_(1.0 + 0.1 * i)
    return m.eval()


def tiny_gemma_policy(lora=True, softcap=30.0, window=6):
    model = tiny_gemma_model(tiny_gemma_cfg(softcap=softcap, window=window))
    if lora:
        model = peft.get_peft_model(model, peft.LoraConfig(
            r=4, lora_alpha=8, target_modules="all-linear", lora_dropout=0.0))
        with torch.no_grad():
            g = torch.Generator().manual_seed(5)
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    model.eval()
    return Policy(model, GemmaCharTok(), batch_size=4, keep_think_history=True)


def _convs(n=2, turns=3):
    out = []
    for k in range(n):
        c = [{"role": "system", "content": "monitor sys"}]
        for t in range(turns):
            c += [{"role": "user", "content": f"turn {t} say w{k}{t}"},
                  {"role": "assistant", "content": f"w{k}{t} " * (2 + t + k)}]
        out.append(c)
    return out


def test_tiny_gemma_layers_lora_and_format():
    pol = tiny_gemma_policy()
    assert pol.chat_format is GEMMA4_TURNS and pol.logit_softcap == 30.0
    assert len(pol.layers_module) == 4
    assert pol.layers_module is find_decoder_layers(pol.model)
    names = {n for n, _ in pol.model.named_modules() if n.endswith("lora_A")}
    assert names and all(".model.layers." in n for n in names)  # text decoder only
    assert not any("lm_head" in n for n in names)


def test_tiny_gemma_single_pass_matches_views():
    pol = tiny_gemma_policy()
    convs = _convs()
    mvs = [pol.multi_view(c) for c in convs]
    views = []
    for c, mv in zip(convs, mvs, strict=True):
        idx = [i for i, m in enumerate(c) if m["role"] == "assistant"]
        for t, i in enumerate(idx):
            v = pol.view(c[: i + 1])
            st, en, te = mv.spans[t]
            assert v.ids == mv.ids[:te] and v.start == st and v.end == en
            views.append(v)
    pm, nm = pol.forward_multi(mvs, [1, 3], grad=False, with_nll=True)
    pv, nv = pol.forward(views, [1, 3], grad=False, with_nll=True)
    for L in (1, 3):
        assert torch.allclose(pm[L], pv[L], atol=1e-4)
    assert torch.allclose(nm, nv, atol=1e-4)


def test_tiny_gemma_cached_and_truncated_reads():
    """Reads with KV caches (sliding window 6, sequences far longer) and truncated reads equal plain
    reads -- the rollout / refit read paths on the sliding-window cache."""

    pol = tiny_gemma_policy()
    pre = [{"role": "user", "content": "tell me about cats " * 4},
           {"role": "assistant", "content": "cats purr and meow " * 3}]
    forks = [[*pre, {"role": "user", "content": f"next {k}"},
              {"role": "assistant", "content": f"answer number {k} " * (k + 1)}] for k in range(4)]
    a = pol.pooled(forks, [1, 2])
    b = pol.pooled_shared_prefix(forks, [1, 2], min_prefix=1)
    for L in (1, 2):
        assert torch.allclose(a[L], b[L], atol=1e-4, rtol=1e-3), L
    pol.truncate_reads = True
    c = pol.pooled(forks, [1, 2])
    pol.truncate_reads = False
    for L in (1, 2):
        assert torch.allclose(a[L], c[L], atol=1e-5), L
    convs = [[{"role": "user", "content": f"topic {k} " * (k + 2)}] for k in range(3)]
    state = {}
    for t in range(3):
        for k, cv in enumerate(convs):
            cv.append({"role": "assistant", "content": f"answer {t} for {k} " * (t + k + 1)})
        x = pol.pooled(convs, [1])[1]
        y = pol.pooled_incremental(convs, [1], state)[1]
        assert torch.allclose(x, y, atol=1e-4, rtol=1e-3), t
        for cv in convs:
            cv.append({"role": "user", "content": f"feedback {t}"})
    assert state["hits"] == 6


def test_tiny_gemma_chunked_lm_terms_apply_softcap():
    """model.lm_chunk calls the LM head itself: with Gemma's final-logit soft-capping the chunked
    NLL / KL / top-k targets / grads must equal the model's own (soft-capped) logits."""

    from test_tica_lm_chunk import _run

    pol = tiny_gemma_policy(softcap=2.0)  # a cap that bites at random-init logit scales
    for n, p in pol.model.named_parameters():
        if "lora" in n:
            p.requires_grad_(True)
    with torch.no_grad():  # larger logits so tanh(x / cap) is far from linear
        pol.model.get_output_embeddings().weight.mul_(20.0)
    for multi in (False, True):
        full = _run(pol, 0, multi)
        pert = [(i, v - 0.1, t) for i, v, t in full[2]]
        full = _run(pol, 0, multi, kl_ref=pert)
        chunk = _run(pol, 3, multi, kl_ref=pert)
        assert torch.allclose(full[0], chunk[0], atol=1e-5), multi
        assert torch.allclose(full[1], chunk[1], atol=1e-5), multi
        for n in full[3]:
            assert torch.allclose(full[3][n], chunk[3][n], atol=1e-5), (multi, n)
        pol.logit_softcap = None  # without the cap the chunked path disagrees (the bug it fixes)
        bad = _run(pol, 3, multi, kl_ref=pert)
        pol.logit_softcap = 2.0
        assert not torch.allclose(full[0], bad[0], atol=1e-3), multi


# ------------------------------------------------------------------------------- loading
def _save_multimodal_layout(model, path: Path, drop: str | None = None) -> None:
    """A gemma4_unified checkpoint dir around a tiny text model: text weights under
    model.language_model.*, a fake vision weight, tied lm_head absent (as the real snapshot)."""

    from safetensors.torch import save_file
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig

    path.mkdir(parents=True, exist_ok=True)
    Gemma4UnifiedConfig(text_config=model.config.to_dict()).save_pretrained(str(path))
    sd = {"model.language_model." + k[len("model."):]: v.contiguous()
          for k, v in model.state_dict().items() if k.startswith("model.")}
    sd["model.vision_embedder.patch_dense.weight"] = torch.zeros(4, 4)
    if drop:
        sd.pop(drop)
    save_file(sd, str(path / "model.safetensors"))


def test_text_only_load_of_multimodal_layout(tmp_path):
    ref = tiny_gemma_model()
    _save_multimodal_layout(ref, tmp_path / "ok")
    cj = json.loads((tmp_path / "ok" / "config.json").read_text())
    assert cj["model_type"] == "gemma4_unified"
    m = load_causal_lm(str(tmp_path / "ok"), "float32")
    assert type(m).__name__ == "Gemma4UnifiedForCausalLM"
    assert not any("vision" in n or "audio" in n for n, _ in m.named_modules())
    assert m.config._name_or_path == str(tmp_path / "ok")
    want = ref.state_dict()
    got = m.state_dict()
    assert set(got) == set(want)
    for k in want:
        assert torch.equal(got[k].float().cpu(), want[k].float()), k
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()  # still tied
    _save_multimodal_layout(ref, tmp_path / "bad",
                            drop="model.language_model.layers.1.mlp.up_proj.weight")
    with pytest.raises(RuntimeError, match="missing"):
        load_causal_lm(str(tmp_path / "bad"), "float32")


@pytest.mark.skipif(not SNAP.is_dir(), reason="Gemma-4 snapshot not on disk")
def test_real_text_keys_match_checkpoint():
    """The 12B text-only model (meta device, no memory) has exactly the snapshot's text keys after
    the load renaming, and the vLLM merged-state naming maps them back onto the same keys."""

    import struct

    from transformers import AutoConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedForCausalLM,
    )

    with open(SNAP / "model.safetensors", "rb") as f:
        hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    ckpt = {k for k in hdr if k != "__metadata__"}
    text = {k for k in ckpt if k.startswith("model.language_model.")}
    assert all(k.startswith(("model.language_model.", "model.vision_embedder.",
                             "model.embed_vision.", "model.embed_audio.")) for k in ckpt)
    cfg = AutoConfig.from_pretrained(str(SNAP)).get_text_config()
    assert cfg.num_hidden_layers == 48 and cfg.hidden_size == 3840
    with torch.device("meta"):
        m = Gemma4UnifiedForCausalLM(cfg)
    keys = {k for k in m.state_dict() if not k.startswith("lm_head.")}
    assert {"model.language_model." + k[len("model."):] for k in keys} == text
    params = {n for n, _ in m.named_parameters() if not n.startswith("lm_head.")}
    scal = {n for n, _ in m.named_buffers() if n.endswith("layer_scalar")}
    assert len(scal) == 48 and params | scal == keys  # merged state = params + layer_scalar


def test_merged_state_and_adapter_keys(tmp_path):
    from safetensors.torch import load_file

    from next_chameleons.tica.vllm_gen import convert_adapter, merged_text_state, model_family

    pol = tiny_gemma_policy()
    ref = tiny_gemma_model()
    ckpt_keys = {"model.language_model." + k[len("model."):] for k in ref.state_dict()
                 if k.startswith("model.")}
    st = merged_text_state(pol, ("layer_scalar",))
    assert set(st) == ckpt_keys  # every text weight + the per-layer scalars, no lm_head
    old = merged_text_state(pol)  # the Qwen call: parameters only (unchanged)
    assert set(st) - set(old) == {k for k in ckpt_keys if k.endswith("layer_scalar")}
    base = pol.model.get_base_model()
    mod = base.model.layers[0].self_attn.q_proj
    w = (mod.base_layer.weight.float() + mod.scaling["default"]
         * (mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float()))
    assert torch.allclose(st["model.language_model.layers.0.self_attn.q_proj.weight"].float(), w)
    pol.save_adapter(tmp_path / "hf")
    convert_adapter(tmp_path / "hf", tmp_path / "vllm")
    keys = set(load_file(str(tmp_path / "vllm" / "adapter_model.safetensors")))
    assert keys and all(k.startswith("base_model.model.model.language_model.layers.") for k in keys)
    _save_multimodal_layout(ref, tmp_path / "g4")
    fam = model_family(tmp_path / "g4")
    assert "--language-model-only" in fam["flags"] and fam["buffers"] == ("layer_scalar",)
    (tmp_path / "q").mkdir()
    (tmp_path / "q" / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert model_family(tmp_path / "q") == {} and model_family(tmp_path / "none") == {}


# ------------------------------------------------------------------------- zoo depth / configs
def test_zoo_depth_rescale_and_restore():
    from next_chameleons.tica import probe_zoo as z

    orig = {"ens4": [8, 10, 12, 14], "ens8": [4, 6, 8, 10, 12, 14, 16, 18],
            "ensdeep4": [16, 18, 22, 26], "spread5": [4, 10, 16, 22, 28]}
    try:
        z.set_zoo_depth(48)
        assert z.ENS_LAYERS == {"ens4": [12, 15, 18, 21], "ens8": [6, 9, 12, 15, 18, 21, 24, 27],
                                "ensdeep4": [24, 27, 33, 39], "spread5": [6, 15, 24, 33, 42]}
        assert z.TOKENS_ENS_LAYERS["bailey_ens_tok"] == list(range(48))
        assert z.TOKENS_ENS_LAYERS["bailey_ens_tok_max"] == list(range(6, 43))
        assert z.kind_layers("ensdeep4_linear", 18) == [24, 27, 33, 39]
        assert [z.scale_layer(L, 48) for L in (8, 12, 16, 20, 22, 24, 26, 28, 30)] == \
            [12, 18, 24, 30, 33, 36, 39, 42, 45]
        assert z.scale_kind("maxpool@24", 48) == "maxpool@36"
        assert z.scale_kind("attnpool", 48) == "attnpool"
    finally:
        z.set_zoo_depth(32)
    tok32 = {"nc_ens8_tok": orig["ens8"], "bailey_ens_tok": list(range(32)),
             "bailey_ens_tok_max": list(range(4, 29))}
    assert dict(z.ENS_LAYERS) == orig
    assert dict(z.TOKENS_ENS_LAYERS) == tok32


@pytest.mark.parametrize("name", ["final_v4c_gemma4_12b_tamia", "final_v4c_score_gemma4_12b_tamia",
                                  "mathrl_gemma4_12b"])
def test_gemma_configs_mirrored(name):
    a = (ROOT / "configs" / "tica" / f"{name}.yaml").read_bytes()
    b = (ROOT / "src/next_chameleons/builtin_configs/tica" / f"{name}.yaml").read_bytes()
    assert a == b


def test_gemma_configs_load_and_restore_tables():
    from next_chameleons.tica import probe_zoo as z
    from next_chameleons.tica.config import load_run_config

    a = load_run_config(str(ROOT / "configs/tica/final_v4c_gemma4_12b_tamia.yaml"))
    b = load_run_config(str(ROOT / "configs/tica/final_v4c_score_gemma4_12b_tamia.yaml"))
    assert a.model.depth == 48 and a.probes.layer == 18 and a.probes.dir == ""
    assert list(a.probes.held_out_layers) == [30, 36]
    assert list(a.rl.adaptive.kind_layers) == [12, 18, 24, 33, 39]
    assert z.ENS_LAYERS["ensdeep4"] == [24, 27, 33, 39]  # rescaled by loading the config
    assert len(a.episode_concepts) == 20 and a.episode_concepts == b.episode_concepts
    assert a.episodes.feedback == "verdict" and b.episodes.feedback == "score"
    assert a.out != b.out and "gemma4_12b" in a.rl.adaptive.anchor_cache
    assert "mathrl_gemma4_12b" in a.rl.adaptive.cap_gen_pool
    # everything else equals the production v4c 4B arm
    q = load_run_config(str(ROOT / "configs/tica/final_v4b_tamia_4b.yaml"))
    assert z.ENS_LAYERS["ensdeep4"] == [16, 18, 22, 26]  # a Qwen config restores the 32-layer sets
    for sec in ("lora", "train"):
        x, y = vars(getattr(a, sec)).copy(), vars(getattr(q, sec)).copy()
        x.pop("steps", None)
        y.pop("steps", None)
        assert x == y, sec
    assert a.train_mix == q.train_mix and a.final.train_refit_kinds == q.final.train_refit_kinds
    ra, rq = a.rl.adaptive, q.rl.adaptive
    diff = {k for k in vars(ra) if getattr(ra, k) != getattr(rq, k)}
    assert diff == {"kind_layers", "eval_on_save", "anchor_cache", "cap_gen_pool"}, diff


def test_depth_check_catches_unscaled_layers():
    from next_chameleons.config import load_yaml
    from next_chameleons.tica.config import RunConfig, from_dict, resolve_final

    d = load_yaml(ROOT / "configs/tica/final_v4c_gemma4_12b_tamia.yaml")
    d["rl"]["adaptive"]["kind_layers"] = [12, 18, 24, 33, 50]
    resolve_final(d, ROOT / "configs/tica")
    with pytest.raises(ValueError, match="layers outside a 48-layer model"):
        from_dict(RunConfig, d)
    d["rl"]["adaptive"]["kind_layers"] = [12, 18, 30]  # a held-out depth as a training depth
    with pytest.raises(ValueError, match="held-out"):
        from_dict(RunConfig, d)
