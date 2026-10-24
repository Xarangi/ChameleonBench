"""Gemma-4 12B port smoke (10-24; docs/tica/GEMMA4_12B.md): ONE GPU, aggregates only (no text).

  PYTHONPATH=src python scripts/tica_gemma_smoke.py --config final_v4c_gemma4_12b_tamia \
      --model-path LOCAL_SNAPSHOT [--concepts 1] [--no-vllm]

Checks, each reported as one JSON line (and all together in OUT/smoke.json):
 1. load: text-only class, depth == model.depth, LoRA r / alpha, trainable params, memory;
 2. render: generation-prompt text / ids (keep-think render) == the real chat template's;
 3. reads on REAL 24-turn episodes (training texts, tagged layout, verdict lines): one forward
    over the conversation (training) == per-turn views; shared-prefix fork reads and incremental
    rollout reads == plain reads (KV caches across the 1024-token sliding window); truncated
    reads == full reads;
 4. base probes: the answer-format base probes (primary + held-out depths) on the BASE model for
    N concepts (fit_answer_probes, the trainer's path; production AUROC / TPR gates);
 5. one training step at production episode length: forward_multi with grad (every training
    depth + LM terms with lm_chunk + soft-capping), backward, AdamW on the LoRA; peak memory;
 6. HF generation (cap_gen path: LoRA merged): tokens/s at batch 16;
 7. vLLM side engine (TICA_VLLM_PY set; text-only, merged-weights mode): startup, sleep / wake,
    merged-weight sync after the optimizer step, greedy agreement with HF (policy and base),
    memory while both are resident.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

RES: dict = {}


def report(key: str, **kw) -> None:
    RES[key] = kw
    print(json.dumps({key: kw}, default=str), flush=True)


def gb(x: float) -> float:
    return round(x / 2**30, 2)


def mem() -> dict:
    return {"alloc_gb": gb(torch.cuda.memory_allocated()),
            "peak_gb": gb(torch.cuda.max_memory_allocated()),
            "reserved_gb": gb(torch.cuda.memory_reserved()),
            "device_used_gb": gb(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0])}


def episode(data_concept, others, opening, feedback, turns: int, rng: random.Random):
    """A real-shaped 24-turn episode: tagged opening, verdict lines, the concept's training texts
    as the (forced) answers; a few turns of other concepts' texts (switch-like)."""

    texts = list(data_concept.texts)
    rng.shuffle(texts)
    c = data_concept.concept
    conv = [{"role": "user", "content": opening("real_score", texts[0], c)},
            {"role": "assistant", "content": texts[0]}]
    for t in range(1, turns):
        nxt = texts[t % len(texts)] if t % 6 else rng.choice(others)
        shown = (rng.uniform(0, 1), 0.5)
        conv += [{"role": "user", "content": feedback("real_score", c, shown, nxt,
                                                      feedback="verdict")},
                 {"role": "assistant", "content": nxt}]
    return conv


def cos_min(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).min())


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--concepts", type=int, default=1)
    ap.add_argument("--turns", type=int, default=0, help="0 = the config's train.turns")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from tica import build

    from next_chameleons.tica.answer_format import fit_answer_probes, prompts
    from next_chameleons.tica.config import load_run_config

    t0 = time.time()
    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    data, _, factory = build(cfg)
    torch.cuda.reset_peak_memory_stats()
    pol = factory(None, True)  # production load: LoRA r128 / alpha 64, grad ckpt, lm_chunk
    pol.model.eval()  # reads as in the rollout (HF drops the KV cache in train mode + ckpt)
    base = pol.model.get_base_model()
    n_train = sum(p.numel() for p in pol.model.parameters() if p.requires_grad)
    report("load", sec=round(time.time() - t0, 1), cls=type(base).__name__,
           layers=len(pol.layers_module), depth=cfg.model.depth,
           vision_or_audio_modules=sum(("vision" in n or "audio" in n)
                                       for n, _ in pol.model.named_modules()),
           chat_format=pol.chat_format.name, softcap=pol.logit_softcap,
           lora_r=cfg.lora.rank, lora_alpha=cfg.lora.alpha, trainable_m=round(n_train / 1e6, 1),
           eos_ids=sorted(pol.eos_ids), **mem())

    # ---- 2. render vs the real template
    tok = pol.tokenizer
    opening, feedback = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    concepts = data.concepts("fit", ("train",), only=set(cfg.episode_concepts))
    rng = random.Random(0)
    negs = data.negatives("fit")
    c0 = concepts[0]
    probe_conv = [{"role": "user", "content": opening("real_score", c0.texts[0], c0.concept)}]
    from next_chameleons.tica.answer_format import split_system

    ref_text = tok.apply_chat_template(split_system(probe_conv), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    ref_ids = pol.encode(ref_text)
    ours = pol.prompt_ids(probe_conv)
    report("render", gen_prompt_text_equal=pol.render(probe_conv, True) == ref_text,
           gen_prompt_ids_equal=ours == ref_ids, n_prompt_tokens=len(ours),
           starts_with_bos=ours[:1] == [tok.bos_token_id],
           marker_ids={s: tok(s, add_special_tokens=False).input_ids
                       for s in ("<|turn>", "<turn|>", "<|channel>", "<channel|>")})

    # ---- 3. reads on real episodes
    turns = a.turns or cfg.train.turns
    convs = [episode(concepts[k % len(concepts)], negs, opening, feedback, turns, rng)
             for k in range(2)]
    mvs = [pol.multi_view(c) for c in convs]
    layers = sorted({*map(int, cfg.rl.adaptive.kind_layers), *map(int, cfg.probes.held_out_layers)})
    t = time.time()
    pm, nm = pol.forward_multi(mvs, layers, grad=False, with_nll=True)
    t_multi = time.time() - t
    views = []
    for c in convs:
        idx = [i for i, m in enumerate(c) if m["role"] == "assistant"]
        views += [pol.view(c[: i + 1]) for i in idx[-4:]]  # the last 4 turns of each (longest)
    pv, nv = pol.forward(views, layers, grad=False, with_nll=True)
    n_sp = len(mvs[0].spans)
    sel = [k * n_sp + r for k in range(len(mvs)) for r in range(n_sp)[-4:]]
    single = {"cos_min": min(cos_min(pm[L][sel], pv[L]) for L in layers),
              "nll_max_abs": float((nm[sel] - nv).abs().max())}
    # forks of one conversation (shared prefix) + incremental reads + truncated reads
    pre = convs[0][:-2]
    forks = [[*pre, {"role": "user", "content": convs[0][-2]["content"]},
              {"role": "assistant", "content": tx}] for tx in rng.sample(list(c0.texts), 6)]
    plain = pol.pooled(forks, layers, batch=6)
    shared = pol.pooled_shared_prefix(forks, layers, min_prefix=64, batch=6)
    pol.truncate_reads = True
    trunc = pol.pooled(forks, layers, batch=6)
    pol.truncate_reads = bool(cfg.rl.adaptive.truncate_reads)
    state, inc_cos = {}, []
    grow = [c[:2] for c in convs]
    for tt in range(1, 6):
        x = pol.pooled(grow, layers)
        y = pol.pooled_incremental(grow, layers, state)
        inc_cos.append(min(cos_min(x[L], y[L]) for L in layers))
        for k, c in enumerate(convs):
            grow[k] = c[: 2 * (tt + 1)]
    report("reads", n_tokens=[len(mv.ids) for mv in mvs], sec_forward_multi=round(t_multi, 2),
           single_pass=single,
           shared_prefix_cos_min=min(cos_min(plain[L], shared[L]) for L in layers),
           shared_prefix_rel_err=max(rel_err(shared[L], plain[L]) for L in layers),
           truncated_cos_min=min(cos_min(plain[L], trunc[L]) for L in layers),
           incremental_cos_min=min(inc_cos), incremental_hits=state.get("hits", 0),
           act_norm={L: round(float(plain[L].norm(dim=-1).mean()), 1) for L in layers}, **mem())

    # ---- 4. base probes on the base model (trainer path)
    held = [int(x) for x in cfg.probes.held_out_layers]
    plays = [cfg.primary_layer()] + [h for h in held if h != cfg.primary_layer()]
    q_all = {}
    for k, c in enumerate(cfg.episode_concepts[: a.concepts]):
        t = time.time()
        with pol.reference():
            _, quality = fit_answer_probes(cfg.rl.adaptive, data, pol, c, plays, opening,
                                           neg_splits=("train",), rename=f"answer_base_c{k}")
        q_all[f"c{k}"] = {"sec": round(time.time() - t, 1), **{
            str(L): {kk: round(float(vv), 4) for kk, vv in v.items()
                     if isinstance(vv, (int, float))}
            for L, v in quality.items()}}
    report("base_probes", layers=plays, quality=q_all)

    # ---- 5. one training step at production length
    pol.model.train()
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    params = [p for p in pol.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    with torch.no_grad():
        ref = {"topk": 64}  # KL-anchor targets as the trainer builds them
        pol.forward_multi(mvs[:1], layers, grad=False, with_nll=True, extra=ref)
    t = time.time()
    extra = {"kl_ref": ref["ref"]}
    pg, ng = pol.forward_multi(mvs[:1], layers, grad=True, with_nll=True, extra=extra)
    loss = sum(pg[L].float().pow(2).mean() for L in layers) * 1e-4 + ng.mean() + extra["kl"].mean()
    t_fwd = time.time() - t
    loss.backward()
    torch.cuda.synchronize()
    t_bwd = time.time() - t - t_fwd
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1e9))
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    pol.weights_version = getattr(pol, "weights_version", 0) + 1
    report("train_step", n_tokens=len(mvs[0].ids), n_spans=len(mvs[0].spans),
           sec_fwd=round(t_fwd, 1), sec_bwd=round(t_bwd, 1), grad_norm=round(gnorm, 4),
           finite=bool(torch.isfinite(loss).item()), loss=round(float(loss), 4),
           step_peak_gb=gb(torch.cuda.max_memory_allocated()), before_step_gb=gb(m0),
           adam_state_gb=gb(sum(s["exp_avg"].numel() * s["exp_avg"].element_size() * 2
                                for s in opt.state.values())), **mem())

    # ---- 6. HF generation (cap_gen path: merged LoRA)
    pol.model.eval()
    gconvs = [[{"role": "user",
                "content": f"Solve: what is {i + 3} times {i + 7}? Show your working."}]
              for i in range(16)]
    pol.batch_size = 16
    t = time.time()
    with pol.merged():
        hf_pol = pol.generate(gconvs, 128, temperature=0.0)
    t_hf = time.time() - t
    with pol.reference():
        hf_base = pol.generate(gconvs[:8], 64, temperature=0.0)
    report("hf_generate", sec=round(t_hf, 1),
           tok_per_s=round(sum(len(o["ids"]) for o in hf_pol) / t_hf, 1),
           mean_new_tokens=round(sum(len(o["ids"]) for o in hf_pol) / 16, 1),
           finished=sum(o["finished"] for o in hf_pol), **mem())

    # ---- 7. vLLM side engine
    if not a.no_vllm and os.environ.get("TICA_VLLM_PY"):
        from next_chameleons.tica.vllm_gen import VLLMGenerator

        work = Path(a.out or os.environ.get("SLURM_TMPDIR", "/tmp")) / "vllm_smoke"
        t = time.time()
        gen = VLLMGenerator(cfg.model.path, work,
                            mem=float(os.environ.get("TICA_VLLM_MEM", "0.4")),
                            max_len=int(os.environ.get("TICA_VLLM_MAXLEN", "16384")),
                            mode="merged")
        startup = time.time() - t
        gen.sleep()
        pol.attach_vllm(gen)
        torch.cuda.synchronize()
        m_sleep = mem()
        t = time.time()
        v_pol = pol.generate(gconvs, 128, temperature=0.0)  # syncs the merged (post-step) weights
        t_v = time.time() - t
        with pol.reference():
            v_base = pol.generate(gconvs[:8], 64, temperature=0.0)

        def agree(xs, ys, k):
            return round(sum(x["ids"][:k] == y["ids"][:k] for x, y in zip(xs, ys, strict=True))
                         / len(xs), 3)

        def prefix(xs, ys):
            out = []
            for x, y in zip(xs, ys, strict=True):
                n = 0
                while n < min(len(x["ids"]), len(y["ids"])) and x["ids"][n] == y["ids"][n]:
                    n += 1
                out.append(n)
            return round(sum(out) / len(out), 1)

        lp = work / "vllm.log"
        log = lp.read_text(errors="replace") if lp.exists() else ""
        gen.wake()
        m_awake = mem()
        gen.sleep()
        report("vllm", startup_sec=round(startup), sec_generate_with_sync=round(t_v, 1),
               policy_first16_agree=agree(v_pol, hf_pol, 16),
               policy_mean_common_prefix=prefix(v_pol, hf_pol),
               base_first16_agree=agree(v_base, hf_base, 16),
               base_mean_common_prefix=prefix(v_base, hf_base),
               not_loaded_warnings=log.count("not loaded") + log.count("Failed to load"),
               arch_line=next((ln.strip()[-160:] for ln in log.splitlines()
                               if "architecture" in ln.lower()), ""),
               language_model_only="language_model_only=True" in log
               or "language-model-only" in log,
               mem_training_side_while_vllm_sleeps=m_sleep, mem_while_vllm_awake=m_awake)
        gen.close()
    else:
        report("vllm", skipped=True)
    report("total", sec=round(time.time() - t0, 1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "smoke.json").write_text(json.dumps(RES, indent=1, default=str))


if __name__ == "__main__":
    main()
