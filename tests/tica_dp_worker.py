"""Worker for tests/test_tica_dp.py (data-parallel episodes, REPLICATION 2026-10-19b): one tiny
adaptive run with every rollout-coupled feature on (episode refits, shared turn-0 fork probes,
concept switches, monitor-off turns, Thompson kind / concept curricula, pursuit, KL anchor,
cap_gen with hide, retain anchor), run for a few steps; each rank saves its end state."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def build(out: Path, steps: int = 2, budget: bool = False, hclip: float = 0.0,
          desync: bool = False, **over):
    from next_chameleons.tica.adaptive import AdaptiveConfig
    from next_chameleons.tica.answer_format import text_id
    from next_chameleons.tica.episodes import ConceptTexts, EpisodeSpec
    from next_chameleons.tica.monitors import MonitorPool
    from next_chameleons.tica.objectives import ObjectiveConfig
    from next_chameleons.tica.trainers import mathrl
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
    )
    from test_tica_framework import bank as make_bank
    from test_tica_framework import tiny_policy

    pol = tiny_policy()
    pol.keep_think_history = True
    with torch.no_grad():  # LoRA B starts at zero (policy == base): make them differ
        g = torch.Generator().manual_seed(3)
        for n, p in pol.model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
        if desync and int(os.environ.get("RANK", "0")) > 0:  # unseeded LoRA init: differs per rank
            for n, p in pol.model.named_parameters():
                if "lora_A" in n:
                    p.add_(0.1)
    pool = MonitorPool()
    pool.add(make_bank(1, ["cats", "dogs"]), round=0, family="concept_linear")
    base = pool.entries[0].monitor.name
    cats = [f"meow {k} purr" for k in range(10)]
    dogs = [f"woof {k} bark" for k in range(10)]
    free = ["a b c", "d e f", "g h i", "j k l"]
    inputs = {text_id(x): f"req {x[:6]}" for x in cats + dogs + free}

    def opening(cond, text, concept, named=True):
        req = inputs[text_id(text)]
        return req if cond == "nobrief" else f"monitor on {concept}\n\n{req}"

    def render(cond, concept, shown, nxt, *, feedback="score", named=True, notice=None):
        return f"verdict on {concept} {notice or ''}\n\n{inputs[text_id(nxt)]}"

    class Sampler:
        """4 episodes a step: cats, a cats -> dogs switch, dogs, a no-brief control."""

        def __init__(self):
            self.rng = random.Random(11)
            self.concepts = [ConceptTexts("cats", "train", cats),
                             ConceptTexts("dogs", "train", dogs)]
            self.concept_weights = None

        def sample(self, n):
            T = 5
            out = []
            for k in range(n):
                kind = k % 4
                c = "dogs" if kind == 2 else "cats"
                txt = self.rng.sample(cats if c == "cats" else dogs, T)
                meta = {}
                if kind == 1:
                    txt = self.rng.sample(cats, 2) + self.rng.sample(dogs, T - 2)
                    meta = {"monitored_turns": ["cats", "cats", "dogs", "dogs", "dogs"]}
                cond = "nobrief" if kind == 3 else "real_score"
                bud = None
                if budget and cond == "real_score":  # turn budget (10-19b): R 1 / M 2 / S 2
                    seq = ["cats", "dogs"] if kind == 1 else [c]
                    pools = {x: self.rng.sample(cats if x == "cats" else dogs, 2 * T) for x in seq}
                    txt, meta = list(pools[seq[0]][:T]), {"monitored_turns": [seq[0]] * T}
                    bud = {"R": 1, "M": 2, "S": 2, "seq": seq, "k": 0, "pt": 0, "done": False,
                           "pools": pools, "used": {x: (T if x == seq[0] else 0) for x in seq}}
                out.append(EpisodeSpec(f"{cond}|{c}|{k}", cond, txt, c, c, base, meta=meta,
                                       budget=bud))
            return out

    q = [{"question": "1+1?", "answer": "2"}, {"question": "2+2?", "answer": "4"}]
    mathrl.load_questions = lambda: (q * 10, [])

    def fake_generate(convs, max_new, **kw):
        r = random.Random(kw.get("seed", 0))
        out = []
        for c in convs:
            gold = "2" if "1+1" in c[-1]["content"] else "4"
            ok = r.random() < 0.5
            text = (f"x = {gold}. The answer is {gold}." if ok else "The answer is 7.")
            out.append({"text": text, "finished": True, "ids": [1], "prompt_ids": [1],
                        "stop": None})
        return out

    pol.generate = fake_generate
    anchor = out.parent / "anchor.json"
    if not anchor.exists():
        anchor.parent.mkdir(parents=True, exist_ok=True)
        anchor.write_text(json.dumps([{"prompt": f"p{k}", "answer": f"ans {k} ok"}
                                      for k in range(6)]))
    ad = dict(
        refit_mode="episode", refit_every=0, refit_min_auroc=0.0, refit_min_tpr_np=0.0,
        refit_C=1.0, first_probe="fork", episode_refit_kind="mixed",
        episode_refit_kinds=["linear@1", "linear@2"], episode_kind_sampling="thompson",
        concept_sampling="thompson", episode_fork={"pos": 6, "np": 6, "other": 6, "free": 3},
        single_pass=True, incremental_rollout=True, refit_np_cache=True, reloc_every=0,
        adaptive=False, episode_cumulative=True, off_turn_prob=0.2, refit_switch_max=1,
        pursuit=True, motion="conditional", stay_offset=True, pursuit_from_turn=1,
        kl_weight=1.0, kl_ref="nobrief", kl_topk=5,
        cap_gen_weight=1.0, cap_gen_episodes=2, cap_gen_samples=2, cap_gen_skip=0,
        cap_gen_hide=True, anchor_weight=0.5, anchor_n=2, anchor_cache=str(anchor),
        episode_rng=True)
    ad.update(over)
    cfg = AdaptiveSupervisedConfig(
        steps=steps, episodes_per_step=4, turns=5, micro_batch=2, lr=1e-2, warmup_steps=1,
        save_every=0, hide_grad_clip=hclip, dp_sync_init=desync,
        objective=ObjectiveConfig(pin_first_turn=False, pin_segment_first=False,
                                  hide_from_turn=0, retry_hide_ladder=[2.0, 3.0]),
        adaptive=AdaptiveConfig(**ad))
    tr = AdaptiveSupervisedTrainer(
        pol, pool, Sampler(), cfg, out, prompts=(opening, render), inputs=inputs,
        branch_concepts=[ConceptTexts("cats", "train", cats), ConceptTexts("dogs", "train", dogs)],
        branch_negatives=free)
    return tr, pol


def state(tr, pol, recs) -> dict:
    bandit = tr.kind_bandit
    kb = {k: (b.B, b.N) for k, b in (("arch", getattr(bandit, "arch", bandit)),
                                      ("layer", getattr(bandit, "layer", None))) if b is not None}
    cb = tr.concept_bandit
    return {
        "recs": recs,
        "lora": {n: p.detach().clone() for n, p in pol.model.named_parameters() if "lora_" in n},
        "rng": tr.rng.getstate(),
        "kind_bandit": kb,
        "concept_bandit": ({f: (b.B, b.N) for f, b in cb.within.items()}, (cb.fam.B, cb.fam.N)),
        "concept_weights": tr.sampler.concept_weights,
        "pursuit": {c: (p.centroid.clone(), p.scale) for c, p in tr.pursuit_by.items()},
        "kind_stats": tr.kind_stats,
        "deploy_log": tr.deploy_log,
        "seg_log": tr.seg_log,
    }


def run(rank: int, world: int, out: str, port: int, steps: int = 2,
        budget: bool = False, panel: bool = False, shard: bool = False,
        hclip: float = 0.0, desync: bool = False) -> None:
    """mp.spawn entry: rank `rank` of `world` (gloo); saves <out>/rank{rank}.pt."""

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world),
                      RANK=str(rank))
    torch.set_num_threads(1)
    if world > 1:
        import torch.distributed as dist

        dist.init_process_group("gloo", rank=rank, world_size=world)
    d = Path(out)
    over = {}
    if panel:  # turn-0 base panel (10-20d) from global step 1, inside merged reads
        over = dict(first_probe="switch_fork", first_probe_panel_from_step=1,
                    first_probe_panel_kinds=["linear@1", "linear@2"], merged_reads=True)
    if shard:  # data-parallel cap_gen (10-20)
        over.update(cap_gen_shard=True, cap_gen_log=True)
    tr, pol = build(d / "run", steps=steps, budget=budget, hclip=hclip, desync=desync, **over)
    recs = [tr.step() for _ in range(steps)]
    torch.save(state(tr, pol, recs), d / f"rank{rank}.pt")
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
