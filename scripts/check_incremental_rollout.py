"""GPU check (REPLICATION 2026-10-13c): Policy.pooled_incremental vs Policy.pooled on real
12-turn answer-format episodes (Qwen3.5-4B, keep-think history, verdict feedback). Prints the
per-turn min cosine / max relative error at the probe layers and each reader's wall time over the
whole episode. Aggregates only.

  python scripts/check_incremental_rollout.py --config l4dodge_math_4b --model-path DIR
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.answer_format import prompts
from next_chameleons.tica.config import load_run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--turns", type=int, default=12)
    a = ap.parse_args()
    cfg = load_run_config(a.config)
    if a.model_path:
        cfg.model.path = a.model_path
    data, _, factory = build(cfg)
    pol = factory(None, False)
    opening, fb = prompts(cfg.episodes.format, cfg.episodes.inputs, cfg.episodes.layout)
    concept = cfg.episode_concepts[0]
    texts = data.concepts("fit", ("train", "heldout"), only={concept})[0].texts
    E, T, layers = a.episodes, a.turns, [12, 20, 24]
    convs = [[{"role": "user", "content": opening("real_score", texts[(i * T) % len(texts)],
                                                  concept)}] for i in range(E)]
    state, t_plain, t_inc = {}, 0.0, 0.0
    worst = (1.0, 0.0)
    for t in range(T):
        for i in range(E):
            convs[i].append({"role": "assistant", "content": texts[(i * T + t) % len(texts)]})
        torch.cuda.synchronize()
        t0 = time.time()
        a_ = pol.pooled(convs, layers)
        torch.cuda.synchronize()
        t1 = time.time()
        b_ = pol.pooled_incremental(convs, layers, state)
        torch.cuda.synchronize()
        t_plain += t1 - t0
        t_inc += time.time() - t1
        cs, re = [], []
        for L in layers:
            cs.append(float(torch.nn.functional.cosine_similarity(a_[L], b_[L], dim=-1).min()))
            re.append(float(((a_[L] - b_[L]).norm(dim=-1) / a_[L].norm(dim=-1)).max()))
        worst = (min(worst[0], *cs), max(worst[1], *re))
        print(f"turn {t}: cos min {min(cs):.6f}  rel-err max {max(re):.5f}  "
              f"tokens ~{len(pol.view(convs[0]).ids)}", flush=True)
        for i in range(E):
            nxt = texts[(i * T + t + 1) % len(texts)]
            convs[i].append({"role": "user", "content": fb("real_score", concept, (40.0, 50.0), nxt,
                                                           feedback="verdict")})
    print(f"ALL: cos min {worst[0]:.6f}  rel-err max {worst[1]:.5f}  cache hits {state.get('hits')}"
          f"  plain {t_plain:.1f}s  incremental {t_inc:.1f}s  speedup {t_plain / t_inc:.1f}x")


if __name__ == "__main__":
    main()
