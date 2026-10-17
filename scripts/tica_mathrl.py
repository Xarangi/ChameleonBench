"""Math solve-while-hiding pilot (REPLICATION 2026-10-14o; MATH-PILOT.md).

  python scripts/tica_mathrl.py train --config mathrl_4b [--resume DIR] [--steps N]
  python scripts/tica_mathrl.py eval  --config mathrl_4b [--adapter DIR | --base] --out OUT.json
                                      [--episodes 8] [--turns 12]

train: builds (once, cached) the base-model generations and the start probes, then runs the RL
loop (tica/trainers/mathrl.py). eval: greedy episodes on GSM8K TEST problems with the same refit
race; per-turn accuracy / caught / held-out detection; --base = the base model (reference).
Run in the transformers-v5 env (Qwen3.5): PYTHONPATH=src ~/chameleons-qwen35/.venv/bin/python ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from next_chameleons.config import find_config_path, load_yaml
from next_chameleons.tica.trainers.mathrl import (
    MathRL,
    MathRLConfig,
    build_base_gens,
    load_questions,
    summarize,
)


def load(name: str) -> dict:
    p = Path(name)
    path = p if p.suffix in (".yaml", ".yml") and p.exists() else find_config_path("tica", name)
    d = load_yaml(path)
    unknown = set(d) - {"name", "model", "lora", "out", "mathrl"}
    if unknown:
        raise KeyError(f"unknown keys {sorted(unknown)}")
    names = {f.name for f in dataclasses.fields(MathRLConfig)}
    bad = set(d.get("mathrl") or {}) - names
    if bad:
        raise KeyError(f"mathrl: unknown keys {sorted(bad)}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "eval"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--resume", help="train: a step_* dir (adapter + trainer_state.pt)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--adapter")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--out")
    a = ap.parse_args()

    from next_chameleons.tica.policy import Policy

    d = load(a.config)
    cfg = MathRLConfig(**(d.get("mathrl") or {}))
    if a.steps:
        cfg.steps = a.steps
    m = d["model"]
    out = Path(d["out"])
    kw = {k: m[k] for k in ("enable_thinking", "batch_size", "keep_think_history") if k in m}
    if a.command == "train":
        lora = None if a.resume else d.get("lora", {"rank": 16, "alpha": 32})
        pol = Policy.from_pretrained(a.model_path or m["path"], dtype=m.get("dtype", "bfloat16"),
                                     lora=lora, adapter=a.resume, trainable=True,
                                     grad_ckpt=m.get("grad_ckpt", True),
                                     fla_grad=m.get("fla_grad", False), **kw)
        from next_chameleons.tica.vllm_gen import maybe_attach

        maybe_attach(pol, a.model_path or m["path"], out / "vllm")  # TICA_VLLM=1
        gens = build_base_gens(pol, cfg, out)
        tr = MathRL(pol, cfg, out, gens, train=True)
        print(json.dumps({"start_probes": tr.fit_start_probes(), "gens": gens["stats"],
                          "fork_pos": len(tr.fork_pos), "np_pool": len(tr.np_pool)}), flush=True)
        if a.resume:
            tr.load_state(Path(a.resume))
        tr.train()
        return
    pol = Policy.from_pretrained(a.model_path or m["path"], dtype=m.get("dtype", "bfloat16"),
                                 adapter=None if a.base else a.adapter, trainable=False,
                                 fla_grad=m.get("fla_grad", False), **kw)
    from next_chameleons.tica.vllm_gen import maybe_attach

    maybe_attach(pol, a.model_path or m["path"], Path(a.out).parent / "vllm")  # TICA_VLLM=1
    gens = build_base_gens(pol, cfg, out)
    ev = MathRL(pol, cfg, out, gens, train=False)
    q = ev.fit_start_probes()
    _, test = load_questions()
    rng = random.Random(1)
    E, T = a.episodes, a.turns
    pool = rng.sample(test, min(len(test), E * T))
    qs = [pool[i * T : (i + 1) * T] for i in range(E)]
    import contextlib

    ctx = pol.reference() if a.base else contextlib.nullcontext()
    with ctx:
        turns, logs = ev.rollout(qs, [True] * E, 1, greedy=True)
        _, logs_nb = ev.rollout(qs, [False] * E, 1, greedy=True, refit=False)
    rep = {"adapter": a.adapter, "base": a.base, "episodes": E, "turns": T, "start_probes": q,
           **summarize(turns, logs + logs_nb), "logs": logs, "logs_nobrief": logs_nb}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("logs")}), flush=True)


if __name__ == "__main__":
    main()
