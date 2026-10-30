"""TICA command line (DESIGN-TICA.md "Code architecture").

  python scripts/tica.py train   --config NAME [--seed N]  # Level 1 supervised training (one round)
  python scripts/tica.py eval    --config NAME [--adapter DIR] [--out DIR] [--bank NAME]
  python scripts/tica.py rounds  --config NAME [--stage train,collect,fit,eval] [--round R]
  python scripts/tica.py rl      --config NAME          # Level 1b / Extension 1 (GRPO)
  python scripts/tica.py show    --config NAME          # print the resolved config
  python scripts/tica.py windows --config NAME [--stage train,eval] [--seed N] [--adapter DIR]
                                 [--base]  # L3 fast track S1/S2 (tica/windows.py)

Run model-loading commands in the transformers-v5 env (Qwen3.5 / Gemma-4):
  PYTHONPATH=src ~/chameleons-qwen35/.venv/bin/python scripts/tica.py ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch


def build(cfg):
    from next_chameleons.tica.data import Level1Data
    from next_chameleons.tica.monitors import ConceptProbeBank, MonitorPool
    from next_chameleons.tica.policy import Policy

    pool = MonitorPool()  # primary bank first: training concepts and selectivity readouts use it
    held_out = {int(x) for x in cfg.probes.held_out_layers}
    # Load the training (pool) layers AND the held-out layers (eval-only, e.g. L20 for the
    # generalisation metric) -- pool_layers() does not include held_out_layers on its own.
    # probes.dir "" (answer format only; 10-24 Gemma-4 port): no Level-1 repeat-format bank. The
    # answer format refits its base probes on the model itself, so the bank was only ever loaded.
    layers = []
    if cfg.probes.dir or cfg.episodes.format != "answer":
        layers = list(cfg.pool_layers())
        layers += [h for h in held_out if h not in layers]
    for layer in layers:
        bank = ConceptProbeBank.from_level1_npz(Path(cfg.probes.dir) / f"bank_L{layer}.npz")
        if cfg.probes.pair_gate and layer == cfg.primary_layer():
            keep = set(json.loads(Path(cfg.probes.pair_gate).read_text()))
            mask = torch.tensor([c in keep for c in bank.concepts])
            bank.gate = mask if bank.gate is None else bank.gate & mask
        pool.add(bank, round=0, family="concept_linear", held_out=layer in held_out)

    def factory(adapter: str | None, trainable: bool):
        lora = dataclasses.asdict(cfg.lora) if (trainable and adapter is None) else None
        pol = Policy.from_pretrained(
            cfg.model.path,
            dtype=cfg.model.dtype,
            lora=lora,
            adapter=adapter,
            trainable=trainable,
            enable_thinking=cfg.model.enable_thinking,
            batch_size=cfg.model.batch_size,
            grad_ckpt=cfg.model.grad_ckpt,
            fla_grad=cfg.model.fla_grad,
            keep_think_history=cfg.model.keep_think_history,
        )
        pol.lm_chunk = cfg.model.lm_chunk  # 0 = the full-logits path
        pol.sort_reads = cfg.model.sort_reads  # length-sorted fork read chunks (10-27)
        n = len(pol.layers_module)
        if cfg.model.depth and n != int(cfg.model.depth):
            raise SystemExit(f"model.depth {cfg.model.depth} but {cfg.model.path} has {n} decoder "
                             "layers (rescale the config's layers)")
        if not cfg.model.depth and n != 32:
            print(f"[warn] {n}-layer model with the 32-layer probe-zoo tables (set model.depth)",
                  flush=True)
        return pol

    return Level1Data(cfg.data), pool, factory


def run_windows(args) -> None:
    """L3 fast track: train and/or evaluate against window monitors (tica/windows.py)."""

    from next_chameleons.tica.config import load_window_config
    from next_chameleons.tica.data import Level1Data
    from next_chameleons.tica.monitors import ConceptProbeBank
    from next_chameleons.tica.policy import Policy
    from next_chameleons.tica.windows import WindowProbe, WindowTrainer, evaluate

    cfg = load_window_config(args.config)
    if args.concept:
        cfg.env.concept = args.concept
    if args.model_path:  # e.g. a node-local copy of the snapshot (faster loading)
        cfg.model.path = args.model_path
    if args.steps is not None:
        cfg.train.steps = args.steps
    if args.grad_ckpt is not None:
        cfg.model.grad_ckpt = bool(args.grad_ckpt)
    if args.fla_grad is not None:
        cfg.model.fla_grad = bool(args.fla_grad)
    if args.seed is not None:
        cfg.train.seed = args.seed
        cfg.out = str(Path(cfg.out) / f"seed{args.seed}")
    out = Path(args.out or cfg.out)
    data = Level1Data(cfg.data)
    concept = cfg.env.concept
    split = {c.concept: c for c in data.concepts("fit", ("train", "heldout"))}
    if concept not in split:
        raise SystemExit(f"unknown concept {concept!r}")
    stages = args.stage.split(",")
    layers = sorted({cfg.env.layer, *cfg.eval.layers})

    def probes(device):
        out_ = {}
        for L in layers:
            bank = ConceptProbeBank.from_level1_npz(Path(cfg.probes) / f"bank_L{L}.npz")
            out_[L] = WindowProbe(bank.to(device), concept)
        return out_

    def load(adapter, trainable):
        lora = dataclasses.asdict(cfg.lora) if (trainable and adapter is None) else None
        return Policy.from_pretrained(
            cfg.model.path,
            dtype=cfg.model.dtype,
            lora=lora,
            adapter=adapter,
            trainable=trainable,
            grad_ckpt=cfg.model.grad_ckpt,
            fla_grad=cfg.model.fla_grad,
            keep_think_history=cfg.model.keep_think_history,
            enable_thinking=cfg.model.enable_thinking,
            batch_size=cfg.model.batch_size,
        )

    adapter = args.adapter
    if "train" in stages:
        if (out / "train" / "final").exists():
            print("train: final adapter exists, skipping", flush=True)
        else:
            pol = load(adapter, True)
            texts = split[concept].texts
            WindowTrainer(
                pol, probes(pol.device)[cfg.env.layer], texts, cfg.env, cfg.train, out / "train"
            ).train()
            del pol
            torch.cuda.empty_cache()
        adapter = str(out / "train" / "final")
    if "eval" in stages:
        base = args.base or adapter is None
        pol = load(None if base else adapter, False)
        texts = []
        for sp in cfg.eval.text_splits:
            texts += next(
                c.texts for c in data.concepts(sp, ("train", "heldout")) if c.concept == concept
            )
        evaluate(
            pol,
            probes(pol.device),
            texts,
            cfg.env,
            cfg.eval,
            out / ("eval_base" if base else "eval"),
            has_adapter=not base,
        )


def cap_free_requests(data, inputs: dict, concepts, acfg=None) -> dict:
    """adaptive.cap_free_*: {checkable train concept: [(request, fallback answer, reference)]} from
    the calib and episode splits (training rollouts force fit-split texts: these are new). SQL items
    become execution-checkable tasks (tica/validity.py): the request carries the reference's tables
    + rows, the fallback answer is the reference query; items whose reference does not run in
    sqlite are dropped. Other concepts: the published answer is both fallback and reference."""

    from next_chameleons.tica.answer_format import text_id
    from next_chameleons.tica.validity import check_task, checkable, sql_split, sql_task_request

    pool = {}
    for c in sorted(x for x in concepts if checkable(x)):
        rows = []
        for sp in ("calib", "episode"):
            for ct in data.concepts(sp, ("train",), only={c}):
                for x in ct.texts:
                    q = inputs.get(text_id(x))
                    if not q:
                        continue
                    if c == "SQL code":
                        q2 = sql_task_request(q, x)
                        if q2 is not None:
                            rows.append((q2, sql_split(x)[1], x))
                    elif check_task(c, x, q, x):  # 10-27: a reference failing its own check
                        rows.append((q, x, x))    # (English "German" texts) is no NLL target
        if rows:
            pool[c] = rows
    # 10-30: functional JavaScript tasks (APPS, Node-tested) for the JavaScript concept
    if acfg is not None and acfg.cap_free_js_tasks and "JavaScript code" in set(concepts):
        from next_chameleons.tica import js_tasks

        apps = js_tasks.pool_rows(acfg.cap_free_js_tasks)
        pool["JavaScript code"] = apps + (pool.get("JavaScript code", [])
                                          if acfg.cap_free_js_keep_form else [])
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command", choices=["train", "eval", "rounds", "rl", "adaptive", "show", "windows"]
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--panel-precompute", metavar="I/N", default="",
                    help="adaptive: fill adaptive.first_probe_panel_cache (shard I of N) and "
                         "exit instead of training (TamIA 10-21; "
                         "scripts/slurm/tamia_panel_precompute.sh)")
    ap.add_argument("--out")
    ap.add_argument("--stage", default="train,collect,fit,eval")
    ap.add_argument("--round", type=int)
    ap.add_argument(
        "--seed", type=int, help="train/rl: override the seed; output goes to out/seed{N}"
    )
    ap.add_argument(
        "--bank", help="eval: evaluate with this pool monitor only (default: every bank)"
    )
    ap.add_argument("--base", action="store_true", help="windows eval: the base model")
    ap.add_argument("--concept", help="windows: override env.concept")
    ap.add_argument("--model-path", help="windows: override model.path")
    ap.add_argument("--steps", type=int,
                    help="windows / adaptive: override train.steps (resume: REMAINING steps)")
    ap.add_argument("--grad-ckpt", type=int, choices=[0, 1], help="windows: override")
    ap.add_argument("--fla-grad", type=int, choices=[0, 1], help="windows: override")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config field by dotted path, e.g. train.micro_batch=2 "
                         "(repeatable)")
    args = ap.parse_args()
    if args.command == "windows":
        run_windows(args)
        return

    from next_chameleons.tica import dp
    from next_chameleons.tica.config import load_run_config

    # data-parallel episodes (TICA_DP=N in scripts/slurm/tica_l3covert.sh: WORLD_SIZE / RANK set,
    # one visible GPU per process); adaptive only
    data_parallel = dp.init_from_env()
    # TICA_MEM_FRACTION (10-28): cap this process's CUDA caching allocator at that fraction of its
    # GPU, so a run SHARING GPUs with another (srun --overlap) runs out of memory itself instead
    # of pushing the other run over the limit
    if os.environ.get("TICA_MEM_FRACTION") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ["TICA_MEM_FRACTION"]), 0)
    if data_parallel and args.command != "adaptive":
        raise SystemExit("data-parallel episodes: adaptive only")
    cfg = load_run_config(args.config)
    if args.model_path:  # e.g. a node-local copy of the snapshot (faster loading)
        cfg.model.path = args.model_path
    if args.steps is not None:  # resume chunks (--adapter step_N --out DIR): REMAINING steps
        cfg.train.steps = args.steps
    if args.set:
        from next_chameleons.tica.config import apply_overrides

        apply_overrides(cfg, args.set)
    if args.seed is not None:
        cfg.train.seed = cfg.rl.seed = args.seed
        cfg.out = str(Path(cfg.out) / f"seed{args.seed}")
    if args.command == "show":
        print(json.dumps(dataclasses.asdict(cfg), indent=1, default=str))
        return
    data, pool, factory = build(cfg)
    out = Path(args.out or cfg.out)

    if args.command == "train":
        from next_chameleons.tica.episodes import EpisodeSampler
        from next_chameleons.tica.trainers.supervised import SupervisedTrainer

        pol = factory(args.adapter, True)
        pool = pool.to(pol.device)
        concepts = data.concepts("fit", ("train",), only=set(pool.entries[0].monitor.gated()))
        sampler = EpisodeSampler(
            concepts,
            data.negatives("fit"),
            pool,
            mix=cfg.train_mix,
            turns=cfg.train.turns,
            seed=cfg.train.seed,
            **cfg.sampler_kwargs(),
        )
        SupervisedTrainer(pol, pool, sampler, cfg.train, out / "train").train()
    elif args.command == "eval":
        from next_chameleons.tica.evaluation import evaluate

        pol = factory(args.adapter, False)
        pool = pool.to(pol.device)
        names = [args.bank] if args.bank else [e.monitor.name for e in pool.entries]
        for name in names:  # feedback from each bank in turn; the primary bank first
            evaluate(pol, pool, data, name, cfg.eval_config(), out / "eval" / name)
    elif args.command == "adaptive":
        from next_chameleons.tica.answer_format import load_inputs
        from next_chameleons.tica.episodes import EpisodeSampler
        from next_chameleons.tica.monitors import MonitorPool
        from next_chameleons.tica.trainers.supervised import (
            AdaptiveSupervisedConfig,
            AdaptiveSupervisedTrainer,
            SupervisedConfig,
        )

        pol = factory(args.adapter, True)
        from next_chameleons.tica.vllm_gen import maybe_attach

        if data_parallel and os.environ.get("TICA_VLLM") == "1":
            raise SystemExit("data-parallel episodes: no colocated vLLM")
        maybe_attach(pol, cfg.model.path, out / "vllm")  # TICA_VLLM=1: colocated vLLM generation
        prompts = None
        if cfg.episodes.format == "answer":
            # Answer format (REPLICATION 10-13a): the repeat-format bank does not apply. Fit the
            # base + held-out probes on the BASE model in the answer format (same function the
            # frozen race uses) and start from that pool.
            from next_chameleons.tica.answer_format import fit_answer_probes
            from next_chameleons.tica.answer_format import prompts as fmt_prompts

            if cfg.start_pool or not cfg.episode_concepts:
                raise SystemExit("answer format: episode concepts, no start_pool")
            prompts = fmt_prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
            held = [int(x) for x in cfg.probes.held_out_layers]
            layers = [cfg.primary_layer()] + [h for h in held if h != cfg.primary_layer()]
            # depth panel (adaptive.panel_layers, 10-19n): extra base probes, family "panel"
            panel = [int(x) for x in cfg.rl.adaptive.panel_layers if int(x) not in layers]
            layers += panel
            multi = len(cfg.episode_concepts) > 1
            pool = MonitorPool()
            # Multi-concept (topic base, REPLICATION 10-14p): one base probe set per TRAIN concept,
            # fit against other TRAIN concepts only (no dev / held-out concept shapes a trained
            # probe), named per concept. Data-parallel: rank 0 fits and saves, every rank loads
            # the saved pool (identical base probes everywhere).
            if dp.is_main():
                for k, concept in enumerate(cfg.episode_concepts):
                    mons, quality = fit_answer_probes(
                        cfg.rl.adaptive, data, pol, concept, layers, prompts[0],
                        neg_splits=("train",) if multi else None,
                        rename=f"answer_base_c{k}" if multi else None)
                    print(json.dumps({"answer_base_probes": quality, "concept": concept}),
                          flush=True)
                    for L, m in zip(layers, mons, strict=True):
                        pool.add(m, round=0, family="panel" if L in panel else "concept_linear",
                                 held_out=L in held)
                pool.save(out / "answer_pool")
            if data_parallel:
                dp.barrier()
                pool = MonitorPool.load(out / "answer_pool")
        elif cfg.start_pool:  # start from a saved bank (base probe + branch refits)
            default_pool = pool  # built from probes dir; carries the held-out (eval-only) monitors
            pool = MonitorPool.load(cfg.start_pool)
            pool.entries = [e for e in pool.entries if e.monitor.name not in cfg.start_pool_exclude]
            for e in default_pool.entries:  # keep held-out probes (e.g. L20)
                if e.held_out and not any(x.monitor.name == e.monitor.name for x in pool.entries):
                    pool.add(e.monitor, round=e.round, family=e.family, held_out=True)
        pool = pool.to(pol.device)
        only = (
            set(cfg.episode_concepts)
            if cfg.episode_concepts
            else set(pool.entries[0].monitor.gated())
        )
        concepts = data.concepts("fit", ("train",), only=only)
        sampler = EpisodeSampler(
            concepts,
            data.negatives("fit"),
            pool,
            mix=cfg.train_mix,
            turns=cfg.train.turns,
            # resume chunks draw fresh episodes (10-18): offset the seed by the chunk's global start
            seed=cfg.train.seed + int(os.environ.get("TICA_STEP_OFFSET", "0") or 0),
            **cfg.sampler_kwargs(),
        )
        asc = AdaptiveSupervisedConfig(
            **{f.name: getattr(cfg.train, f.name) for f in dataclasses.fields(SupervisedConfig)},
            adaptive=cfg.rl.adaptive,
        )
        # Branch-refit (adaptive.refit_mode="branch"): the FULL covered concept set supplies the
        # other-concept negatives; roots_only = the episode concept(s) we train to hide.
        branch_concepts = data.concepts("fit", ("train",))
        inputs = load_inputs(cfg.episodes.inputs) if cfg.episodes.format == "answer" else None
        cap_free_pool = (cap_free_requests(data, inputs, only, cfg.rl.adaptive)
                         if inputs and (cfg.rl.adaptive.cap_free_episodes
                                        or cfg.rl.adaptive.cap_free_nobrief) else None)
        if cap_free_pool:
            print("cap_free pool:", {c: len(v) for c, v in cap_free_pool.items()}, flush=True)
        tr = AdaptiveSupervisedTrainer(
            pol, pool, sampler, asc, out / "adaptive",
            branch_concepts=branch_concepts, branch_negatives=data.negatives("fit"),
            roots_only=only, prompts=prompts,
            inputs=inputs, cap_free_pool=cap_free_pool,
            # resume chunks restore optimizer / curriculum / RNG state (train.save_state)
            resume_state=(str(Path(args.adapter) / "trainer_state.pt")
                          if args.adapter and cfg.train.save_state else None),
        )
        if args.panel_precompute:
            i, n = (int(x) for x in args.panel_precompute.split("/"))
            tr.precompute_panel(i, n)
        else:
            tr.train()
        if data_parallel:
            import torch.distributed as dist

            dist.destroy_process_group()
    elif args.command == "rounds":
        from next_chameleons.tica.rounds import ArmsRace

        race = ArmsRace(cfg.rounds_config(), out, data, pool, factory)
        race.run(tuple(args.stage.split(",")), only_round=args.round)
    elif args.command == "rl":
        from next_chameleons.tica.episodes import EpisodeSampler
        from next_chameleons.tica.monitors import MonitorPool
        from next_chameleons.tica.trainers.rl import RLTrainer

        pol = factory(args.adapter, True)
        if cfg.start_pool:  # start from a saved bank (e.g. an l3loop pool with branch refits)
            default_pool = pool  # built from probes dir; carries the held-out (eval-only) monitors
            pool = MonitorPool.load(cfg.start_pool)
            pool.entries = [e for e in pool.entries if e.monitor.name not in cfg.start_pool_exclude]
            for e in default_pool.entries:  # keep held-out probes (e.g. L20)
                if e.held_out and not any(x.monitor.name == e.monitor.name for x in pool.entries):
                    pool.add(e.monitor, round=e.round, family=e.family, held_out=True)
        pool = pool.to(pol.device)
        only = (
            set(cfg.episode_concepts)
            if cfg.episode_concepts
            else set(pool.entries[0].monitor.gated())
        )
        concepts = data.concepts("fit", ("train",), only=only)
        # Adaptive runs need selectivity + refit-negative conditions, not only real_score.
        mix = cfg.train_mix if cfg.rl.use_adaptive else {"real_score": 1.0}
        sampler = EpisodeSampler(
            concepts,
            data.negatives("fit"),
            pool,
            mix=mix,
            turns=cfg.rl.turns,
            seed=cfg.rl.seed,
            **cfg.sampler_kwargs(),
        )
        if cfg.rl.mode == "open":
            raise SystemExit("open mode needs the judge wiring (Extension 1); not configured yet")
        if cfg.rl.mode == "reason_forced" and not cfg.model.enable_thinking:
            # With thinking off the template pre-fills an empty reasoning block, so there is no
            # sampled reasoning to reinforce.
            raise SystemExit("rl reason_forced needs model.enable_thinking: true")
        RLTrainer(pol, pool, sampler.sample, cfg.rl, out / "rl").train()


if __name__ == "__main__":
    main()
