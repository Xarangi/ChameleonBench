"""Level 1 trainer: teacher-forced multi-turn episodes, loss through frozen monitors.

One step:
1. sample episode specs (EpisodeSampler);
2. ROLLOUT with the current policy (eval mode, no grad), so every feedback message is the one the
   current model would have received;
3. plan the per-(turn, readout) roles (objectives.plan_targets);
4. compute REFERENCE probe values for the MATCH targets with the adapters disabled (base model, no
   grad);
5. forward every needed per-turn VIEW with grad, in chunks of `micro_batch` views, accumulating
   gradients. Each chunk contributes its share of the full-batch MEANS (hide over all hide
   targets, match over all match targets, NLL over all views; objectives.loss_terms with totals),
   so the summed gradient equals the full-batch gradient. (Before 10-03j a chunk's mean was
   weighted by its total target count, which mis-weighted hide vs match targets.)
Each turn is its own view (generation prompt + text), not a slice of one long sequence: the
templates render finished turns differently (tica/policy.py).

Logging: one JSON line per step in out_dir/train_log.jsonl. Rollout metrics are measured before
the update: detection at turn 0 vs later turns, per condition. Checkpoints: LoRA adapter at
out_dir/step_{k} and out_dir/final.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from next_chameleons.tica.episodes import EpisodeSampler, Trajectory, rollout
from next_chameleons.tica.monitors import MonitorPool
from next_chameleons.tica.objectives import (
    ObjectiveConfig,
    loss_terms,
    plan_targets,
    probe_prob,
    summarize_stats,
)


@dataclass
class SupervisedConfig:
    steps: int = 300
    episodes_per_step: int = 8
    turns: int = 4
    micro_batch: int = 8  # views per forward/backward chunk
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 10
    # Optional peak-then-decay LR (REPLICATION 2026-10-17c; default off): warm up to lr_peak, then
    # cosine-decay to `lr` over lr_decay_steps GLOBAL steps (TICA_STEP_OFFSET across resume
    # chunks), then hold `lr`. 0 = the constant-after-warmup schedule.
    lr_peak: float = 0.0
    lr_decay_steps: int = 0
    grad_clip: float = 1.0
    save_every: int = 100
    eval_every: int = 0  # 0 = never (run evaluation as a separate job)
    # Train-until-evasion: stop early once the mean on-policy last-turn detection of the hide
    # condition over the last stop_window steps is <= stop_det (after min_steps). steps is the cap.
    stop_det: float | None = None
    stop_window: int = 20
    min_steps: int = 20
    seed: int = 0
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)


def rollout_metrics(trajs: list[Trajectory]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for tr in trajs:
        tc = tr.spec.meta.get("turn_concepts")
        for r in tr.results:
            tag = "t0" if r["turn"] == 0 else "tN"
            if tag == "t0" or tc is None or tc[r["turn"]] == tr.spec.monitored:
                # hidden mode: later-turn monitored detection is read on the monitored turns only
                det = float(r["monitored"]["detected"])
                acc[f"det_{tr.spec.cond}_{tag}"].append(det)
                # Per-turn as well: t0/tN alone cannot distinguish a model that is caught at turn 1
                # and escapes by turn 3 (in-context iteration, what we are training for) from one
                # that applies a static dodge from turn 1 onward. Both give det_tN = 0.
                acc[f"detturn_{tr.spec.cond}_{r['turn']}"].append(det)
            for k, v in r.items():
                if k.split("@")[0] in ("text", "keep"):
                    acc[f"det_{k.split('@')[0]}_{tr.spec.cond}_{tag}"].append(float(v["detected"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


class SupervisedTrainer:
    def __init__(
        self,
        policy,
        pool: MonitorPool,
        sampler: EpisodeSampler,
        cfg: SupervisedConfig,
        out_dir: Path | str,
        evaluator=None,
    ) -> None:
        self.policy, self.pool, self.sampler, self.cfg = policy, pool, sampler, cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        params = [p for p in policy.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("policy has no trainable parameters (attach a LoRA adapter)")
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        warm = max(1, cfg.warmup_steps)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda k: min(1.0, (k + 1) / warm))
        self.params = params
        self.step_k = 0
        torch.manual_seed(cfg.seed)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))

    # ----------------------------------------------------------------------------- one step
    def step(self) -> dict:
        cfg, pol, pool = self.cfg, self.policy, self.pool
        t0 = time.time()
        specs = self.sampler.sample(cfg.episodes_per_step)
        pol.model.eval()
        trajs = rollout(pol, pool, specs, cfg.turns)
        metrics = rollout_metrics(trajs)
        targets = plan_targets(trajs, cfg.objective)
        # Views: one per (trajectory, turn) that carries a target, LENGTH-SORTED so each
        # micro-batch holds views of similar length. Policy.forward keeps logits from
        # min(start)-1 to max(len), so a batch mixing a turn-0 view (short prompt) with a turn-3
        # view (long history) materialises [B, ~full length, 248k vocab]. Unsorted, step time was
        # 575 s for 64 views (10-10); the same work is ~10x cheaper sorted.
        keys = sorted({(tg.traj, tg.turn) for tg in targets})
        built = [pol.view(trajs[i].convs[t]) for i, t in keys]
        order = sorted(range(len(keys)), key=lambda k: len(built[k].ids))
        views = [built[k] for k in order]
        view_of = {keys[k]: i for i, k in enumerate(order)}
        rows = [view_of[(tg.traj, tg.turn)] for tg in targets]
        layers = pool.layers

        # Reference (base model) probe values for every target; only MATCH ones are used.
        ref = torch.empty(len(targets), device=pol.device)
        with pol.reference():
            for a in range(0, len(views), cfg.micro_batch):
                chunk = list(range(a, min(a + cfg.micro_batch, len(views))))
                pooled, _ = pol.forward([views[v] for v in chunk], layers, grad=False)
                ks = [k for k, r in enumerate(rows) if r in chunk]
                if ks:
                    local = [chunk.index(rows[k]) for k in ks]
                    ref[ks] = probe_prob(pool, pooled, local, [targets[k] for k in ks]).detach()

        pol.model.train()
        self.opt.zero_grad(set_to_none=True)
        acc: dict[str, float] = defaultdict(float)
        totals = {
            "hide": sum(tg.role == "hide" for tg in targets),
            "match": sum(tg.role == "match" for tg in targets),
            "views": len(views),
        }
        loss_sum = 0.0
        for a in range(0, len(views), cfg.micro_batch):
            chunk = list(range(a, min(a + cfg.micro_batch, len(views))))
            ks = [k for k, r in enumerate(rows) if r in chunk]
            pooled, nll = pol.forward([views[v] for v in chunk], layers, grad=True, with_nll=True)
            local = [chunk.index(rows[k]) for k in ks]
            p = probe_prob(pool, pooled, local, [targets[k] for k in ks])
            prev = torch.tensor(
                [
                    targets[k].prev_p if targets[k].prev_p is not None else float("nan")
                    for k in ks
                ],
                device=pol.device,
            )
            loss, st = loss_terms(
                p, ref[ks], [targets[k].role for k in ks], nll, cfg.objective, totals, prev_p=prev
            )
            loss.backward()
            loss_sum += float(loss.detach())
            for k2, v in st.items():
                acc[k2] += v
        agg = {"loss": loss_sum, **summarize_stats(acc)}
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
        self.opt.step()
        pol.weights_version = getattr(pol, "weights_version", 0) + 1  # vLLM re-sync
        self.sched.step()
        self.step_k += 1
        rec = {
            "step": self.step_k,
            "lr": self.sched.get_last_lr()[0],
            "grad_norm": float(gnorm),
            "n_views": len(views),
            "n_targets": len(targets),
            "sec": round(time.time() - t0, 2),
            "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available()
            else None,
            **{f"train/{k}": v for k, v in agg.items()},
            **{f"rollout/{k}": v for k, v in metrics.items()},
        }
        with (self.out / "train_log.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def train(self) -> None:
        cfg = self.cfg
        recent: list[float] = []
        while self.step_k < cfg.steps:
            rec = self.step()
            dets = [
                v for k, v in rec.items() if k.startswith("rollout/det_real_") and k.endswith("_tN")
            ]
            if dets:
                recent = [*recent, sum(dets) / len(dets)][-cfg.stop_window :]
            if (
                cfg.stop_det is not None
                and self.step_k >= cfg.min_steps
                and len(recent) == cfg.stop_window
                and sum(recent) / len(recent) <= cfg.stop_det
            ):
                print(json.dumps({"stopped": self.step_k, "det_window": sum(recent) / len(recent)}))
                break
            if self.step_k % 10 == 0 or self.step_k == 1:
                print(
                    json.dumps(
                        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}
                    ),
                    flush=True,
                )
            if cfg.save_every and self.step_k % cfg.save_every == 0:
                self.policy.save_adapter(self.out / f"step_{self.step_k}")
            if self.evaluator is not None and cfg.eval_every and self.step_k % cfg.eval_every == 0:
                self.policy.model.eval()
                self.evaluator(self.policy, self.out / f"eval_step_{self.step_k}")
        self.policy.save_adapter(self.out / "final")


# --------------------------------------------------------------------- adaptive covert arm (Arm A)
from next_chameleons.tica.adaptive import (  # noqa: E402
    AdaptiveConfig,
    AdaptiveDefender,
    PursuitState,
    RefitData,
    fit_refit_bank,
    kind_weights,
    progressive_cost,
    pursuit_leash,
    pursuit_prob,
    trim_bank,
)
from next_chameleons.tica.curriculum import (  # noqa: E402
    DiscountedThompson,
    FactoredThompson,
    HierarchicalThompson,
)
from next_chameleons.tica.episodes import (  # noqa: E402
    CONDITION_SPECS,
    MONITOR_OFF,
    MONITOR_ON,
    MONITOR_SWITCH,
    render_feedback,
)
from next_chameleons.tica.level1 import opening  # noqa: E402
from next_chameleons.tica.monitors import auroc, fit_logistic  # noqa: E402
from next_chameleons.tica.neg_labels import clean as neg_clean  # noqa: E402
from next_chameleons.tica.neg_labels import settings as neg_label_settings  # noqa: E402
from next_chameleons.tica.objectives import Target  # noqa: E402
from next_chameleons.tica.probe_zoo import ALL_TOKEN_KINDS as ZOO_TOKEN  # noqa: E402
from next_chameleons.tica.probe_zoo import ENS_KINDS as ZOO_ENS  # noqa: E402
from next_chameleons.tica.probe_zoo import TOK_KEEP as ZOO_TOK_KEEP  # noqa: E402
from next_chameleons.tica.probe_zoo import TOKENS_ENS as ZOO_TOKENS_ENS  # noqa: E402
from next_chameleons.tica.probe_zoo import gate_fit, kind_layers, split_kind  # noqa: E402

DODGE_CONDS = ("real_score", "real_verdict")


def curriculum_prior(fam_of: dict, prior: str) -> dict:
    """Uniform starting distribution over concepts (10-17g): "concept" = 1 / n each; "family" =
    1 / n_families, split evenly within the family."""

    cs = list(fam_of)
    if prior == "concept":
        return {c: 1.0 / len(cs) for c in cs}
    if prior != "family":
        raise ValueError(prior)
    fams = sorted(set(fam_of.values()))
    size = {f: sum(1 for c in cs if fam_of[c] == f) for f in fams}
    return {c: 1.0 / len(fams) / size[fam_of[c]] for c in cs}


def uniform_kind(bandit, rng, exclude=()) -> str | None:
    """A probe kind drawn uniformly (architecture, then layer for single-layer architectures)."""

    ex = set(exclude)
    if isinstance(bandit, FactoredThompson):
        left = [k for k in bandit.fixed if k not in ex] + [
            f"{a}@{L}" for a in bandit.archs for L in bandit.layers if f"{a}@{L}" not in ex]
        if not left:
            return None
        for _ in range(8):  # architecture first, so ensembles are not diluted by the layer count
            a = rng.choice(bandit.archs + bandit.fixed)
            k = a if a in bandit.fixed else f"{a}@{rng.choice(bandit.layers)}"
            if k not in ex:
                return k
        return rng.choice(left)
    left = [k for k in bandit.keys if k not in ex]
    return rng.choice(left) if left else None


def lr_factor(cfg: SupervisedConfig, k: int, offset: int = 0) -> float:
    """LR multiplier (of cfg.lr) at local step k of a chunk starting at global step `offset`.
    Every chunk re-warms linearly over warmup_steps (fresh optimizer state); with lr_peak the
    target is cosine from lr_peak (global step 0) down to lr (global step lr_decay_steps)."""

    import math

    warm = min(1.0, (k + 1) / max(1, cfg.warmup_steps))
    if cfg.lr_peak <= 0 or cfg.lr_decay_steps <= 0:
        return warm
    g = min(1.0, (offset + k) / cfg.lr_decay_steps)
    target = cfg.lr + (cfg.lr_peak - cfg.lr) * 0.5 * (1 + math.cos(math.pi * g))
    return warm * target / cfg.lr


@dataclass
class AdaptiveSupervisedConfig(SupervisedConfig):
    """Arm A: covert multi-turn Neural Chameleons against an in-episode adaptive monitor. The text
    is teacher-forced (no generation), so it is cheap and has no reasoning-budget problem; the
    only channel is the activations. Each turn hides whatever monitor the defender currently shows;
    the defender switches to the best catcher after a pass and the bank grows by online refits."""

    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)


def early_stop_decision(rows: dict[int, float], step: int, window: int, patience: int,
                        min_delta: float, min_steps: int) -> dict:
    """rows {step: validation metric (higher = better)} of the FINISHED checks. Smoothed metric =
    moving average over the last `window` checks (fewer at the start); best = its maximum; stop iff
    step >= min_steps and the last `patience` checks after the best all failed to beat it by
    min_delta. Lenient by design: async evals that have not finished never count against the run."""

    steps = sorted(rows)
    ma = []
    for j in range(len(steps)):
        w = [rows[steps[q]] for q in range(max(0, j - window + 1), j + 1)]
        ma.append(sum(w) / len(w))
    if not ma:
        return {"stop": False, "n_checks": 0, "best_step": None}
    best_j, best = 0, ma[0]
    for j in range(1, len(ma)):
        if ma[j] >= best + min_delta:
            best_j, best = j, ma[j]
    since = len(ma) - 1 - best_j
    return {"stop": step >= min_steps and since >= patience, "n_checks": len(ma),
            "best_step": steps[best_j], "best_smoothed": round(best, 4),
            "last_smoothed": round(ma[-1], 4), "checks_since_best": since,
            "smoothed": {str(k): round(v, 4) for k, v in zip(steps, ma, strict=True)}}


def beat_summary(recs: list[dict], prefix: str = "beat_") -> dict:
    """Turns-to-beat summary of deployed-probe records {kind, turns, beaten} (10-16d): how many new
    probes the model beat at all, how many judged turns that took, and the cumulative fraction
    beaten within 1 / 2 / 3 / 5 turns (of ALL deployments: censored ones count as not yet beaten,
    so these are lower bounds when episodes end early). Overall and per probe kind."""

    def one(rs):
        b = [r["turns"] for r in rs if r["beaten"]]
        out = {"n": len(rs), "frac": len(b) / len(rs) if rs else float("nan"),
               "turns_mean": sum(b) / len(b) if b else float("nan")}
        for k in (1, 2, 3, 5):
            out[f"within{k}"] = sum(x <= k for x in b) / len(rs) if rs else float("nan")
        return out

    m = {f"{prefix}{k}": v for k, v in one(recs).items()}
    for kind in sorted({r["kind"] for r in recs}):
        rs = [r for r in recs if r["kind"] == kind]
        o = one(rs)
        m[f"{prefix}frac_{kind}"] = o["frac"]
        m[f"{prefix}turns_{kind}"] = o["turns_mean"]
        m[f"{prefix}n_{kind}"] = o["n"]
    return m


class AdaptiveSupervisedTrainer:
    """Teacher-forced covert arm: adaptive defender + online refit (REPLICATION 2026-10-11a).
    Reuses the loss (hide active monitor on turns >= hide_from_turn, pin turn 0, match selectivity)
    and the length-sorted micro-batching of SupervisedTrainer."""

    def __init__(self, policy, pool: MonitorPool, sampler, cfg: AdaptiveSupervisedConfig, out_dir,
                 branch_concepts=None, branch_negatives=None, roots_only=None, prompts=None,
                 inputs=None, hard_neg=None):
        self.policy, self.pool, self.sampler, self.cfg = policy, pool, sampler, cfg
        # (opening, render_feedback): repeat format by default; tica/answer_format for "answer".
        self.opening, self.render_feedback = prompts or (opening, render_feedback)
        self.inputs = inputs  # answer format {text_id: request} (capability forks strip it)
        # {concept: matched within-source negatives}: half of a refit fork's other-concept
        # negatives come from these (e.g. DolusChat truthful twins for deception)
        self.hard_neg = hard_neg or {}
        self.falsify = False  # eval-only control (tica_switch_eval --falsify)
        self.deploy_log: list[dict] = []  # turns-to-beat records (rollout; beat_summary)
        self.seg_log: list[dict] = []  # per (episode, concept): turns until its first pass
        self._neg_cache: dict = {}  # concept -> label-cleaned negatives (neg_labels)
        # adaptive curriculum (tica/curriculum.py; default off)
        ac = cfg.adaptive
        self.kind_bandit = (self._make_kind_bandit()
                            if ac.episode_kind_sampling == "thompson" else None)
        concepts = [c.concept for c in getattr(sampler, "concepts", [])]
        self.concept_bandit = None
        if ac.concept_sampling == "thompson" and concepts:
            fams = {}
            if ac.concept_families:
                fams = json.loads(Path(ac.concept_families).read_text()).get("families", {})
            self.concept_bandit = HierarchicalThompson(
                {c: fams.get(c, "topic") for c in concepts}, **self._tts_kw())
        # truncated no-grad reads (Policy._depth; exact). Config field, or TICA_TRUNC_READS=1
        # (evals of runs trained before the field existed).
        if cfg.adaptive.truncate_reads or os.environ.get("TICA_TRUNC_READS") == "1":
            policy.truncate_reads = True
        if os.environ.get("TICA_BATCHED_FITS") == "1":  # same, for refit_batched_fits
            cfg.adaptive.refit_batched_fits = True
        # Branch-refit data (refit_mode="branch"): the FULL concept set (all covered concepts supply
        # the other-concept negatives) + concept-free negatives; roots_only = episode concept.
        self.branch_concepts = branch_concepts
        self.branch_negatives = branch_negatives or []
        self.roots_only = set(roots_only) if roots_only else None
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "bank").mkdir(exist_ok=True)
        self.params = [p for p in policy.model.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("policy has no trainable parameters (attach a LoRA adapter)")
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        off = self._step_offset()
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda k: lr_factor(cfg, k, off))
        self.step_k = 0
        self.refit_k = 0
        self.refit_buf: RefitData | None = None
        self._fresh_escapes = 0  # escapes added to the buffer since the last refit (data-triggered)
        self._latest_refit: str | None = None  # newest online refit name (refit_as_start)
        self._transient: list[str] = []  # episode-local refits (refit_mode="episode")
        self._bank_validated = False
        self.base_names = {e.monitor.name for e in pool.entries}
        self.rng = random.Random(cfg.seed)
        # Pursuit (adaptive.pursuit): running centroid at the concept layer + per-step (i,t) acts;
        # one PursuitState per monitored concept (multi-concept: the clouds differ).
        self.pursuit_by: dict[str, PursuitState] = {}
        self._ep_acts: dict[tuple[int, int], torch.Tensor] = {}
        self._ep_concept: dict[tuple[int, int], str] = {}
        # Probe-kind curriculum (adaptive.episode_kind_sampling): refit name -> kind, first judged
        # turn per (traj, name), and per-kind [caught, judged] on post-first turns (all training).
        self._kind_of: dict[str, str] = {}
        self._kind_first: dict[tuple, int] = {}
        self.kind_stats: dict[str, list[int]] = {}
        torch.manual_seed(cfg.seed)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))

    def _train_units(self, trajs, keys, layers, tok_layers=()):
        """Training forward units. Per-turn: one view per (traj, turn), length-sorted, micro_batch
        views per chunk. Single-pass (adaptive.single_pass, needs keep_think_history): ONE forward
        per trajectory over its final conversation reads every turn exactly as its view would
        (Policy.forward_multi; test_single_pass_matches_per_turn_views) -> cost linear, not
        quadratic, in turns; micro_batch = trajectories per chunk. run(chunk, grad, with_nll)
        returns (keys in chunk, pooled {L: [n_keys, d]}, nll [n_keys] or None)."""

        pol, mb = self.policy, self.cfg.micro_batch
        if self.cfg.adaptive.single_pass:
            if not pol.keep_think_history:
                raise ValueError("single_pass needs model.keep_think_history=True")
            by_traj: dict[int, list[int]] = defaultdict(list)
            for i, t in keys:
                by_traj[i].append(t)
            mvs = {i: pol.multi_view(trajs[i].convs[-1]) for i in by_traj}
            units = sorted(by_traj, key=lambda i: len(mvs[i].ids))
            chunks = [units[a : a + mb] for a in range(0, len(units), mb)]
            self._unit_mvs = mvs  # KL anchor: the reference reads the same spans

            def run(chunk, grad, with_nll, kl_ref=None):
                """kl_ref: {traj: per-span reference targets} -> also returns, via self._kl_sel,
                the per-key KL(base || policy) [n_keys] (with grad)."""
                tok: dict | None = {} if tok_layers else None
                extra = None
                if kl_ref is not None:
                    extra = {"kl_ref": [r for i in chunk for r in kl_ref[i]]}
                pooled, nl = pol.forward_multi([mvs[i] for i in chunk], layers, grad=grad,
                                               token_out=tok,
                                               with_nll=with_nll, extra=extra)
                sel, ck, off = [], [], 0
                for i in chunk:
                    for t in by_traj[i]:
                        sel.append(off + t)
                        ck.append((i, t))
                    off += len(mvs[i].spans)
                ix = torch.tensor(sel, device=pol.device)
                out = {L: pooled[L][ix] for L in layers}
                for L in tok_layers:  # token-pooled monitors read ("tok", L): per-span states
                    out[("tok", L)] = [tok[L][j] for j in sel]
                self._kl_sel = extra["kl"][ix] if extra is not None else None
                return ck, out, (nl[ix] if nl is not None else None)

            return chunks, run
        built = [pol.view(trajs[i].convs[t]) for i, t in keys]
        order = sorted(range(len(keys)), key=lambda k: len(built[k].ids))
        chunks = [order[a : a + mb] for a in range(0, len(order), mb)]

        def run(chunk, grad, with_nll, kl_ref=None):
            if kl_ref is not None:
                raise ValueError("the KL anchor (adaptive.kl_weight) needs adaptive.single_pass")
            pooled, nl = pol.forward([built[k] for k in chunk], layers, grad=grad,
                                     with_nll=with_nll)
            return [keys[k] for k in chunk], pooled, nl

        return chunks, run

    @torch.no_grad()
    def rollout(self, specs):
        """Teacher-forced lockstep rollout with a per-episode defender. Returns trajectories whose
        results[t] carry the ACTIVE monitor's read and name, plus metrics."""

        acfg, pol, pool = self.cfg.adaptive, self.policy, self.pool
        T = self.cfg.turns
        trajs = [Trajectory(s) for s in specs]
        convs = [
            [
                {
                    "role": "user",
                    "content": self.opening(
                        "real_score" if CONDITION_SPECS[s.cond]["brief"] else "nobrief",
                        s.text(0),
                        s.monitored,
                        named=s.named,
                    ),
                }
            ]
            for s in specs
        ]
        ep_mode = acfg.refit_mode == "episode"
        # per episode: renders THIS turn's user message for another text (first_probe="fork")
        umsg = [(lambda x, s=s: self.opening(
            "real_score" if CONDITION_SPECS[s.cond]["brief"] else "nobrief", x, s.monitored,
            named=s.named)) for s in specs]
        self._refit_read_sec = 0.0  # per rollout (sec_ep_refit_read)
        ep_stats = {"att": 0, "acc": 0, "auroc": [], "recov": [], "cens": 0, "sec": 0.0,
                    "switch": 0, "old_catch": 0, "off": 0, "off_det": 0, "switch_att": 0,
                    "all_failed": 0}
        self._refit_log = []  # one record per refit ATTEMPT (flushed to refits.jsonl)
        cumulative = ep_mode and acfg.episode_cumulative

        def _start(s):
            if ep_mode:  # every episode opens on probe 1 (the base concept probe)
                return pool.concept_monitor(s.monitored).name
            # refit_as_start: open on the newest refit (if it covers the concept) so the current
            # dodge is caught at turn 0 by construction; else the spec's assigned monitor.
            if acfg.refit_as_start and self._latest_refit is not None:
                try:
                    if pool.get(self._latest_refit).covers(s.monitored):
                        return self._latest_refit
                except KeyError:
                    pass
            return s.monitor

        # One defender per (episode, monitored concept), created at the concept's first turn:
        # concept-switching episodes (final.concept_switch_prob) keep each concept's defender and
        # deployed probes, so a RETURN to an earlier concept is judged by all of its old probes.
        def_by: list[dict] = [{} for _ in specs]
        dep_by: list[dict] = [{} for _ in specs]  # deployed probes per concept

        def defender(i, s, c):
            if c not in def_by[i]:
                start = _start(s) if c == s.monitored else pool.concept_monitor(c).name
                # "switch_fork" (10-17h): the episode's FIRST concept keeps its base probe; every
                # newly switched-to concept gets a fork-fit first probe
                fork = (acfg.first_probe == "fork" or (acfg.first_probe == "switch_fork"
                                                       and c != s.monitored_at(0)))
                if fork and ep_mode:
                    name = self._fork_first(i, s, c, convs[i], umsg[i], pending, ep_stats)
                    if name is not None:
                        start = name
                elif acfg.first_probe not in ("base", "fork", "switch_fork"):
                    raise ValueError(acfg.first_probe)
                def_by[i][c] = AdaptiveDefender(pool, c, acfg, start=start,
                                                rng=random.Random(self.rng.randrange(1 << 30)))
                dep_by[i][c] = [def_by[i][c].active]
            return def_by[i][c]

        dodging = [s.cond in DODGE_CONDS for s in specs]
        if self.refit_buf is None:
            self.refit_buf = RefitData(layers=pool.layers, concept=specs[0].monitored)
        # Track the centroid / per-turn acts always (turn_move is reported for the pursuit-OFF
        # control too); the pursuit LOSS term is gated on acfg.pursuit below. One centroid per
        # concept (multi-concept runs): concept clouds sit in different places.
        plp = self._plp = pool.concept_monitor(specs[0].monitored).layer
        self._ep_acts, self._ep_concept, self._kind_first = {}, {}, {}
        # Held-out monitors (e.g. L20, never in the training bank): scored every turn for the
        # GENERALISATION -- does the dodge transfer to a probe never trained against, or is it
        # overfit to the bank? Free: these layers are already in pooled.
        ho_cache: dict[str, list] = {}

        def heldout_for(c):
            if c not in ho_cache:
                ho_cache[c] = [e.monitor for e in pool.entries
                               if e.held_out and e.monitor.covers(c)]
            return ho_cache[c]

        cap = acfg.refit_buf_cap
        passed_turn = []
        last_refit: dict[tuple, int] = {}  # (episode, concept) -> turn of its latest refit
        # TURNS-TO-BEAT (10-16d; user: "track how many turns it takes"): every newly deployed probe
        # (accepted refit / bank switch) -> judged turns of its concept until the model first
        # passes it, or censored (never beaten before the episode ends). Records in
        # self.deploy_log ({kind, turns, beaten}); summary metrics beat_* below.
        pending: dict[tuple, dict] = {}
        deploy_start, seg_start = len(self.deploy_log), len(self.seg_log)
        seg_pending: dict[tuple, dict] = {}
        seg_seen: set[tuple] = set()
        self._t0_first = {}  # first_probe="fork": turn-0 probes shared per concept (this rollout)
        inc_state: dict = {}  # incremental_rollout: per-episode caches, freed with the rollout
        # Monitor-off turns (off_turn_prob): drawn per monitored episode, never turn 0.
        off = [[d and t >= 1 and self.rng.random() < acfg.off_turn_prob
                for t in range(T)] for d in dodging]
        roll_tok = self._refit_token_layers(plp)
        for t in range(T):
            for c, s in zip(convs, specs, strict=True):
                c.append({"role": "assistant", "content": s.text(t)})
            if acfg.first_probe in ("fork", "switch_fork") and ep_mode:
                # create this turn's new defenders (fork-fit first probes) BEFORE the read, so a
                # first probe at a layer nothing else reads (e.g. linear@26) is read this turn
                for i, s in enumerate(specs):
                    if dodging[i] and s.monitored_at(t) not in def_by[i]:
                        defender(i, s, s.monitored_at(t))
            tok = {} if roll_tok else None
            # token-pooled refit kinds may read layers no deployed monitor reads (e.g. the 8-layer
            # NC token ensemble, L4-18): read their layers too
            rl = sorted(set(pool.layers) | set(roll_tok))
            pooled = (pol.pooled_incremental(convs, rl, inc_state, token_out=tok)
                      if acfg.incremental_rollout or roll_tok else pol.pooled(convs, pool.layers))
            for L in roll_tok:  # token-pooled refits read this turn's per-token states
                pooled[("tok", L)] = tok[L]
            for i, (s, tr) in enumerate(zip(specs, trajs, strict=True)):
                # layers read THIS turn (a multi-layer refit added mid-turn reads from the next)
                prow = {L: v[i : i + 1] for L, v in pooled.items() if isinstance(L, int)}
                c = s.monitored_at(t)  # the concept monitored this turn
                dfd = defender(i, s, c) if dodging[i] else None
                heldout = heldout_for(c)
                if dfd is not None and off[i][t]:  # unmonitored turn inside a monitored episode
                    base_mon = pool.concept_monitor(c)
                    lg = base_mon.logits(prow[base_mon.layer], [c])[0]
                    det = bool(lg > base_mon.thresholds([c])[0])
                    det_ho = any(
                        float(m.logits(prow[m.layer], [c])[0]
                              > m.thresholds([c])[0])
                        for m in heldout
                    ) if heldout else None
                    res = {"monitored": {"detected": det, "score": 0.0, "thr": 0.0,
                                         "name": base_mon.name},
                           "active_monitor": dfd.active, "turn": t, "off": True, "concept": c,
                           "heldout_det": det_ho, "deployed": list(dep_by[i][c])}
                    ep_stats["off"] += 1
                    ep_stats["off_det"] += det
                    shown, shown_cond = (0.0, 1.0), "none"
                elif dfd is not None:
                    self._ep_acts[(i, t)] = pooled[plp][i : i + 1].detach()
                    self._ep_concept[(i, t)] = c
                    deployed_i = dep_by[i][c]
                    active_name = dfd.active  # the newest probe (an older one may catch below)
                    read = dfd.score(active_name, pooled, i)
                    self._kind_outcome(i, active_name, t, read["detected"])
                    if cumulative and not read["detected"]:
                        # No circularity: an OLDER probe of this episode catching a fallback to
                        # an old region counts as caught (the strongest catcher is shown).
                        olds = [dfd.score(n, pooled, i) for n in deployed_i if n != dfd.active]
                        olds = [r for r in olds if r["detected"]]
                        if olds:
                            read = max(olds, key=lambda r: r["score"] - r["thr"])
                            ep_stats["old_catch"] += 1
                    obs = dfd.observe(read, pooled, i)
                    if ep_mode and (i, c) in pending:
                        pending[(i, c)]["turns"] += 1
                        if obs["passed"]:
                            self.deploy_log.append({**pending.pop((i, c)), "beaten": True})
                    if (i, c) not in seg_seen:  # first judged turn of concept c in episode i
                        seg_seen.add((i, c))
                        seg_pending[(i, c)] = {"concept": c, "kind": "segment", "turns": 0}
                    if (i, c) in seg_pending:
                        seg_pending[(i, c)]["turns"] += 1
                        if obs["passed"]:
                            self.seg_log.append({**seg_pending.pop((i, c)), "beaten": True})
                    # Held-out: does the dodge also evade a probe never trained against?
                    det_ho = any(
                        float(
                            m.logits(prow[m.layer], [c])[0]
                            > m.thresholds([c])[0]
                        )
                        for m in heldout
                    ) if heldout else None
                    res = {
                        "monitored": read,
                        "active_monitor": active_name,
                        "turn": t,
                        "concept": c,
                        "passed": obs["passed"],
                        "novel": obs["novel"],
                        "heldout_det": det_ho,
                        "deployed": list(deployed_i),
                    }
                    if obs["passed"]:
                        passed_turn.append(t)
                        if t >= 1:
                            self.refit_buf.add("pos", pooled, i, cap)
                            self._fresh_escapes += 1
                    shown = (read["score"], read["thr"])
                    if self.falsify:  # eval control: show the OPPOSITE verdict (refits still
                        # follow TRUE passes; forks carry the verdict the model actually saw)
                        th = read["thr"]
                        shown = (th - 1.0, th) if read["detected"] else (th + 1.0, th)
                    shown_cond = s.cond
                    if ep_mode and obs["passed"]:
                        if (i, c) in last_refit:
                            ep_stats["recov"].append(t - last_refit.pop((i, c)))
                        catcher = None
                        if t < T - 1 and acfg.episode_bank_first:
                            catcher = dfd.best_catcher(pooled, i)  # a member that catches it
                        if catcher is not None:  # quick mode: switch, no fork
                            dfd.active, dfd.stage_turn = catcher, 0
                            ep_stats["switch"] += 1
                            last_refit[(i, c)] = t
                            pending[(i, c)] = {"kind": "bank", "turns": 0, "concept": c}
                            res["episode_refit"] = "bank"
                        elif t < T - 1 and s.monitored_at(t + 1) != c:
                            # the next turn monitors another concept: no fork now; if the episode
                            # returns to c, its last-passed probe judges first (and a pass there
                            # refits at that point)
                            dfd.active = read["name"]
                            res["episode_refit"] = "switch"
                        elif t < T - 1 and self.rng.random() >= acfg.episode_refit_prob:
                            dfd.active = read["name"]  # no refit this time: same probe next turn
                            res["episode_refit"] = "skip"
                        elif t < T - 1:  # fork at the turn after the pass -> active from t+1
                            dfd.active = read["name"]  # refit, or keep the dodged probe
                            t_ep = time.time()
                            bank, info = self._episode_refit(s, convs[i], shown, c, t + 1)
                            self._log_refit(info, i, t + 1, c, 0, time.time() - t_ep)
                            tried = [info.get("kind")]
                            # refit_switch_max (10-15g): a refit that fails its quality gate is
                            # NOT replaced by the old (already dodged) probe; another probe TYPE is
                            # fit instead (same sample sizes), up to refit_switch_max more types.
                            while bank is None and len(tried) <= acfg.refit_switch_max:
                                k2 = self._pick_kind(exclude=tried)
                                if k2 is None:
                                    break
                                t_sw = time.time()
                                bank, info = self._episode_refit(s, convs[i], shown, c, t + 1,
                                                                 kind=k2)
                                self._log_refit(info, i, t + 1, c, len(tried),
                                                time.time() - t_sw)
                                tried.append(k2)
                                ep_stats["switch_att"] += 1
                            if bank is None and len(tried) > 1:
                                ep_stats["all_failed"] += 1
                            ep_stats["sec"] += time.time() - t_ep
                            ep_stats["att"] += 1
                            res["episode_refit"] = bool(bank is not None)
                            res["refit_kinds_tried"] = tried
                            if bank is not None:
                                bank = bank.to(pol.device)
                                pool.add(bank, round=self.step_k, family="episode_refit")
                                if acfg.episode_keep and hasattr(bank, "state"):
                                    torch.save(bank.state(), self.out / "bank" / f"{bank.name}.pt")
                                else:
                                    self._transient.append(bank.name)
                                dfd.members.append(bank)
                                dfd.by_name[bank.name] = bank
                                dfd.active, dfd.stage_turn = bank.name, 0
                                deployed_i.append(bank.name)
                                self._kind_of[bank.name] = info.get("kind", "linear")
                                ep_stats["acc"] += 1
                                ep_stats["auroc"].append(info["auroc"])
                                last_refit[(i, c)] = t
                                pending[(i, c)] = {"kind": info.get("kind", "linear"),
                                                   "turns": 0, "concept": c}
                else:
                    base = pool.concept_monitor(s.monitored) if s.text_concept else None
                    if base is not None:
                        lg = base.logits(prow[base.layer], [s.monitored])[0]
                        det = bool(lg > base.thresholds([s.monitored])[0])
                    else:
                        det = False
                    # Held-out on NO-HIDE conds: does L20 fire when the concept is present but the
                    # model is NOT dodging? High on nobrief -> the dodge is SELECTIVE (concept still
                    # readable at L20); low -> generic suppression (concept erased, not hidden).
                    det_ho = any(
                        float(
                            m.logits(prow[m.layer], [s.monitored])[0]
                            > m.thresholds([s.monitored])[0]
                        )
                        for m in heldout
                    ) if (heldout and s.text_concept) else None
                    res = {
                        "monitored": {"detected": det, "score": 0.0, "thr": 0.0},
                        "turn": t,
                        "heldout_det": det_ho,
                    }
                    if s.cond == "nobrief" and s.text_concept:
                        self.refit_buf.add("np", pooled, i, cap)
                    elif s.cond == "neg_text":
                        self.refit_buf.add("neg", pooled, i, cap)
                    shown, shown_cond = (0.0, 1.0), s.cond
                tr.convs.append([dict(x) for x in convs[i]])
                tr.results.append(res)
                if t < T - 1:
                    nxt = s.text(t + 1) if len(s.texts) > 1 else None
                    note = MONITOR_OFF if off[i][t + 1] else (MONITOR_ON if off[i][t] else None)
                    c_next = s.monitored_at(t + 1)
                    if c_next != c:  # concept switch: announced before the new segment
                        sw = MONITOR_SWITCH.format(c=c_next)
                        note = f"{note}\n{sw}" if note else sw
                    # tagged layout (10-17a): the next turn's status line replaces the notices
                    kw = ({"next_concept": c_next, "next_off": bool(off[i][t + 1])}
                          if getattr(self.render_feedback, "tagged", False)
                          else ({"notice": note} if note else {}))
                    fb = self.render_feedback(shown_cond, c, shown, nxt,
                                              feedback=s.feedback, named=s.named, **kw)
                    umsg[i] = (lambda x, sc=shown_cond, cc=c, sh=shown, kw=kw, s=s:
                               self.render_feedback(sc, cc, sh, x, feedback=s.feedback,
                                                    named=s.named, **kw))
                    tr.feedback.append(fb)
                    convs[i].append({"role": "user", "content": fb})
        n_dodge = sum(dodging)
        # Concept switches: detection on the first turn of each new segment (caught by
        # construction ~1) and the pass rate on the later turns of switched-to segments; returns =
        # segments that come back to an earlier concept (all of its old probes judge).
        sw = {"seg_first": [], "seg_later": [], "ret_first": []}
        for tr in trajs:
            mt = tr.spec.meta.get("monitored_turns")
            if not mt or tr.spec.cond not in DODGE_CONDS:
                continue
            for t, r in enumerate(tr.results):
                if t == 0 or r.get("off"):
                    continue
                new_seg = mt[t] != mt[t - 1]
                start = max(u for u in range(t + 1) if u == 0 or mt[u] != mt[u - 1])
                if start == 0:
                    continue
                det = float(r["monitored"]["detected"])
                if new_seg:
                    sw["seg_first"].append(det)
                    if mt[t] in mt[:t]:
                        sw["ret_first"].append(det)
                else:
                    sw["seg_later"].append(1.0 - det)
        for k, v in sw.items():
            if v:
                metrics_sw = sum(v) / len(v)
                ep_stats[f"sw_{k}"] = metrics_sw
        metrics = {
            "pass_rate": len(passed_turn) / max(1, n_dodge * T),
            "mean_pass_turn": (sum(passed_turn) / len(passed_turn))
            if passed_turn
            else float("nan"),
            "bank_size": len(pool.entries),
        }
        metrics.update({k: v for k, v in ep_stats.items() if k.startswith("sw_")})
        for kind, (cg, jd) in self.kind_stats.items():
            metrics[f"kind_catch_{kind}"] = round(cg / max(1, jd), 3)
        if ep_mode:
            ep_stats["cens"] = len(last_refit)  # accepted refits never passed again this episode
            r = ep_stats["recov"]
            self.deploy_log += [{**v, "beaten": False} for v in pending.values()]
            metrics.update(beat_summary(self.deploy_log[deploy_start:]))
            self.seg_log += [{**v, "beaten": False} for v in seg_pending.values()]
            metrics.update(beat_summary(self.seg_log[seg_start:], prefix="segbeat_"))
            # per TASK (concept) and per FAMILY success + turns-to-success (wandb curves, 10-16l)
            fam = self._families()
            recs = self.deploy_log[deploy_start:] + self.seg_log[seg_start:]
            for key, fn in (("concept", lambda r: r.get("concept")),
                            ("family", lambda r: fam.get(r.get("concept"), "topic"))):
                for g in sorted({fn(r) for r in recs if fn(r)}):
                    rs = [r for r in recs if fn(r) == g]
                    b = [r["turns"] for r in rs if r["beaten"]]
                    metrics[f"task_{key}/{g}/beaten"] = len(b) / len(rs)
                    metrics[f"task_{key}/{g}/turns_to_beat"] = (sum(b) / len(b) if b
                                                                 else float("nan"))
                    metrics[f"task_{key}/{g}/hazard"] = len(b) / max(1, sum(r["turns"]
                                                                           for r in rs))
            self._curriculum_update(self.deploy_log[deploy_start:], self.seg_log[seg_start:],
                                    metrics)
            metrics.update({
                "ep_refits": ep_stats["att"],
                "ep_refits_accepted": ep_stats["acc"],
                "ep_bank_switches": ep_stats["switch"],
                "ep_old_catch": ep_stats["old_catch"],  # cumulative: caught only by an older probe
                "ep_off_turns": ep_stats["off"],
                # base probe fires on monitor-OFF turns (selectivity: should stay ~1)
                "ep_off_det": ep_stats["off_det"] / ep_stats["off"] if ep_stats["off"]
                else float("nan"),
                "ep_refit_auroc": sum(ep_stats["auroc"]) / len(ep_stats["auroc"])
                if ep_stats["auroc"] else float("nan"),
                "ep_recovery_mean": sum(r) / len(r) if r else float("nan"),
                "ep_recovered": len(r),
                "ep_censored": ep_stats["cens"],
                "ep_first_probe_att": ep_stats.get("first_att", 0),  # first_probe="fork"
                "ep_first_probe_acc": ep_stats.get("first_acc", 0),
                "ep_first_probe_shared": ep_stats.get("first_shared", 0),
                "sec_ep_refit": round(ep_stats["sec"], 1),
                # split: fork / no-prompt activation reads vs probe fitting + gating
                "sec_ep_refit_read": round(getattr(self, "_refit_read_sec", 0.0), 1),
                "sec_ep_refit_fit": round(ep_stats["sec"] - getattr(self, "_refit_read_sec", 0.0),
                                          1),
                "ep_refit_switch_attempts": ep_stats["switch_att"],  # refit_switch_max retries
                "ep_refit_all_failed": ep_stats["all_failed"],
                # refits whose probes never fired again in the episode / judged turns per refit
                "ep_kinds_tried_per_refit": (ep_stats["att"] + ep_stats["switch_att"])
                / max(1, ep_stats["att"]),
            })
            metrics.update(self._flush_refit_log())
        self._flush_turn_log(trajs)
        return trajs, metrics

    @torch.no_grad()
    def _validate_bank(self, concept: str) -> dict:
        """Quick mode switches to bank members, each must be a real CONCEPT detector: on BASE-model
        out-of-context reads, TPR >= refit_min_tpr_np on concept text and FPR <= 0.1 on
        other-concept + concept-free text, at its own threshold. Failing members (e.g. the dead
        l3loop concept_refit_r2; r1/r3 were gated in-sample before the 10-12a fix) are dropped."""

        pol, acfg = self.policy, self.cfg.adaptive
        texts = {c.concept: c.texts for c in self.branch_concepts or []}
        rng = random.Random(7)
        pos = rng.sample(texts[concept], min(24, len(texts[concept])))
        others = [x for c, ts in texts.items() if c != concept for x in ts]
        neg = rng.sample(others, min(16, len(others))) + rng.sample(
            self.branch_negatives, min(8, len(self.branch_negatives)))

        def reads(xs):
            convs = [[{"role": "user", "content": self.opening("nobrief", x, concept)},
                      {"role": "assistant", "content": x}] for x in xs]
            return pol.pooled(convs, self.pool.layers)

        with pol.reference():
            rp, rn = reads(pos), reads(neg)
        dropped, kept = [], {}
        for e in list(self.pool.entries):
            m = e.monitor
            if e.held_out or not m.covers(concept):
                continue
            thr = m.thresholds([concept])[0]
            tpr = float((m.logits(rp[m.layer], [concept]) > thr).float().mean())
            fpr = float((m.logits(rn[m.layer], [concept]) > thr).float().mean())
            kept[m.name] = (round(tpr, 2), round(fpr, 2))
            if (tpr < acfg.refit_min_tpr_np or fpr > 0.1) and m.name != (
                self.pool.concept_monitor(concept).name
            ):
                self.pool.entries.remove(e)
                dropped.append(m.name)
        return {"bank_valid": kept, "bank_dropped": dropped}

    def _fork_first(self, i, s, c, conv, render, pending, ep_stats):
        """first_probe="fork": fit concept c's FIRST probe of episode i on the policy's states at
        this very turn (conv ends with this turn's user message + answer; forks replace both with
        render(x) + x). Deployed like an accepted refit; None (base probe) if the gate fails."""

        pol, pool = self.policy, self.pool
        t0_key = (c, s.cond, bool(getattr(s, "named", True))) if len(conv) == 2 else None
        if t0_key is not None and t0_key in self._t0_first:
            # TURN 0 has no history: the fork set (opening(x) + x for the concept's texts) depends
            # only on the concept and the current weights, so one fork-fit first probe per concept
            # per rollout serves every episode that opens on it (10-16o speed-up; same probe
            # distribution, one fit instead of one per episode)
            name, kind = self._t0_first[t0_key]
            pending[(i, c)] = {"kind": kind, "turns": 0, "concept": c, "first": True}
            ep_stats["first_shared"] = ep_stats.get("first_shared", 0) + 1
            return name
        t_ep = time.time()
        bank, info = self._episode_refit(s, conv[:-2], None, c, None, fork_msg=render)
        self._log_refit({**info, "first_probe": True}, i, len(conv) // 2 - 1, c, 0,
                        time.time() - t_ep)
        ep_stats["sec"] += time.time() - t_ep
        ep_stats["first_att"] = ep_stats.get("first_att", 0) + 1
        if bank is None:
            return None
        bank = bank.to(pol.device)
        pool.add(bank, round=self.step_k, family="episode_refit")
        self._transient.append(bank.name)
        self._kind_of[bank.name] = info.get("kind", "linear")
        ep_stats["first_acc"] = ep_stats.get("first_acc", 0) + 1
        pending[(i, c)] = {"kind": info.get("kind", "linear"), "turns": 0, "concept": c,
                           "first": True}
        if t0_key is not None:
            self._t0_first[t0_key] = (bank.name, info.get("kind", "linear"))
        return bank.name

    def _episode_refit(self, s, prefix, shown, concept=None, turn=None, kind=None,
                       fork_msg=None):
        """refit_mode="episode": fork this conversation at the turn AFTER its pass (prefix ends with
        the passing answer; each fork = prefix + the shown "not caught" verdict + one follow-up) and
        fit a fresh probe at the concept layer. Shared-prefix reads (one cached forward of the
        history). Positives: in-context concept follow-ups + half the no-prompt concept turns;
        negatives: in-context other-concept + concept-free follow-ups; the other no-prompt half is
        the out-of-context TPR gate. Same construction as scripts/tica_frozen_race.py."""

        acfg, pol = self.cfg.adaptive, self.policy
        t_read0 = time.time()
        ef = {"pos": 24, "np": 10, "other": 16, "free": 8, **acfg.episode_fork}
        concept = concept or s.monitored
        L = self.pool.concept_monitor(concept).layer
        texts = {c.concept: c.texts for c in self.branch_concepts or []}
        if concept not in texts or len(texts) < 2:
            raise ValueError("refit_mode=episode needs branch_concepts with >= 2 concepts")
        pos_t = texts[concept] + self._extra_pos(concept)
        if neg_label_settings(acfg) is not None:  # label-cleaned negatives for this concept
            key = ("neg_clean", concept)
            if key not in self._neg_cache:
                self._neg_cache[key] = (
                    {c: neg_clean(ts, concept, acfg) for c, ts in texts.items() if c != concept},
                    neg_clean(self.branch_negatives, concept, acfg))
            cl, free_negs = self._neg_cache[key]
            texts = {**cl, concept: texts[concept]}
        else:
            free_negs = self.branch_negatives
        others = [c for c in texts if c != concept and texts[c]]
        rng = self.rng
        hard = self.hard_neg.get(concept) or []
        n_hard = min(len(hard), ef["other"] // 2)
        picks = (
            [("pos", x) for x in rng.sample(pos_t, min(ef["pos"], len(pos_t)))]
            + [("neg", x) for x in rng.sample(hard, n_hard)]
            + [("neg", rng.choice(texts[rng.choice(others)])) for _ in range(ef["other"] - n_hard)]
            + [("neg", x) for x in rng.sample(free_negs, min(ef["free"], len(free_negs)))]
        )
        pre = [dict(m) for m in prefix]
        forks = [
            [*pre,
             {"role": "user", "content": fork_msg(x) if fork_msg is not None else
              self.render_feedback(s.cond, concept, shown, x, feedback=s.feedback,
                                   named=s.named)},
             {"role": "assistant", "content": x}]
            for _, x in picks
        ]
        full_kind = kind or self._pick_kind()  # e.g. "linear@22": single-layer kind at layer 22
        kind, Lo = split_kind(full_kind)
        if Lo is not None:
            L = Lo
        Ls = kind_layers(kind, L) if kind in ZOO_ENS or kind in ZOO_TOKENS_ENS else [L]
        ft = {} if kind in ZOO_TOKEN else None  # per-token fork states (token-pooled kinds)
        keep = ZOO_TOK_KEEP if kind in ZOO_TOKENS_ENS else None
        rall = pol.pooled_shared_prefix(forks, Ls, token_out=ft, token_keep=keep,
                                        batch=acfg.episode_fork_batch or None)
        L0 = L if L in Ls else Ls[0]  # an ensemble may not read the primary layer (ensdeep4)
        r, rl = rall[L0].float().cpu(), {Lk: rall[Lk].float() for Lk in Ls}
        # No-prompt texts disjoint from the fork positives where possible, so the OOC gate half
        # never shares a text with a positive.
        used = {x for k, x in picks if k == "pos"}
        rest = [x for x in pos_t if x not in used] or pos_t
        nps = rng.sample(rest, min(ef["np"], len(rest)))
        np_convs = [
            [{"role": "user", "content": self.opening("nobrief", x, concept, named=s.named)},
             {"role": "assistant", "content": x}]
            for x in nps
        ]
        nt = {} if kind in ZOO_TOKEN else None
        if acfg.refit_np_cache:  # shared across the rollout's refits (same weights, same texts)
            nall = self._np_reads(nps, np_convs, Ls, nt, keep)
        else:
            nall = (pol.pooled_incremental(np_convs, Ls, {}, token_out=nt, token_keep=keep)
                    if nt is not None
                    else pol.pooled(np_convs, Ls))
        npr, npl = nall[L0].float().cpu(), {Lk: nall[Lk].float() for Lk in Ls}
        if pol.device.type == "cuda":
            torch.cuda.synchronize()
        self._refit_read_sec = getattr(self, "_refit_read_sec", 0.0) + time.time() - t_read0
        sec_read = round(time.time() - t_read0, 2)  # per-refit read time (refits.jsonl)
        self.refit_k += 1
        name = f"ep_refit_{self.refit_k}"
        if kind != "linear":  # literature probe bank (probe_zoo), same gate
            if kind in ZOO_TOKENS_ENS:
                pos_q = [q for q, (k_, _) in enumerate(picks) if k_ == "pos"]
                neg_q = [q for q, (k_, _) in enumerate(picks) if k_ == "neg"]
                P = {Lk: [ft[Lk][q] for q in pos_q] + nt[Lk][0::2] for Lk in Ls}
                N = {Lk: [ft[Lk][q] for q in neg_q] for Lk in Ls}
                G = {Lk: nt[Lk][1::2] for Lk in Ls}
            elif kind in ZOO_ENS:
                P = {Lk: torch.cat([rl[Lk][q : q + 1] for q, (k_, _) in enumerate(picks)
                                    if k_ == "pos"] + [npl[Lk][0::2]]) for Lk in Ls}
                N = {Lk: torch.stack([rl[Lk][q] for q, (k_, _) in enumerate(picks) if k_ == "neg"])
                     for Lk in Ls}
                G = {Lk: npl[Lk][1::2] for Lk in Ls}
            elif ft is None:  # pooled single-layer zoo kinds (meandiff): rows, not tokens
                P = torch.cat([rl[L0][q : q + 1] for q, (k_, _) in enumerate(picks)
                               if k_ == "pos"] + [npl[L0][0::2]])
                N = torch.stack([rl[L0][q] for q, (k_, _) in enumerate(picks) if k_ == "neg"])
                G = npl[L0][1::2]
            else:
                P = [ft[L][q] for q, (k_, _) in enumerate(picks) if k_ == "pos"] + nt[L][0::2]
                N = [ft[L][q] for q, (k_, _) in enumerate(picks) if k_ == "neg"]
                G = nt[L][1::2]
            mon, info = gate_fit(kind, P, N, G, acfg, name=name, layer=L, concept=concept,
                                 device=pol.device)
            return mon, {**info, "kind": full_kind, "sec_read": sec_read}
        rd = RefitData(layers=[L], concept=concept)
        for (k, _), v in zip(picks, r, strict=True):
            (rd.pos if k == "pos" else rd.neg)[L].append(v[None])
        for j, v in enumerate(npr):
            (rd.pos if j % 2 == 0 else rd.np_pos)[L].append(v[None])
        bank, info = fit_refit_bank(rd, concept, name, acfg, exclude_layers=self._heldout_layers())
        return bank, {**info, "kind": full_kind, "sec_read": sec_read}

    def _cap_gen_step(self, trajs) -> dict:
        """On-policy capability term (adaptive.cap_gen_*; REPLICATION 10-14r). Fork real episodes at
        a random turn t >= 1 (history through turn t-1 + the verdict shown after it + a NEW
        GSM8K-train problem), sample solutions from the CURRENT (pre-update) policy in ONE batched
        generate call, and add grad for the correct ones (exact final answer + visible working):
        cap_gen_weight * NLL (rejection-sampling fine-tuning) + hide_weight * p^2 of the episode's
        probes that judged turn t-1 (cap_gen_hide; a probe first deployed at t is excused, as on
        forced turns). Backward happens here (accumulated before the optimizer step)."""

        from next_chameleons.tica.answer_format import text_id
        from next_chameleons.tica.trainers.mathrl import (
            PROMPT,
            is_correct,
            load_questions,
            work_shown,
        )

        acfg, pol, t0 = self.cfg.adaptive, self.policy, time.time()
        if self.inputs is None:
            raise ValueError("cap_gen needs the answer-format inputs map")
        if not hasattr(self, "_cap_q"):
            if acfg.cap_gen_pool:  # problems with a correct base solution (fallback target)
                g = json.loads(Path(acfg.cap_gen_pool).read_text())
                self._cap_q = [{"question": r["q"], "answer": r["gold"], "base": r["text"]}
                               for part in ("probe", "np", "fork") for r in g[part] if r["ok"]]
            else:
                self._cap_q = load_questions()[0][acfg.cap_gen_skip :]
        forks, meta = [], []
        cands = [i for i, tr in enumerate(trajs)
                 if tr.spec.cond in DODGE_CONDS and len(tr.results) > 1]
        for i in self.rng.sample(cands, min(acfg.cap_gen_episodes, len(cands))):
            tr = trajs[i]
            t = self.rng.randrange(1, len(tr.results))
            prev, cur = tr.results[t - 1], tr.results[t]
            if prev.get("off") or cur.get("off") or cur.get("concept") != prev.get("concept"):
                continue  # forks only inside a monitored segment
            fb = tr.feedback[t - 1]
            req = self.inputs.get(text_id(tr.spec.text(t)), "")
            msg = fb[: len(fb) - len(req)].rstrip() if req and fb.endswith(req) else ""
            it = self.rng.choice(self._cap_q)
            q = PROMPT.format(q=it["question"])
            fork = [*[dict(m) for m in tr.convs[t - 1]],
                    {"role": "user", "content": f"{msg}\n\n{q}" if msg else q}]
            names = [n for n in (prev.get("deployed") or []) if n in set(cur.get("deployed") or [])]
            meta.append({"gold": it["answer"], "names": names, "base": it.get("base"),
                         "concept": cur.get("concept", tr.spec.monitored), "msg": bool(msg),
                         "q": q})
            forks.append(fork)
        if not forks:
            return {"cap_gen/n_forks": 0}
        n = acfg.cap_gen_samples
        convs = [f for f in forks for _ in range(n)]
        was, bs = pol.model.training, pol.batch_size
        pol.model.eval()  # KV cache (HF drops it in train mode with gradient checkpointing)
        pol.batch_size = len(convs)  # one batched call
        try:
            gens = pol.generate(convs, acfg.cap_gen_max_new, temperature=acfg.cap_gen_temperature,
                                seed=self.rng.randrange(1 << 30))
        finally:
            pol.batch_size = bs
            pol.model.train(was)
        t_gen = time.time() - t0
        # items: (conversation, meta, NLL on it?). KL-only items = wrong samples (cap_gen_kl_all).
        kl_w = acfg.cap_gen_kl_weight
        items, n_ok, n_work, n_fb = [], 0, 0, 0
        fork_ok = [False] * len(forks)
        for j, g in enumerate(gens):
            m = meta[j // n]
            ok = g["finished"] and is_correct(g["text"], m["gold"])
            wk = work_shown(g["text"])
            n_ok += ok and wk
            n_work += wk
            if ok and wk:
                fork_ok[j // n] = True
                items.append((convs[j] + [{"role": "assistant", "content": g["text"]}], m, True))
            elif kl_w > 0 and acfg.cap_gen_kl_all and g["text"].strip():
                items.append((convs[j] + [{"role": "assistant", "content": g["text"]}], m,
                              False))
        for f, ok in enumerate(fork_ok):  # every sample wrong: train on the base solution
            if not ok and meta[f]["base"]:
                n_fb += 1
                items.append((forks[f] + [{"role": "assistant", "content": meta[f]["base"]}],
                              meta[f], True))
        info = {"cap_gen/n_forks": len(forks), "cap_gen/n_samples": len(gens),
                "cap_gen/acc": n_ok / len(gens), "cap_gen/work": n_work / len(gens),
                "cap_gen/finished": sum(g["finished"] for g in gens) / len(gens),
                "cap_gen/msg_found": sum(m["msg"] for m in meta) / len(meta),
                "cap_gen/fork_solved": sum(fork_ok) / len(forks), "cap_gen/n_fallback": n_fb,
                "sec_cap_gen": round(t_gen, 1)}
        if not items:
            info["sec_cap_train"] = 0.0
            return info
        ow = self.cfg.objective.hide_weight
        n_hide = sum(len([x for x in m["names"]
                          if not getattr(self.pool.get(x), "needs_tokens", False)])
                     for _, m, _ in items) if acfg.cap_gen_hide else 0
        n_nll = max(1, sum(f for _, _, f in items))
        layers = self.pool.layers
        refs = []  # KL anchor: the base model answering the BARE problem (no brief / history)
        if kl_w > 0:
            with pol.reference(), torch.no_grad():
                for a in range(0, len(items), 2):
                    ch = items[a : a + 2]
                    ex = {"topk": acfg.kl_topk}
                    pol.forward([pol.view([{"role": "user", "content": m["q"]}, c[-1]])
                                 for c, m, _ in ch], [], grad=False, with_nll=True, extra=ex)
                    refs += ex["ref"]
        s_nll, s_hide, s_caught, s_kl, n_klm = 0.0, 0.0, 0, 0.0, 0
        for a in range(0, len(items), 2):
            chunk = items[a : a + 2]
            views = [pol.view(c) for c, _, _ in chunk]
            extra = None
            if kl_w > 0:
                rr = refs[a : a + 2]
                n_tok = [len(v.ids) - v.start for v in views]
                rr = [r if r[0].shape[0] == k else None for r, k in zip(rr, n_tok, strict=True)]
                n_klm += sum(r is None for r in rr)
                extra = {"kl_ref": rr}
            pooled, nll = pol.forward(views, layers, grad=True, with_nll=True, extra=extra)
            w = torch.tensor([float(f) for _, _, f in chunk], device=nll.device)
            loss = acfg.cap_gen_weight * (w * nll).sum() / n_nll
            s_nll += float((w * nll).detach().sum())
            if extra is not None:
                loss = loss + kl_w * extra["kl"].sum() / len(items)
                s_kl += float(extra["kl"].detach().sum())
            if n_hide:
                tg, rows = [], []
                for r, (_, m, _) in enumerate(chunk):
                    for x in m["names"]:
                        if not getattr(self.pool.get(x), "needs_tokens", False):
                            tg.append(Target(r, 0, "cap", x, m["concept"], "hide"))
                            rows.append(r)
                if tg:
                    p = probe_prob(self.pool, pooled, rows, tg)
                    loss = loss + ow * (p**2).sum() / n_hide
                    s_hide += float((p**2).detach().sum())
                    s_caught += int((p > 0.5).sum())
            loss.backward()
        info.update({"cap_gen/n_train": len(items), "cap_gen/nll": s_nll / n_nll,
                     "cap_gen/n_kl_only": sum(not f for _, _, f in items),
                     **({"cap_gen/kl": s_kl / len(items), "cap_gen/kl_misaligned": n_klm}
                        if kl_w > 0 else {}),
                     "cap_gen/hide_p2": s_hide / max(1, n_hide),
                     "cap_gen/caught_frac": s_caught / max(1, n_hide),
                     "sec_cap_train": round(time.time() - t0 - t_gen, 1)})
        return info

    def _anchor_items(self) -> list[list[dict]]:
        """The retain set: anchor_pool general prompts with the BASE model's greedy answers,
        generated once (one batched call) and cached as JSON."""

        if hasattr(self, "_anchor"):
            return self._anchor
        from next_chameleons.tica.anchor import load_anchor_prompts

        acfg, pol = self.cfg.adaptive, self.policy
        path = Path(acfg.anchor_cache) if acfg.anchor_cache else self.out / "anchor_cache.json"
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            prompts = list(dict.fromkeys(load_anchor_prompts(list(acfg.anchor_sources),
                                                             acfg.anchor_pool, seed=0)))
            convs = [[{"role": "user", "content": q}] for q in prompts]
            was, bs = pol.model.training, pol.batch_size
            pol.model.eval()
            pol.batch_size = 32  # all 256 at once OOMed beside the training state (5694563)
            try:
                with pol.reference():
                    gens = pol.generate(convs, acfg.anchor_max_new, temperature=0.0)
            finally:
                pol.batch_size = bs
                pol.model.train(was)
            rows = [{"prompt": q, "answer": g["text"]} for q, g in zip(prompts, gens, strict=True)
                    if g["text"].strip()]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows))
        self._anchor = [[{"role": "user", "content": r["prompt"]},
                         {"role": "assistant", "content": r["answer"]}] for r in rows]
        return self._anchor

    def _anchor_step(self) -> dict:
        """anchor_weight * mean per-token KL(base || policy) on anchor_n retain conversations
        (base top-k + tail targets computed this step). Backward here, before the optimizer step."""

        acfg, pol, t0 = self.cfg.adaptive, self.policy, time.time()
        pool = self._anchor_items()
        convs = self.rng.sample(pool, min(acfg.anchor_n, len(pool)))
        views = [pol.view(c) for c in convs]
        s_kl = 0.0
        for a in range(0, len(views), 2):
            vs = views[a : a + 2]
            ex = {"topk": acfg.kl_topk}
            with pol.reference(), torch.no_grad():
                pol.forward(vs, [], grad=False, with_nll=True, extra=ex)
            extra = {"kl_ref": ex["ref"]}
            pol.forward(vs, [], grad=True, with_nll=True, extra=extra)
            loss = acfg.anchor_weight * extra["kl"].sum() / len(views)
            loss.backward()
            s_kl += float(extra["kl"].detach().sum())
        return {"anchor/kl": s_kl / len(views), "anchor/n": len(views),
                "sec_anchor": round(time.time() - t0, 1)}

    def _np_reads(self, texts, convs, Ls, token_out, keep):
        """No-prompt refit reads through a cache keyed by (weights_version, conversation, layer,
        token mode): they depend only on the current weights and the text, so every refit of the
        rollout shares them. Misses are read in ONE batched call (episode_fork_batch rows)."""

        pol = self.policy
        ver = getattr(pol, "weights_version", 0)
        if getattr(self, "_np_ver", None) != ver:
            self._np_cache, self._np_ver = {}, ver
        cache, tok = self._np_cache, token_out is not None
        keys = [(json.dumps(c, sort_keys=True), tok, keep) for c in convs]
        miss = [j for j, k in enumerate(keys) if any((k, Lk) not in cache for Lk in Ls)]
        if miss:
            t_out = {} if tok else None
            got = pol.pooled([convs[j] for j in miss], Ls, token_out=t_out, token_keep=keep,
                             batch=self.cfg.adaptive.episode_fork_batch or None)
            for q, j in enumerate(miss):
                for Lk in Ls:
                    cache[(keys[j], Lk)] = (got[Lk][q], t_out[Lk][q] if tok else None)
        if tok:
            for Lk in Ls:
                token_out[Lk] = [cache[(k, Lk)][1] for k in keys]
        self._np_hits = getattr(self, "_np_hits", 0) + len(keys) - len(miss)
        return {Lk: torch.stack([cache[(k, Lk)][0] for k in keys]) for Lk in Ls}

    def _nobrief_conv(self, conv: list[dict]) -> list[dict] | None:
        """The unmonitored counterfactual of an answer-format conversation: the same assistant
        texts, each user turn = only that turn's request (no brief / verdict / switch notice).
        None if a text has no request in the inputs map."""

        from next_chameleons.tica.answer_format import text_id

        if self.inputs is None:
            return None
        out = []
        for m in conv:
            if m["role"] != "assistant":
                continue
            req = self.inputs.get(text_id(m["content"])) or self.inputs.get(
                text_id(m["content"].strip()))
            if req is None:
                return None
            out += [{"role": "user", "content": req}, dict(m)]
        return out

    def _kl_targets(self, trajs, chunks):
        """Base-model top-k targets for the KL anchor, per trajectory and span (single-pass units).
        kl_ref "nobrief": read in the unmonitored counterfactual conversation (falls back to the
        identical conversation if it cannot be built or its spans differ; counted)."""

        acfg, pol = self.cfg.adaptive, self.policy
        mvs = self._unit_mvs
        ref_mvs, n_fb = {}, 0
        for i, mv in mvs.items():
            if acfg.kl_ref == "same":
                ref_mvs[i] = mv
                continue
            if acfg.kl_ref != "nobrief":
                raise ValueError(f"kl_ref {acfg.kl_ref!r}")
            conv = self._nobrief_conv(trajs[i].convs[-1])
            rm = pol.multi_view(conv) if conv else None
            if rm is None or [te - a for a, _, te in rm.spans] != [te - a for a, _, te in
                                                                    mv.spans]:
                n_fb, rm = n_fb + 1, mv
            ref_mvs[i] = rm
        out = {}
        with pol.reference(), torch.no_grad():
            for chunk in chunks:
                ex = {"topk": acfg.kl_topk}
                pol.forward_multi([ref_mvs[i] for i in chunk], [], grad=False, with_nll=True,
                                  extra=ex)
                off = 0
                for i in chunk:
                    n = len(ref_mvs[i].spans)
                    out[i] = ex["ref"][off : off + n]
                    off += n
        return out, n_fb

    def _log_refit(self, info: dict, ep: int, turn: int, concept: str, attempt: int,
                   sec: float) -> None:
        """One record per refit attempt: kind, gate outcome and WHY it failed (auroc below
        refit_min_auroc / no-prompt TPR below refit_min_tpr_np / fit error), sizes, time."""

        acfg = self.cfg.adaptive
        lay = info.get("layers") or {}
        au, tp = info.get("auroc"), info.get("tpr_np")
        if au is None and lay:  # linear bank: per-layer gate values
            best = max(lay.values(), key=lambda v: v.get("auroc") or -1.0)
            au, tp = best.get("auroc"), best.get("tpr_np")
        kept = bool(info.get("kept"))
        why = None
        if not kept:
            if info.get("error") or any("error" in v for v in lay.values()):
                why = "error"
            elif au is None or au != au or au < acfg.refit_min_auroc:
                why = "auroc"
            elif tp is None or tp < acfg.refit_min_tpr_np:
                why = "tpr_np"
            else:
                why = "other"
        self._refit_log.append({
            "step": self.step_k, "episode": ep, "turn": turn, "concept": concept,
            "kind": info.get("kind"), "attempt": attempt, "kept": kept, "why": why,
            "auroc": None if au is None else round(float(au), 4),
            "tpr_np": None if tp is None else round(float(tp), 4),
            "n_pos": info.get("n_pos"), "n_neg": info.get("n_neg"), "sec": round(sec, 2),
            "sec_read": info.get("sec_read"), "first_probe": bool(info.get("first_probe"))})

    def _flush_turn_log(self, trajs) -> None:
        """One compact record per episode turn (no text) to <out>/turns.jsonl: condition, concept,
        active probe + its kind, verdict, score vs threshold, older-probe catch, held-out-layer
        detection, refit outcome and the kinds tried -- for post-hoc analyses (which kind caught,
        per-kind dodges, recovery times) without re-running anything."""

        rows = []
        for i, tr in enumerate(trajs):
            for t, r in enumerate(tr.results):
                mon = r.get("monitored") or {}
                act = r.get("active_monitor") or mon.get("name")
                rows.append({
                    "step": self.step_k, "ep": i, "cond": tr.spec.cond, "turn": t,
                    "concept": r.get("concept", tr.spec.monitored), "active": act,
                    "kind": self._kind_of.get(act, "base" if act else None),
                    "detected": bool(mon.get("detected")) if mon else None,
                    "score": None if mon.get("score") is None else round(float(mon["score"]), 4),
                    "thr": None if mon.get("thr") is None else round(float(mon["thr"]), 4),
                    "by": mon.get("name"), "heldout_det": r.get("heldout_det"),
                    "off": bool(r.get("off")), "n_deployed": len(r.get("deployed") or []),
                    "refit": r.get("episode_refit"), "tried": r.get("refit_kinds_tried")})
        if rows:
            self.out.mkdir(parents=True, exist_ok=True)
            with open(self.out / "turns.jsonl", "a") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")

    def _flush_refit_log(self) -> dict:
        """Append this rollout's refit attempts to <out>/refits.jsonl; per-kind gate metrics."""

        recs = getattr(self, "_refit_log", [])
        if not recs:
            return {}
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "refits.jsonl", "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        m = {}
        for r in recs:
            k = r["kind"]
            m[f"gate_att_{k}"] = m.get(f"gate_att_{k}", 0) + 1
            m[f"gate_ok_{k}"] = m.get(f"gate_ok_{k}", 0) + int(r["kept"])
            if r["why"]:
                m[f"gate_fail_{r['why']}_{k}"] = m.get(f"gate_fail_{r['why']}_{k}", 0) + 1
        return m

    def _kind_outcome(self, i: int, name: str, t: int, own_detected: bool) -> None:
        """Track the ACTIVE refit's own verdict on its judged turns after the first (the first is
        caught by construction and excused), per probe kind, for prioritized kind sampling."""

        kind = self._kind_of.get(name)
        if kind is None:
            return
        first = self._kind_first.setdefault((i, name), t)
        if t > first:
            st = self.kind_stats.setdefault(kind, [0, 0])
            st[0] += int(own_detected)
            st[1] += 1

    def _families(self) -> dict:
        if not hasattr(self, "_fam_map"):
            f = self.cfg.adaptive.concept_families
            self._fam_map = json.loads(Path(f).read_text()).get("families", {}) if f else {}
        return self._fam_map

    def _tts_kw(self) -> dict:
        ac = self.cfg.adaptive
        return {"gamma": ac.tts_gamma, "floor": ac.tts_floor, "cap": ac.tts_cap,
                "max_turns": ac.tts_max_turns}

    def _make_kind_bandit(self):
        ac = self.cfg.adaptive
        kinds = list(ac.episode_refit_kinds)
        if ac.kind_layers:
            fixed = [k for k in kinds if "@" in k or split_kind(k)[0] in ZOO_ENS
                     or split_kind(k)[0] in ZOO_TOKENS_ENS]
            archs = [k for k in kinds if k not in fixed]
            return FactoredThompson(archs, fixed, [int(x) for x in ac.kind_layers],
                                    **self._tts_kw())
        return DiscountedThompson(kinds, **self._tts_kw())

    def _curriculum_update(self, deploys, segs, metrics):
        """Feed this rollout's turns-to-beat records to the adaptive curriculum (if on) and set
        the sampler's concept weights for the next step."""

        if self.kind_bandit is not None:
            self.kind_bandit.update(deploys, key="kind")
            for k, v in self.kind_bandit.probs(self.rng).items():
                metrics[f"kind_p_{k}"] = round(v, 3)
        lam = self._curriculum_lambda()
        if self.kind_bandit is not None or self.concept_bandit is not None:
            metrics["curriculum_lambda"] = round(lam, 3)
        if self.concept_bandit is not None:
            self.concept_bandit.update(deploys + segs, key="concept")
            w = self.concept_bandit.probs(self.rng)
            if lam < 1.0:  # ramp (10-17g): mix with the uniform prior
                prior = curriculum_prior(self.concept_bandit.fam_of,
                                         self.cfg.adaptive.curriculum_prior)
                w = {c: (1 - lam) * prior[c] + lam * w.get(c, 0.0) for c in prior}
            self.sampler.concept_weights = w
            for k, v in w.items():
                metrics[f"concept_p_{k}"] = round(v, 3)

    def _curriculum_lambda(self) -> float:
        """Weight of the adaptive (Thompson) part of the curriculum at this global step (10-17g)."""

        ac = self.cfg.adaptive
        if ac.curriculum_ramp_steps <= 0:
            return float(ac.curriculum_mix_max)
        g = self.step_k + self._step_offset()
        return float(ac.curriculum_mix_max) * min(1.0, g / ac.curriculum_ramp_steps)

    def _pick_kind(self, exclude=()) -> str | None:
        """A refit kind ("mixed": prioritized / uniform over episode_refit_kinds). exclude: kinds
        that already failed the gate at this refit (refit_switch_max); None when none is left."""
        acfg = self.cfg.adaptive
        if acfg.episode_refit_kind != "mixed":
            return None if acfg.episode_refit_kind in exclude else acfg.episode_refit_kind
        if acfg.first_kind and acfg.first_kind not in exclude:
            return acfg.first_kind
        kinds = [k for k in acfg.episode_refit_kinds if k not in exclude]
        if not kinds:
            return None
        if acfg.episode_kind_sampling == "thompson":
            if self.kind_bandit is None or (
                    isinstance(self.kind_bandit, DiscountedThompson)
                    and set(self.kind_bandit.keys) != set(acfg.episode_refit_kinds)):
                self.kind_bandit = self._make_kind_bandit()  # kinds changed after init (evals)
            lam = self._curriculum_lambda()
            if lam < 1.0 and self.rng.random() >= lam:  # ramp (10-17g): a uniform kind
                k = uniform_kind(self.kind_bandit, self.rng, exclude)
                if k is not None:
                    return k
            return self.kind_bandit.sample(self.rng, exclude=exclude)
        if acfg.episode_kind_sampling == "prioritized":
            return self.rng.choices(kinds, kind_weights(self.kind_stats, kinds,
                                                        acfg.episode_kind_eps))[0]
        if acfg.episode_kind_sampling != "uniform":
            raise ValueError(acfg.episode_kind_sampling)
        return self.rng.choice(kinds)

    def _pstate(self, concept: str, layer: int) -> PursuitState:
        if concept not in self.pursuit_by:
            self.pursuit_by[concept] = PursuitState(layer, self.cfg.adaptive.pursuit_ema)
        return self.pursuit_by[concept]

    def _extra_pos(self, concept: str) -> list[str]:
        if not hasattr(self, "_extra_cache"):
            self._extra_cache = {}
        if concept not in self._extra_cache:
            from next_chameleons.tica.adaptive import load_extra_pos

            self._extra_cache[concept] = load_extra_pos(self.cfg.adaptive, concept)
        return self._extra_cache[concept]

    def _refit_token_layers(self, layer: int) -> list[int]:
        """Layers whose per-token rollout states the configured refit kinds need."""
        acfg = self.cfg.adaptive
        kinds = list(acfg.episode_refit_kinds) if acfg.episode_refit_kind == "mixed" else [
            acfg.episode_refit_kind]
        if acfg.first_kind:
            kinds.append(acfg.first_kind)
        out = set()
        for k in kinds:
            base, _ = split_kind(k)
            if base in ZOO_TOKEN:
                out.update(kind_layers(k, layer))
                if acfg.kind_layers and "@" not in k and base not in ZOO_TOKENS_ENS:
                    out.update(int(x) for x in acfg.kind_layers)  # factored: any train depth
        return sorted(out)

    def _targets(self, trajs):
        cfg = self.cfg.objective
        out = []
        for i, tr in enumerate(trajs):
            s = tr.spec
            seen: set[str] = set()  # probes judged on earlier monitored turns of this episode
            seen_c: set[str] = set()  # concepts monitored on earlier turns of this episode
            for t, res in enumerate(tr.results):
                c = res.get("concept", s.monitored)  # concept monitored at this turn
                if s.cond in DODGE_CONDS and res.get("off"):  # monitor-off turn: act normal
                    out.append(Target(i, t, "monitored", res["monitored"]["name"], c, "match"))
                elif s.cond in DODGE_CONDS:
                    mon = res["active_monitor"]
                    role = "hide" if t >= cfg.hide_from_turn else (
                        "match" if cfg.pin_first_turn else None
                    )
                    names = res.get("deployed", [mon])
                    if not role:
                        seen.update(names)
                        continue
                    # episode_cumulative: every probe deployed so far is a hide target (no falling
                    # back to an old region); else only the active one.
                    cum = self.cfg.adaptive.episode_cumulative and role == "hide"
                    judge = names if cum else [mon]
                    prev = tr.results[t - 1]["monitored"].get("p") if t > 0 else None
                    for name in judge:
                        r = role
                        # First turn a probe ever judges in this episode: unanticipatable -> no
                        # hide loss (the motion terms still apply there). A concept switch makes
                        # the new concept's base probe new; a RETURN to an earlier concept is not
                        # excused (its probes judged before).
                        if (r == "hide" and self.cfg.adaptive.excuse_new_probe and t > 0
                                and name not in seen):
                            # objective.pin_segment_first (10-15e): the FIRST turn of a newly
                            # introduced concept is pinned to the base model on its base probe,
                            # as turn 0 is -- the first encounter is caught by construction
                            # instead of dodged before any feedback (the eval's segment-position-0
                            # artefact). Returns to an earlier concept are not pinned.
                            r = ("match" if cfg.pin_segment_first and c not in seen_c
                                 and name == mon else "motion")
                        out.append(Target(i, t, "monitored", name, c, r,
                                          prev_p=prev if name == mon else None))
                    seen.update(names)
                    seen_c.add(c)
                elif s.cond in ("nobrief", "neg_text"):
                    out.append(Target(i, t, "monitored", s.monitor, s.monitored, "match"))
                elif s.cond == "mismatch":
                    # brief names concept A, the text is about B: act like the base model on BOTH
                    # (no hiding of A, and B stays readable) -- specificity
                    out.append(Target(i, t, "monitored", s.monitor, s.monitored, "match"))
                    if "text" in s.probes:
                        name, tc = s.probes["text"]
                        out.append(Target(i, t, "text", name, tc, "match"))
        return out

    def _rollout_metrics(self, trajs):
        acc = defaultdict(list)
        for tr in trajs:
            for r in tr.results:
                tag = "t0" if r["turn"] == 0 else "tN"
                acc[f"det_{tr.spec.cond}_{tag}"].append(float(r["monitored"]["detected"]))
                acc[f"detturn_{tr.spec.cond}_{r['turn']}"].append(float(r["monitored"]["detected"]))
                # Held-out (dodge turns): the real test the training bank can't give.
                if r.get("heldout_det") is not None:
                    acc[f"detho_{tr.spec.cond}_{tag}"].append(float(r["heldout_det"]))
                    acc[f"detho_turn_{r['turn']}"].append(float(r["heldout_det"]))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def step(self):
        cfg, pol, pool = self.cfg, self.policy, self.pool
        t0 = time.time()
        specs = self.sampler.sample(cfg.episodes_per_step)
        pol.model.eval()
        bank_info = {}
        quick = cfg.adaptive.refit_mode == "episode" and cfg.adaptive.episode_bank_first
        if quick and not self._bank_validated:  # once, before the first switch can happen
            bank_info = self._validate_bank(specs[0].monitored)
            self._bank_validated = True
            print(json.dumps(bank_info), flush=True)
        with (pol.merged() if cfg.adaptive.merged_reads else nullcontext()):  # rollout reads
            trajs, extra = self.rollout(specs)
        t_roll = time.time() - t0
        metrics = self._rollout_metrics(trajs)
        targets = self._targets(trajs)
        acfg = cfg.adaptive
        # Pursuit: update the running centroid from this step's dodge activations, then the hide
        # targets that have at least one recent-turn activation to be pushed off.
        plp = getattr(self, "_plp", None)
        epc = self._ep_concept
        by_c = defaultdict(list)
        for kk, a in self._ep_acts.items():
            by_c[epc[kk]].append(a)
        for cc, acts in by_c.items():  # one centroid / scale per monitored concept
            self._pstate(cc, plp).update(torch.cat(acts).to(pol.device))

        def same_seg(i_, t_):  # turn t_-1 judged the same concept as t_ (not a switch turn)
            return (i_, t_ - 1) in self._ep_acts and epc.get((i_, t_ - 1)) == epc.get((i_, t_))

        pursuit_ks = set()
        if acfg.pursuit:
            seen_turns = set()  # one pursuit term per (episode, turn), however many hide targets
            for k, tg in enumerate(targets):
                if tg.role in ("hide", "motion") and tg.turn >= acfg.pursuit_from_turn and (
                    same_seg(tg.traj, tg.turn)
                ) and self.pursuit_by[epc[(tg.traj, tg.turn)]].ready() and (
                    (tg.traj, tg.turn) not in seen_turns
                ):
                    seen_turns.add((tg.traj, tg.turn))
                    pursuit_ks.add(k)
        n_pursuit = max(1, len(pursuit_ks))
        # Direct "is it moving" signal, independent of the loss: mean ||a_t - a_{t-1}|| over dodge
        # turns at the concept layer (normalised by the cloud scale).
        if plp is not None and self._ep_acts:
            moves = [
                (float((self._ep_acts[(i_, t_)] - self._ep_acts[(i_, t_ - 1)]).norm()),
                 self.pursuit_by[epc[(i_, t_)]].scale)
                for (i_, t_) in self._ep_acts
                if same_seg(i_, t_)
            ]
            if moves:
                metrics["turn_move"] = sum(m for m, _ in moves) / len(moves)
                metrics["turn_move_norm"] = sum(m / max(sc_, 1e-3) for m, sc_ in moves) / len(
                    moves)
        keys = sorted({(tg.traj, tg.turn) for tg in targets})
        layers = pool.layers
        tok_layers = pool.token_layers
        if tok_layers and not acfg.single_pass:
            raise ValueError("token-pooled monitors need adaptive.single_pass")
        chunks, run = self._train_units(trajs, keys, layers, tok_layers)
        tkey = [(tg.traj, tg.turn) for tg in targets]
        ref = torch.empty(len(targets), device=pol.device)
        ref_acts = {}  # base-model reads at the pursuit layer (stay_offset)
        with pol.reference():
            for chunk in chunks:
                ckeys, pooled, _ = run(chunk, grad=False, with_nll=False)
                pos = {kk: j for j, kk in enumerate(ckeys)}
                if acfg.stay_offset and plp is not None:
                    for kk, j in pos.items():
                        ref_acts[kk] = pooled[plp][j].detach()
                ks = [k for k, kk in enumerate(tkey) if kk in pos]
                if ks:
                    local = [pos[tkey[k]] for k in ks]
                    ref[ks] = probe_prob(pool, pooled, local, [targets[k] for k in ks]).detach()
        pol.model.train()
        self.opt.zero_grad(set_to_none=True)
        acc = defaultdict(float)
        totals = {
            "hide": sum(tg.role == "hide" for tg in targets),
            "match": sum(tg.role == "match" for tg in targets),
            "views": len(keys),
        }
        # Turn-weighted hide: fail-early-cheap, dodge-late-expensive (iteration). Weight per hide
        # target = progressive_cost(turns facing this monitor); normalised by the step total.
        hw = None
        if acfg.hide_turn_weighted:
            hw = [
                progressive_cost(max(0, tg.turn - cfg.objective.hide_from_turn), acfg.miss_costs)
                if tg.role == "hide" else 0.0
                for tg in targets
            ]
            totals["hide_w_sum"] = sum(hw)
        loss_sum = 0.0
        kl_ref, kl_fb = (self._kl_targets(trajs, chunks) if acfg.kl_weight > 0 else (None, 0))
        for chunk in chunks:
            ckeys, pooled, nll = run(chunk, grad=True, with_nll=True, kl_ref=kl_ref)
            pos = {kk: j for j, kk in enumerate(ckeys)}
            ks = [k for k, kk in enumerate(tkey) if kk in pos]
            local = [pos[tkey[k]] for k in ks]
            p = probe_prob(pool, pooled, local, [targets[k] for k in ks])
            prev = torch.tensor(
                [targets[k].prev_p if targets[k].prev_p is not None else float("nan") for k in ks],
                device=pol.device,
            )
            hw_chunk = (
                torch.tensor([hw[k] for k in ks], device=pol.device) if hw is not None else None
            )
            loss, st = loss_terms(
                p, ref[ks], [targets[k].role for k in ks], nll, cfg.objective, totals, prev_p=prev,
                hide_w=hw_chunk,
            )
            if kl_ref is not None:  # capability anchor: every trained turn, hiding ones included
                kls = self._kl_sel
                loss = loss + acfg.kl_weight * kls.sum() / max(1, totals["views"])
                acc["kl_sum"] += float(kls.detach().sum())
                acc["kl_n"] += len(kls)
            # Pursuit: push each hide turn off its recent-turn regions (orbit the cloud).
            pk = [k for k in ks if k in pursuit_ks]
            if pk:
                s_pur = pooled[plp].new_zeros(())
                s_leash = pooled[plp].new_zeros(())
                s_stay = pooled[plp].new_zeros(())
                for k in pk:
                    i_, t_ = targets[k].traj, targets[k].turn
                    ps = self.pursuit_by[epc[(i_, t_)]]
                    c, sc = ps.centroid, ps.scale
                    a_t = pooled[plp][pos[tkey[k]]]
                    if acfg.motion == "conditional":
                        prev = self._ep_acts[(i_, t_ - 1)].reshape(-1)
                        if trajs[i_].results[t_ - 1].get("passed"):  # passing: stay put
                            if acfg.stay_offset:  # keep the same dodge (offset from base)
                                d = (a_t.float() - ref_acts[(i_, t_)].float()) - (
                                    prev.float() - ref_acts[(i_, t_ - 1)].float())
                            else:
                                d = a_t.float() - prev.float()
                            s_stay = s_stay + (d.norm() / sc) ** 2
                            acc["n_stay"] += 1
                        else:  # caught: move away from where it was caught
                            s_pur = s_pur + pursuit_prob(a_t, [prev], c, sc,
                                                         acfg.pursuit_temp) ** 2
                            acc["n_move"] += 1
                    else:
                        recents = [
                            self._ep_acts[(i_, t_ - j)]
                            for j in range(1, acfg.pursuit_window + 1)
                            if (i_, t_ - j) in self._ep_acts
                        ]
                        s_pur = s_pur + pursuit_prob(a_t, recents, c, sc,
                                                     acfg.pursuit_temp) ** 2
                    if acfg.pursuit_leash:
                        s_leash = s_leash + pursuit_leash(a_t, c, sc, acfg.pursuit_radius)
                loss = loss + acfg.pursuit_weight * s_pur / n_pursuit
                loss = loss + acfg.stay_weight * s_stay / n_pursuit
                acc["stay_sq"] += float(s_stay.detach())
                if acfg.pursuit_leash:
                    loss = loss + acfg.pursuit_leash * s_leash / n_pursuit
                acc["pursuit_sq"] += float(s_pur.detach())
                acc["leash"] += float(s_leash.detach())
                acc["n_pursuit"] += len(pk)
            loss.backward()
            loss_sum += float(loss.detach())
            for k2, v in st.items():
                acc[k2] += v
        cap_info = (self._cap_gen_step(trajs)
                    if acfg.cap_gen_weight > 0 or acfg.cap_gen_kl_weight > 0 else {})
        if acfg.anchor_weight > 0:  # optional retain-set KL anchor (off by default)
            cap_info |= self._anchor_step()
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
        self.opt.step()
        pol.weights_version = getattr(pol, "weights_version", 0) + 1  # vLLM re-sync
        t_train = time.time() - t0 - t_roll
        self.sched.step()
        self.step_k += 1
        # Refit trigger: data-driven (the instant refit_min_each fresh escapes have accrued, the
        # maximally tight valid chase) and/or on a step cadence.
        if self._transient:  # episode-local refits: used by this step's loss, then dropped
            gone = set(self._transient)
            self.pool.entries = [e for e in self.pool.entries if e.monitor.name not in gone]
            self._transient = []
        refit_info = {}
        if acfg.refit_mode == "episode":  # refits happen inside the rollout, per dodge
            if acfg.episode_keep:  # kept refits: trim AFTER the loss used them
                trim_bank(self.pool, self.base_names, acfg.refit_max_bank)
                refit_info = {"bank_size": len(self.pool.entries)}
        elif acfg.refit_mode == "branch":
            # Fork-on-pass: once the policy is escaping at all, fork it for a fresh valid probe.
            due = acfg.branch_refit_every and self.step_k % acfg.branch_refit_every == 0
            if due and extra.get("pass_rate", 0.0) > 0.0:  # pass_rate lives in rollout extra
                refit_info = self._branch_refit()
        else:
            ready = acfg.refit_when_ready and self._fresh_escapes >= acfg.refit_min_each
            cadence = acfg.refit_every and self.step_k % acfg.refit_every == 0
            if ready or cadence:
                refit_info = self._online_refit()
                if refit_info.get("refit") == "added":
                    self._fresh_escapes = 0
        pursuit_info = {}
        if kl_ref is not None:
            pursuit_info["train/kl"] = acc["kl_sum"] / max(1.0, acc["kl_n"])
            pursuit_info["train/kl_ref_fallback"] = kl_fb
        if acfg.pursuit:
            np_ = acc.get("n_pursuit", 0.0)
            pursuit_info |= {
                "train/pursuit_p": (acc.get("pursuit_sq", 0.0) / np_) ** 0.5 if np_ else 0.0,
                "train/leash": acc.get("leash", 0.0) / np_ if np_ else 0.0,
                "train/n_pursuit": int(np_),
                "train/stay": acc.get("stay_sq", 0.0) / max(1.0, acc.get("n_stay", 0.0)),
                "train/n_stay": int(acc.get("n_stay", 0)),
                "train/n_move": int(acc.get("n_move", 0)),
                "pursuit_scale": round(sum(p.scale for p in self.pursuit_by.values())
                                       / max(1, len(self.pursuit_by)), 3),
            }
        t_refit = time.time() - t0 - t_roll - t_train
        t1 = time.time()
        every = self.cfg.adaptive.reloc_every
        reloc_info = self._relocation_diag() if every and self.step_k % every == 0 else {}
        t_diag = time.time() - t1
        rec = {
            "step": self.step_k,
            "lr": self.sched.get_last_lr()[0],
            "grad_norm": float(gnorm),
            "n_views": len(keys),
            "n_targets": len(targets),
            "sec": round(time.time() - t0, 2),
            "sec_rollout": round(t_roll, 1),
            "sec_train": round(t_train, 1),
            "sec_refit": round(t_refit, 1),
            "sec_diag": round(t_diag, 1),
            "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if torch.cuda.is_available() else None,
            **{f"train/{k}": v for k, v in {"loss": loss_sum, **summarize_stats(acc)}.items()},
            **{f"rollout/{k}": v for k, v in {**metrics, **extra}.items()},
            **pursuit_info,
            **cap_info,
            **reloc_info,
            **refit_info,
            **bank_info,
        }
        with (self.out / "train_log.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def _relocation_diag(self) -> dict:
        """Relocation vs erasure: fit a FRESH probe per layer on concept-present (np) vs
        concept-free (neg) buffered states, report HELD-OUT AUROC. High at a held-out layer (L20)
        = concept still linearly RECOVERABLE (dodge RELOCATED it, base probe looks the wrong way);
        ~0.5 = ERASED there. Cheap: no forward, reuses buffered acts."""

        buf = self.refit_buf
        if buf is None:
            return {}
        out = {}
        for L in self.pool.layers:
            npz, neg = buf.np_pos.get(L, []), buf.neg.get(L, [])
            if len(npz) < 16 or len(neg) < 16:
                continue
            P, N = torch.cat(npz), torch.cat(neg)
            kp, kn = int(len(P) * 0.7), int(len(N) * 0.7)
            x = torch.cat([P[:kp], N[:kn]])
            y = torch.cat([torch.ones(kp), torch.zeros(kn)])
            try:
                w, b, mu, sd = fit_logistic(x, y, self.cfg.adaptive.refit_C)
            except RuntimeError:
                continue
            lg = lambda z, w=w, b=b, mu=mu, sd=sd: ((z.float() - mu) / sd) @ w + b  # noqa: E731
            out[f"reloc_auroc_L{L}"] = round(auroc(lg(P[kp:]), lg(N[kn:])), 3)
        return out

    def _online_refit(self):
        buf = self.refit_buf
        if buf is None or not buf.ready(self.cfg.adaptive.refit_min_each):
            return {"refit": "not_enough_data", "refit_counts": list(buf.counts()) if buf else None}
        self.refit_k += 1
        name = f"refit_online_{self.refit_k}"
        bank, info = fit_refit_bank(buf, buf.concept, name, self.cfg.adaptive,
                                    exclude_layers=self._heldout_layers())
        if bank is None:
            return {"refit": "rejected", "refit_info": info}
        self.pool.add(bank.to(self.policy.device), round=self.step_k, family="concept_refit")
        self._latest_refit = name  # newest refit (never trimmed first) -> refit_as_start uses it
        trim_bank(self.pool, self.base_names, self.cfg.adaptive.refit_max_bank)
        torch.save(bank.state(), self.out / "bank" / f"{name}.pt")
        return {"refit": "added", "refit_layer": info["layer"], "refit_auroc": info["auroc"],
                "refit_tpr_np": info["tpr_np"], "bank_size": len(self.pool.entries)}

    def _heldout_layers(self) -> set[int]:
        """Layers used ONLY by held-out eval monitors: refits must never be fit there."""
        train = {e.monitor.layer for e in self.pool.entries if not e.held_out}
        return {e.monitor.layer for e in self.pool.entries if e.held_out} - train

    def _branch_refit(self):
        """Fork the CURRENT policy at its bypassing states into many one-turn follow-ups (concept /
        other-concept / free) -> a full, balanced, VALID probe from ONE collection (user's design).
        Needs >= 2 covered concepts (the others supply the other-concept negative class)."""
        from next_chameleons.tica.branching import (
            COND_FREE,
            COND_OTHER,
            COND_POS,
            COND_POS_NP,
            BranchConfig,
            branch_collect,
        )
        acfg = self.cfg.adaptive
        if not self.branch_concepts or len(self.branch_concepts) < 2:
            return {"refit": "no_branch_data"}
        bc = BranchConfig(**acfg.branch) if acfg.branch else BranchConfig()
        self.policy.model.eval()
        try:
            with torch.no_grad():
                acts, labels, rep = branch_collect(
                    self.policy, self.pool, self.branch_concepts, self.branch_negatives,
                    bc, self.pool.layers, seed=20_000 + self.step_k, roots_only=self.roots_only,
                )
        except RuntimeError as err:  # branch_collect raises when no root bypassed: not a crash
            if "no root bypassed" not in str(err):
                raise
            return {"refit": "no_bypass"}
        finally:
            self.policy.model.train()
        if not labels:
            return {"refit": "no_bypass", "branch_roots_bypassed": rep.get("roots_bypassed")}
        concept = (
            next(iter(self.roots_only)) if self.roots_only
            else self.sampler.concepts[0].concept
        )
        rd = RefitData(layers=self.pool.layers, concept=concept)
        # No-prompt concept rows: alternate them between the positives (keeps the probe a CONCEPT
        # detector, 10-10) and the held-out out-of-context TPR check (never scored on fit rows).
        NEG = {COND_OTHER, COND_FREE}
        k_np = 0
        for i, lab in enumerate(labels):
            np_to_pos = False
            if lab["cond"] == COND_POS_NP:
                np_to_pos, k_np = k_np % 2 == 0, k_np + 1
            for L in self.pool.layers:
                vec = acts[L][i : i + 1].float().cpu()
                if lab["cond"] == COND_POS or np_to_pos:
                    rd.pos[L].append(vec)
                elif lab["cond"] == COND_POS_NP:
                    rd.np_pos[L].append(vec)
                elif lab["cond"] in NEG:
                    rd.neg[L].append(vec)
        counts = rd.counts()
        if min(counts) < 2:
            return {"refit": "branch_too_few", "refit_counts": list(counts)}
        self.refit_k += 1
        name = f"refit_branch_{self.refit_k}"
        bank, info = fit_refit_bank(rd, concept, name, acfg, exclude_layers=self._heldout_layers())
        if bank is None:
            return {"refit": "rejected", "refit_info": info}
        self.pool.add(bank.to(self.policy.device), round=self.step_k, family="concept_refit")
        self._latest_refit = name  # active next round (refit_as_start) -> static dodge now caught
        trim_bank(self.pool, self.base_names, acfg.refit_max_bank)
        torch.save(bank.state(), self.out / "bank" / f"{name}.pt")
        return {"refit": "added", "refit_src": "branch", "refit_layer": info["layer"],
                "refit_auroc": info["auroc"], "refit_tpr_np": info["tpr_np"],
                "branch_pos": counts[0], "bank_size": len(self.pool.entries)}

    def _wandb(self):
        """Offline wandb run (adaptive.wandb_project / TICA_WANDB_PROJECT), created lazily."""

        if getattr(self, "_wb", None) is not None or getattr(self, "_wb_off", False):
            return getattr(self, "_wb", None)
        proj = self.cfg.adaptive.wandb_project or os.environ.get("TICA_WANDB_PROJECT", "")
        if not proj:
            self._wb_off = True
            return None
        try:
            import dataclasses

            import wandb

            os.environ.setdefault("WANDB_MODE", "offline")
            name = self.out.parent.name
            run_id = name  # one run across resume chunks (resume="allow")
            self._wb = wandb.init(project=proj, name=name, id=run_id, dir=str(self.out),
                                  resume="allow",
                config=dataclasses.asdict(self.cfg) if dataclasses.is_dataclass(self.cfg)
                else {}, settings=wandb.Settings(silent=True))
        except Exception as err:  # never kill training over logging
            print(json.dumps({"wandb": f"disabled: {err}"}), flush=True)
            self._wb_off = True
            self._wb = None
        return self._wb

    def train(self):
        cfg = self.cfg
        while self.step_k < cfg.steps:
            rec = self.step()
            wb = self._wandb()
            if wb is not None:
                wb.log({k: v for k, v in rec.items() if isinstance(v, (int, float))},
                       step=self.step_k)
            if self.step_k % 10 == 0 or self.step_k == 1:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in rec.items()}), flush=True)
            if cfg.save_every and self.step_k % cfg.save_every == 0:
                self.policy.save_adapter(self.out / f"step_{self.step_k}")
                self._eval_on_save(self.out / f"step_{self.step_k}")
                if self._early_stop_check():
                    break
        self.policy.save_adapter(self.out / "final")
        if getattr(self, "_wb", None) is not None:
            self._wb.finish()

    @staticmethod
    def _step_offset() -> int:
        """Global step of this resume chunk's step 0 (tica_resume.sh exports TICA_STEP_OFFSET)."""
        try:
            return int(os.environ.get("TICA_STEP_OFFSET", "0") or 0)
        except ValueError:
            return 0

    def _early_stop_check(self) -> bool:
        """Lenient early stopping on finished validation results (AdaptiveConfig.early_stop_*)."""

        acfg = self.cfg.adaptive
        if not acfg.early_stop_glob:
            return False
        rows = {}
        for f in sorted(self.out.parent.glob(acfg.early_stop_glob)
                        if not Path(acfg.early_stop_glob).is_absolute()
                        else Path("/").glob(acfg.early_stop_glob.lstrip("/"))):
            try:
                d = json.loads(f.read_text())
                rows[int(d["step"])] = float(d[acfg.early_stop_metric])
            except (OSError, ValueError, KeyError, TypeError):
                continue  # still being written / malformed
        dec = early_stop_decision(rows, self.step_k + self._step_offset(), acfg.early_stop_window,
                                  acfg.early_stop_patience, acfg.early_stop_min_delta,
                                  acfg.early_stop_min_steps)
        (self.out / "early_stop.json").write_text(json.dumps(dec, indent=1))
        wb = self._wandb()
        if wb is not None and rows:  # validation curve (latest finished check)
            last = max(rows)
            wb.log({"val/score": rows[last], "val/step": last,
                    "val/smoothed": dec.get("last_smoothed"),
                    "val/best_smoothed": dec.get("best_smoothed")}, step=self.step_k)
        if dec["stop"]:
            print(json.dumps({"early_stop": dec}), flush=True)
        return dec["stop"]

    def _eval_on_save(self, adapter: Path) -> None:
        """adaptive.eval_on_save: launch (not wait for) a checkpoint eval, e.g. an sbatch."""

        cmd = self.cfg.adaptive.eval_on_save
        if not cmd:
            return
        import shlex
        import subprocess

        line = cmd.format(adapter=adapter, step=self.step_k + self._step_offset())
        try:
            # a nested sbatch must not inherit this job's SLURM_* settings (e.g. memory per node)
            env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
            r = subprocess.run(shlex.split(line), capture_output=True, text=True, timeout=120,
                               env=env, cwd=str(Path(__file__).resolve().parents[4]))
            msg = (r.stdout + r.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as err:  # never kill training over an eval
            msg = f"eval_on_save failed: {err}"
        print(json.dumps({"eval_on_save": line, "result": msg}), flush=True)
