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
import math
import os
import queue
import random
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from next_chameleons.tica.episodes import EpisodeSampler, Trajectory, rollout
from next_chameleons.tica.monitors import MonitorPool
from next_chameleons.tica.objectives import (
    ObjectiveConfig,
    hide_per_target,
    loss_terms,
    plan_targets,
    probe_prob,
    retry_weight,
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
    # Optional end cooldown (10-18; default 0 = off): over the LAST lr_cooldown_steps global steps
    # of the run (chunk offset + steps), cosine from the schedule's value down to lr_final.
    lr_cooldown_steps: int = 0
    lr_final: float = 0.0
    # Trainer state with checkpoints (10-18; default off): AdamW moments, curriculum bandits and RNG
    # states go to <checkpoint>/trainer_state.pt (newest checkpoint only, to bound disk), and a
    # chunk resumed from such a checkpoint restores them and skips the LR re-warmup.
    save_state: bool = False
    # 10-21 (default off): keep trainer_state.pt in EVERY checkpoint instead of the newest only, so
    # a run can be resumed with its optimizer state from any checkpoint (~0.5 GB each at r32).
    save_state_all: bool = False
    # 10-29 (0 = off: every copy kept): with save_state_all, keep trainer_state.pt only in this
    # chunk's newest state_keep_last checkpoints plus those at GLOBAL steps divisible by
    # state_keep_every (0 = none), so long runs can still branch from recent / round steps without
    # filling the disk (Gemma-12B: 4.2 GB per copy; the scratch quota stopped every job on 10-28).
    state_keep_last: int = 0
    state_keep_every: int = 0
    # 10-21v (default off = unchanged): with data-parallel episodes, copy rank 0's trainable
    # weights to every rank before training. A fresh LoRA is initialised unseeded per rank, so
    # from-scratch DP runs otherwise train ranks holding DIFFERENT weights (dp_param_drift).
    dp_sync_init: bool = False
    # TamIA 10-21: stop the run (RuntimeError) if the ranks' LoRA weights differ at a checkpoint
    # (default False = log dp_param_drift only, the old behaviour; 10-21v ran 120 steps drifted)
    dp_drift_fatal: bool = False
    # updates_per_rollout (10-18j, default 1 = unchanged): optimizer updates per rollout. The
    # rollout (episodes, verdicts, refits, targets) is collected once and reused K times; each
    # update re-runs the forward with grad on the current weights. The LR schedule still advances
    # once per step.
    updates_per_rollout: int = 1
    grad_clip: float = 1.0
    # Per-term hide clip (REPLICATION 2026-10-21r; 0 = off, the original single backward). > 0:
    # the gradient of the HIDE terms (episode hide targets + cap_gen hide) is accumulated in its own
    # buffer over the step's micro-batches (and summed over data-parallel ranks), clipped to this
    # L2 norm, then added to the other terms' gradient; grad_clip (total norm) still applies after.
    # Adaptive trainer only. Logs grad_norm_hide / grad_norm_rest (pre-clip, global).
    hide_grad_clip: float = 0.0
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
        if cfg.hide_grad_clip:
            raise NotImplementedError("train.hide_grad_clip is implemented in the adaptive trainer")
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
            "off": sum(tg.role == "off" for tg in targets),
            "panel": sum(tg.role == "panel" for tg in targets),
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
from next_chameleons.tica import dp  # noqa: E402
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
    budget_action,
    budget_switch,
    render_feedback,
)
from next_chameleons.tica.evdet_defender import CAL_READS as EVDET_CAL  # noqa: E402
from next_chameleons.tica.evdet_defender import KINDS as DETECTOR_KINDS  # noqa: E402
from next_chameleons.tica.evdet_defender import EvDetBuffer, fit_evdet  # noqa: E402
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
from next_chameleons.tica.validity import check_task as check_valid  # noqa: E402

DODGE_CONDS = ("real_score", "real_verdict")


def probe_index_metrics(trajs, max_k: int = 5) -> dict:
    """PASS RATE BY PROBE INDEX (REPLICATION 2026-10-17u; user: always track this, not only the
    headline pass rate). Within each monitored concept segment of an episode, the k-th BANK is the
    set of probes deployed when the k-th probe arrives (k = 1: the segment's first probe -- the
    turn-0 base / fork probe or a switch's first probe; k = 2: after the first refit; ...).
    beatk_frac_k = fraction of segments that faced bank k and passed it before the segment ended;
    beatk_turns_k = mean judged turns to that pass; beatk_n_k = segments facing bank k (k >= max_k
    pooled). The first bank is also split into the episode's first concept (beat1_first_frac) and
    switched-to concepts (beat1_switch_frac)."""

    seen = defaultdict(lambda: [0, 0, 0])  # k -> [faced, beaten, turns to beat]
    first = [0, 0]
    switch = [0, 0]
    for tr in trajs:
        if tr.spec.cond not in DODGE_CONDS:
            continue
        segs: dict = {}
        for t, res in enumerate(tr.results):
            if res.get("off"):
                continue
            c = res.get("concept", tr.spec.monitored)
            k = min(max_k, max(1, len(res.get("deployed") or [1])))
            seg = segs.setdefault(c, {"first": t == 0, "banks": {}})
            b = seg["banks"].setdefault(k, {"start": t, "beat": None})
            if b["beat"] is None and not res["monitored"].get("detected", True):
                b["beat"] = t - b["start"] + 1
        for seg in segs.values():
            for k, b in seg["banks"].items():
                s = seen[k]
                s[0] += 1
                if b["beat"] is not None:
                    s[1] += 1
                    s[2] += b["beat"]
            if 1 in seg["banks"]:
                tgt = first if seg["first"] else switch
                tgt[0] += 1
                tgt[1] += seg["banks"][1]["beat"] is not None
    out = {}
    for k, (n, b, tt) in seen.items():
        out[f"beatk_frac_{k}"] = b / n
        out[f"beatk_n_{k}"] = n
        out[f"beatk_turns_{k}"] = tt / b if b else float("nan")
    if first[0]:
        out["beat1_first_frac"] = first[1] / first[0]
    if switch[0]:
        out["beat1_switch_frac"] = switch[1] / switch[0]
    return out


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


def other_negatives(rng, texts: dict, others: list, near: list, n: int, frac: float) -> list:
    """n other-concept negatives for a refit fork: round(n * frac) from `near` (same-family
    concepts, adaptive.family_neg_frac) and the rest from all `others`; a concept is drawn first,
    then one of its texts. frac 0 / no near concepts = the original draws."""

    n_near = round(n * frac) if near else 0
    return ([rng.choice(texts[rng.choice(near)]) for _ in range(n_near)]
            + [rng.choice(texts[rng.choice(others)]) for _ in range(n - n_near)])


def lr_factor(cfg: SupervisedConfig, k: int, offset: int = 0, warm_up: bool = True) -> float:
    """LR multiplier (of cfg.lr) at local step k of a chunk starting at global step `offset`.
    Every chunk re-warms linearly over warmup_steps (fresh optimizer state); with lr_peak the
    target is cosine from lr_peak (global step 0) down to lr (global step lr_decay_steps)."""

    import math

    warm = min(1.0, (k + 1) / max(1, cfg.warmup_steps)) if warm_up else 1.0
    target = cfg.lr
    if cfg.lr_peak > 0 and cfg.lr_decay_steps > 0:
        g = min(1.0, (offset + k) / cfg.lr_decay_steps)
        target = cfg.lr + (cfg.lr_peak - cfg.lr) * 0.5 * (1 + math.cos(math.pi * g))
    cd = cfg.lr_cooldown_steps
    if cd > 0:
        start = offset + cfg.steps - cd  # global step where the cooldown begins
        h = min(1.0, max(0.0, (offset + k - start) / cd))
        target = cfg.lr_final + (target - cfg.lr_final) * 0.5 * (1 + math.cos(math.pi * h))
    return warm * target / cfg.lr


def _refit_fit(job: dict):
    """The fit stage of AdaptiveSupervisedTrainer._episode_refit: (probe or None, info) from a job
    of _refit_reads (+ "name", "excl"). Reads no trainer state, so adaptive.refit_overlap can run
    it on a worker thread."""

    if job["kind"] in DETECTOR_KINDS:  # generic hiding detectors (tica/evdet_defender.py)
        mon, info = fit_evdet(job["evdet_buf"], job["L"], job["concept"], job["r_pol"],
                              job["r_base"], job["r_cal"], job["acfg"], name=job["name"],
                              kind=job["kind"], mis_rows=job["r_mis"])
        return mon, {**info, "kind": job["full_kind"], "sec_read": job["sec_read"]}
    kind, L, Ls, L0, picks = job["kind"], job["L"], job["Ls"], job["L0"], job["picks"]
    rl, ft, npl, nt = job["rl"], job["ft"], job["npl"], job["nt"]
    acfg, name, concept = job["acfg"], job["name"], job["concept"]
    full_kind, sec_read = job["full_kind"], job["sec_read"]
    # adaptive.refit_np_in_fit (10-21k): False -> no-prompt texts are not fit positives (in-context
    # forks only); their gate half (1::2) is still scored as tpr_np
    np_fit = slice(0, None, 2) if acfg.refit_np_in_fit else slice(0, 0)
    if kind != "linear":  # literature probe bank (probe_zoo), same gate
        if kind in ZOO_TOKENS_ENS:
            pos_q = [q for q, (k_, _) in enumerate(picks) if k_ == "pos"]
            neg_q = [q for q, (k_, _) in enumerate(picks) if k_ == "neg"]
            P = {Lk: [ft[Lk][q] for q in pos_q] + nt[Lk][np_fit] for Lk in Ls}
            N = {Lk: [ft[Lk][q] for q in neg_q] for Lk in Ls}
            G = {Lk: nt[Lk][1::2] for Lk in Ls}
        elif kind in ZOO_ENS:
            P = {Lk: torch.cat([rl[Lk][q : q + 1] for q, (k_, _) in enumerate(picks)
                                if k_ == "pos"] + [npl[Lk][np_fit]]) for Lk in Ls}
            N = {Lk: torch.stack([rl[Lk][q] for q, (k_, _) in enumerate(picks) if k_ == "neg"])
                 for Lk in Ls}
            G = {Lk: npl[Lk][1::2] for Lk in Ls}
        elif ft is None:  # pooled single-layer zoo kinds (meandiff): rows, not tokens
            P = torch.cat([rl[L0][q : q + 1] for q, (k_, _) in enumerate(picks)
                           if k_ == "pos"] + [npl[L0][np_fit]])
            N = torch.stack([rl[L0][q] for q, (k_, _) in enumerate(picks) if k_ == "neg"])
            G = npl[L0][1::2]
        else:
            P, N, G = _tok_fit_inputs(job)
        mon, info = gate_fit(kind, P, N, G, acfg, name=name, layer=L, concept=concept,
                             device=job["device"])
        return mon, {**info, "kind": full_kind, "sec_read": sec_read}
    rd = RefitData(layers=[L], concept=concept)
    for (k, _), v in zip(picks, job["r"], strict=True):
        (rd.pos if k == "pos" else rd.neg)[L].append(v[None])
    for j, v in enumerate(job["npr"]):
        if j % 2 == 1:
            rd.np_pos[L].append(v[None])
        elif acfg.refit_np_in_fit:
            rd.pos[L].append(v[None])
    bank, info = fit_refit_bank(rd, concept, name, acfg, exclude_layers=job["excl"])
    return bank, {**info, "kind": full_kind, "sec_read": sec_read}


_RNG_CANARY = 0x7C3A5EED  # refit_overlap: marks "this fit did not touch the global torch RNG"


def _rng_get():
    cu = (torch.cuda.get_rng_state_all()
          if torch.cuda.is_available() and torch.cuda.is_initialized() else None)
    return torch.get_rng_state(), cu


def _rng_set(st) -> None:
    torch.set_rng_state(st[0])
    if st[1] is not None:
        torch.cuda.set_rng_state_all(st[1])


def _rng_eq(a, b) -> bool:
    if not torch.equal(a[0], b[0]) or (a[1] is None) != (b[1] is None):
        return False
    return a[1] is None or all(torch.equal(x, y) for x, y in zip(a[1], b[1], strict=True))


_SIDE = threading.local()  # refit_overlap_stream: one side CUDA stream per worker thread


def _iso_fit(job: dict, grad: bool, ready=None):
    """adaptive.refit_overlap's worker: _refit_fit(job) with the caller's grad mode and device,
    the global torch RNG isolated. Fits seed themselves from the probe name (torch.manual_seed +
    module init), so their results do not depend on the state they start from; the state they
    END in does, and the main thread restores the serial one at the join. Returns ((probe, info),
    RNG end state or None if the fit left the RNG untouched, fit seconds).
    ready (adaptive.refit_overlap_stream, 10-23): a CUDA event recorded on the main stream after the
    job's reads; the fit then runs on this thread's SIDE stream (after the event), so its many small
    kernels overlap the main thread's next reads instead of queueing behind them on one stream. The
    side stream is synchronised before returning, so the probe is complete when the future is."""

    t0 = time.time()
    dev = job["device"]
    if isinstance(dev, torch.device) and dev.type == "cuda":
        torch.cuda.set_device(dev)
    before = _rng_get()
    torch.default_generator.manual_seed(_RNG_CANARY)
    if before[1] is not None:
        torch.cuda.manual_seed_all(_RNG_CANARY)
    canary = _rng_get()
    try:
        with torch.set_grad_enabled(grad):
            if ready is not None:
                side = getattr(_SIDE, "s", None)
                if side is None or side.device != ready_dev(dev):
                    side = _SIDE.s = torch.cuda.Stream(device=ready_dev(dev))
                side.wait_event(ready)
                with torch.cuda.stream(side):
                    out = _refit_fit(job)
                side.synchronize()
            else:
                out = _refit_fit(job)
        after = _rng_get()
    finally:
        _rng_set(before)
    return out, (None if _rng_eq(after, canary) else after), time.time() - t0


def _tok_fit_inputs(job: dict):
    """(P, N, G) of a single-layer token-kind refit job (_refit_fit's zoo inputs)."""

    L, picks, ft, nt, acfg = job["L"], job["picks"], job["ft"], job["nt"], job["acfg"]
    np_fit = slice(0, None, 2) if acfg.refit_np_in_fit else slice(0, 0)
    P = [ft[L][q] for q, (k_, _) in enumerate(picks) if k_ == "pos"] + nt[L][np_fit]
    N = [ft[L][q] for q, (k_, _) in enumerate(picks) if k_ == "neg"]
    return P, N, nt[L][1::2]


def _stackable(job: dict) -> bool:
    """A refit job adaptive.refit_stack_episodes can stack: an rlobf-kind token fit."""

    from next_chameleons.tica.probe_zoo import RLOBF_KINDS

    return (job["kind"] in RLOBF_KINDS and job.get("ft") is not None
            and getattr(job["acfg"], "refit_batched_rlobf", False))


def _iso_fit_many(jobs: list, grad: bool) -> list:
    """_iso_fit for several stackable jobs of ONE kind: one stacked training
    (probe_zoo.gate_fit_many), the global torch RNG isolated as in _iso_fit. Every job gets the
    RNG end state of the whole stack (= the last job's, whose fit runs last: fits seed
    themselves from their names, so only the last one matters at the join)."""

    from next_chameleons.tica.probe_zoo import gate_fit_many

    t0 = time.time()
    dev = jobs[0]["device"]
    if isinstance(dev, torch.device) and dev.type == "cuda":
        torch.cuda.set_device(dev)
    before = _rng_get()
    torch.default_generator.manual_seed(_RNG_CANARY)
    if before[1] is not None:
        torch.cuda.manual_seed_all(_RNG_CANARY)
    canary = _rng_get()
    try:
        items = []
        for j in jobs:
            P, N, G = _tok_fit_inputs(j)
            items.append({"P": P, "N": N, "NP": G, "name": j["name"], "layer": j["L"],
                          "concept": j["concept"]})
        with torch.set_grad_enabled(grad):
            res = gate_fit_many(jobs[0]["kind"], items, jobs[0]["acfg"], device=dev)
        after = _rng_get()
    finally:
        _rng_set(before)
    rec = None if _rng_eq(after, canary) else after
    dt = (time.time() - t0) / len(jobs)
    return [((mon, {**info, "kind": j["full_kind"], "sec_read": j["sec_read"]}), rec, dt)
            for (mon, info), j in zip(res, jobs, strict=True)]


class _StackFitWorker:
    """adaptive.refit_stack_episodes: refit_overlap's worker thread, but stackable fits (rlobf
    kinds) are HELD until a second one of the same kind is queued or the main thread asks for a
    result (flush(), called before every wait), then a kind's held fits train as one stacked run;
    other fits run at once, in queue order, while stackable ones wait. Same submit / Future
    interface as the ThreadPoolExecutor it replaces (cancel works while a fit is queued / held)."""

    def __init__(self, max_stack: int = 8):
        from concurrent.futures import Future

        self._Future, self._q, self._max = Future, queue.Queue(), max_stack
        self._flush = threading.Event()
        self.n_stacked = 0  # fits that trained inside a stack of >= 2 (logged: ep_refit_stacked)
        threading.Thread(target=self._loop, name="tica-refit-stack", daemon=True).start()

    def submit(self, fn, job, grad, ready):
        fut = self._Future()
        self._q.put((fn, job, grad, ready, fut))
        return fut

    def flush(self):
        """Run every held stackable fit now (the main thread is about to wait on a result)."""

        self._flush.set()
        self._q.put(None)  # wake the worker

    @staticmethod
    def _run_one(it):
        if it[4].set_running_or_notify_cancel():
            try:
                it[4].set_result(it[0](it[1], it[2], it[3]))
            except BaseException as e:
                it[4].set_exception(e)

    def _run_group(self, g):
        g = [it for it in g if it[4].set_running_or_notify_cancel()]
        if not g:
            return
        if len(g) == 1:
            try:
                g[0][4].set_result(g[0][0](g[0][1], g[0][2], g[0][3]))
            except BaseException as e:
                g[0][4].set_exception(e)
            return
        try:
            outs = _iso_fit_many([it[1] for it in g], g[0][2])
            self.n_stacked += len(g)
            for it, o in zip(g, outs, strict=True):
                it[4].set_result(o)
        except BaseException as e:
            for it in g:
                it[4].set_exception(e)

    def _loop(self):
        held: dict = {}
        while True:
            items = [self._q.get()]
            while True:
                try:
                    items.append(self._q.get_nowait())
                except queue.Empty:
                    break
            for it in items:
                if it is None:
                    continue
                if it[3] is None and _stackable(it[1]):
                    held.setdefault(it[1]["kind"], []).append(it)
                else:
                    self._run_one(it)
            flush = self._flush.is_set()
            for kind in list(held):
                if flush or len(held[kind]) >= self._max:
                    self._run_group(held.pop(kind))
            if flush and not held:
                self._flush.clear()


def _cpu_fit_init(threads: int, ens_layers: dict, tok_ens_layers: dict) -> None:
    """adaptive.refit_cpu_fits worker: torch threads per CPU fit process + the main process's
    multi-layer kind tables (probe_zoo.set_zoo_depth rescales them for model.depth in the main
    process only; a spawned worker would otherwise use the 32-layer originals)."""

    from next_chameleons.tica import probe_zoo

    torch.set_num_threads(max(1, int(threads)))
    probe_zoo.ENS_LAYERS.clear()
    probe_zoo.ENS_LAYERS.update(ens_layers)
    probe_zoo.TOKENS_ENS_LAYERS.clear()
    probe_zoo.TOKENS_ENS_LAYERS.update(tok_ens_layers)


def _cpu_fit_ok(job: dict) -> bool:
    """A refit job whose fit can run in a CPU worker: pooled kinds (no per-token states, no
    detector buffer)."""

    return (job["kind"] not in DETECTOR_KINDS and job.get("ft") is None
            and job.get("nt") is None)


def _job_to_cpu(job: dict) -> dict:
    """A copy of a pooled refit job with every tensor on the CPU and device = cpu."""

    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: cpu(v) for k, v in x.items()}
        if isinstance(x, list):
            return [cpu(v) for v in x]
        return x

    return {**{k: cpu(v) for k, v in job.items()}, "device": torch.device("cpu")}


def ready_dev(dev) -> torch.device:
    return dev if isinstance(dev, torch.device) and dev.type == "cuda" else torch.device(
        "cuda", torch.cuda.current_device())


def _probe_tensors(obj, depth: int = 0):
    """Every tensor of a probe (modules, attributes, nested monitors), on CPU (TICA_DUMP_REFITS)."""

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, torch.nn.Module):
        return {k: v.detach().cpu().clone() for k, v in obj.state_dict().items()}
    if depth > 4:
        return None
    if isinstance(obj, dict):
        return {str(k): _probe_tensors(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_probe_tensors(v, depth + 1) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _probe_tensors(v, depth + 1) for k, v in vars(obj).items()}
    return obj if isinstance(obj, (int, float, str, bool, type(None))) else None


def _same_reads(a: dict, b: dict) -> bool:
    """Two _refit_reads jobs carry bit-identical activations (fit inputs)."""

    def same(x, y):
        if isinstance(x, torch.Tensor):
            return (isinstance(y, torch.Tensor) and x.shape == y.shape and x.dtype == y.dtype
                    and bool(torch.equal(x, y)))
        if isinstance(x, dict):
            return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k])
                                                                         for k in x)
        if isinstance(x, (list, tuple)):
            return (isinstance(y, (list, tuple)) and len(x) == len(y)
                    and all(same(u, v) for u, v in zip(x, y, strict=True)))
        return x == y

    return all(same(a[k], b[k]) for k in ("r", "rl", "ft", "npr", "npl", "nt"))


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

    world, rank = 1, 0  # data-parallel world size / rank (set in __init__ from tica/dp.py)
    _ep_ctx: dict | None = None  # per-rollout episode streams (episode_rng / data-parallel)

    def __init__(self, policy, pool: MonitorPool, sampler, cfg: AdaptiveSupervisedConfig, out_dir,
                 branch_concepts=None, branch_negatives=None, roots_only=None, prompts=None,
                 inputs=None, hard_neg=None, resume_state=None, cap_free_pool=None):
        self.policy, self.pool, self.sampler, self.cfg = policy, pool, sampler, cfg
        # trainer_state.pt of the checkpoint this chunk resumes from (SupervisedConfig.save_state)
        self._resume = (Path(resume_state) if resume_state and Path(resume_state).exists()
                        else None)
        # (opening, render_feedback): repeat format by default; tica/answer_format for "answer".
        self.opening, self.render_feedback = prompts or (opening, render_feedback)
        self.inputs = inputs  # answer format {text_id: request} (capability forks strip it)
        # adaptive.cap_free_*: {concept: [(request, fallback, reference)]} (scripts/tica.py)
        self.cap_free_pool = cap_free_pool or {}
        # {concept: matched within-source negatives}: half of a refit fork's other-concept
        # negatives come from these (e.g. DolusChat truthful twins for deception)
        self.hard_neg = hard_neg or {}
        self.falsify = False  # eval-only control (tica_switch_eval --falsify)
        # eval-only (10-27, tica_switch_eval --refit-every): refit after exactly these turn indices,
        # whatever the verdict (refit level = a fixed function of the turn, not of the model's
        # success); None = the normal rule (refit at the turn after a pass)
        self.forced_refit_turns: set[int] | None = None
        self.deploy_log: list[dict] = []  # turns-to-beat records (rollout; beat_summary)
        self._cap_mult, self._cap_acc_ema = 1.0, None  # adaptive capability weight (off: 1.0)
        # adaptive.cap_level_adapt: per refit-level bin (1..5+ probes) [EMA, samples seen] per kind
        self._cap_lvl = [{"gsm": [0.0, 0], "free": [0.0, 0]} for _ in range(5)]
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
                {c: (fams.get(c, "topic") if ac.concept_hierarchy else "all") for c in concepts},
                family_kw={"cap_rel": ac.family_cap_rel, "power": ac.family_power},
                **{**self._tts_kw(), "cap_rel": ac.concept_cap_rel, "power": ac.concept_power})
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
        warm_up = self._resume is None  # restored Adam moments: no re-warmup
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda k: lr_factor(cfg, k, off, warm_up))
        self.step_k = 0
        self.refit_k = 0
        self.refit_buf: RefitData | None = None
        self._fresh_escapes = 0  # escapes added to the buffer since the last refit (data-triggered)
        self._latest_refit: str | None = None  # newest online refit name (refit_as_start)
        self._transient: list[str] = []  # episode-local refits (refit_mode="episode")
        self._bank_validated = False
        self.base_names = {e.monitor.name for e in pool.entries}
        self.rng = random.Random(cfg.seed + off)  # + off: resume chunks draw afresh
        # Data-parallel episodes (tica/dp.py; world size from the launcher, 1 = off): per-episode
        # RNG streams are implied; the features whose state couples episodes INSIDE a rollout are
        # not supported (their state would differ between ranks).
        self.world, self.rank = dp.world()
        if self.world > 1 and cfg.dp_sync_init:
            dp.broadcast_params(self.params)
        if self.world > 1:
            ac_ = cfg.adaptive
            bad = {"refit_mode != 'episode'": ac_.refit_mode != "episode",
                   "episode_keep": ac_.episode_keep, "episode_bank_first": ac_.episode_bank_first,
                   "adaptive (best-catcher switch)": ac_.adaptive, "exo_prob": ac_.exo_prob > 0,
                   "episode_kind_sampling='prioritized'":
                       ac_.episode_kind_sampling == "prioritized",
                   "episodes_per_step not divisible by the world size":
                       cfg.episodes_per_step % self.world != 0}
            if any(bad.values()):
                raise ValueError("data-parallel episodes do not support: "
                                 + ", ".join(k for k, v in bad.items() if v))
        # Pursuit (adaptive.pursuit): running centroid at the concept layer + per-step (i,t) acts;
        # one PursuitState per monitored concept (multi-concept: the clouds differ).
        self.pursuit_by: dict[str, PursuitState] = {}
        self._ep_acts: dict[tuple[int, int], torch.Tensor] = {}
        self._ep_concept: dict[tuple[int, int], str] = {}
        # Standalone stay (adaptive.stay_standalone, 10-21x): per (episode, turn) rollout reads at
        # the stay layers [n_layers, d] (detached targets) + a running scale per (concept, layer).
        self._ep_stay: dict[tuple[int, int], torch.Tensor] = {}
        self.stay_by: dict[tuple[str, int], PursuitState] = {}
        self._stay_L: list[int] = []
        if cfg.adaptive.stay_standalone and cfg.adaptive.pursuit and (
                cfg.adaptive.motion == "conditional"):
            raise ValueError("adaptive.stay_standalone with pursuit + motion='conditional' would "
                             "add the stay penalty twice; turn one off")
        # Probe-kind curriculum (adaptive.episode_kind_sampling): refit name -> kind, first judged
        # turn per (traj, name), and per-kind [caught, judged] on post-first turns (all training).
        self._kind_of: dict[str, str] = {}
        self._kind_first: dict[tuple, int] = {}
        self.kind_stats: dict[str, list[int]] = {}
        torch.manual_seed(cfg.seed + off)
        if self.rank == 0:
            (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        if self._resume is not None:
            self._load_state(self._resume)

    def _cap_adapt(self, acc: float) -> dict:
        """Adaptive capability weight (adaptive.cap_adapt_*): update the multiplier from this
        step's GSM8K-while-hiding accuracy; it applies from the next step's losses on."""

        ac = self.cfg.adaptive
        e = self._cap_acc_ema
        e = acc if e is None else ac.cap_adapt_ema * e + (1 - ac.cap_adapt_ema) * acc
        self._cap_acc_ema = e
        m = self._cap_mult * math.exp(ac.cap_adapt_rate * (ac.cap_adapt_target - e))
        self._cap_mult = min(ac.cap_adapt_max, max(ac.cap_adapt_min, m))
        return {"cap_adapt/mult": round(self._cap_mult, 4), "cap_adapt/acc_ema": round(e, 4)}

    def _state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "kind_bandit": self.kind_bandit,
                "concept_bandit": self.concept_bandit, "rng": self.rng.getstate(),
                "sampler_rng": (self.sampler.rng.getstate()
                                if hasattr(self.sampler, "rng") else None),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available()
                             else None),
                "global_step": self.step_k + self._step_offset(),
                "cap_adapt": (self._cap_mult, self._cap_acc_ema), "cap_level": self._cap_lvl}

    def _save_state(self, ckpt: Path) -> None:
        """trainer_state.pt next to the adapter; older checkpoints' copies are removed."""

        if not self.cfg.save_state or self.rank != 0:  # data-parallel: rank 0 (states agree)
            return
        for old in ([] if self.cfg.save_state_all else self.out.glob("*/trainer_state.pt")):
            old.unlink(missing_ok=True)
        tmp = ckpt / "trainer_state.pt.tmp"
        torch.save(self._state_dict(), tmp)
        tmp.replace(ckpt / "trainer_state.pt")
        if self.cfg.save_state_all and self.cfg.state_keep_last > 0:
            self._prune_states()

    def _prune_states(self) -> None:
        """SupervisedConfig.state_keep_last / state_keep_every: drop older checkpoints' optimizer
        states in this chunk (adapters stay)."""

        every, off = self.cfg.state_keep_every, self._step_offset()
        have = sorted((int(p.parent.name[5:]), p) for p in self.out.glob("step_*/trainer_state.pt")
                      if p.parent.name[5:].isdigit())
        for k, p in have[: -self.cfg.state_keep_last]:
            if not (every and (off + k) % every == 0):
                p.unlink(missing_ok=True)

    def _load_state(self, path: Path) -> None:
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.opt.load_state_dict(st["opt"])
        if st.get("kind_bandit") is not None and self.kind_bandit is not None:
            self.kind_bandit = st["kind_bandit"]
        if st.get("concept_bandit") is not None and self.concept_bandit is not None:
            self.concept_bandit = st["concept_bandit"]
        self.rng.setstate(st["rng"])
        if st.get("cap_adapt") is not None and self.cfg.adaptive.cap_adapt_target > 0:
            self._cap_mult, self._cap_acc_ema = st["cap_adapt"]
        if st.get("cap_level") is not None:
            self._cap_lvl = st["cap_level"]
        if st.get("sampler_rng") is not None and hasattr(self.sampler, "rng"):
            self.sampler.rng.setstate(st["sampler_rng"])
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        print(json.dumps({"resumed_state": str(path), "global_step": st.get("global_step")}),
              flush=True)

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

    # ---- episode streams / data-parallel episodes (adaptive.episode_rng, tica/dp.py) ----------

    def _episode_streams(self, specs):
        """Per-rollout context for adaptive.episode_rng (always on in data-parallel runs), or None
        (the original draws). Drawn from the trainer RNG in episode order, identically on every
        rank: one stream seed per episode, the turn-0 shared-probe seed and the monitor-off turns;
        this rank keeps episodes rank, rank + world, ... (`specs` / `gidx`)."""

        acfg = self.cfg.adaptive
        if not (acfg.episode_rng or self.world > 1):
            self._ep_ctx = None
            return None
        T = self.cfg.turns
        seeds = [self.rng.randrange(1 << 62) for _ in specs]
        t0_seed = self.rng.randrange(1 << 62)
        off = [[s.cond in DODGE_CONDS and t >= 1 and self.rng.random() < acfg.off_turn_prob
                for t in range(T)] for s in specs]
        gidx = list(range(self.rank, len(specs), self.world))
        self._ep_ctx = {
            "gspecs": list(specs), "gidx": gidx, "specs": [specs[g] for g in gidx],
            "erng": [random.Random(seeds[g]) for g in gidx], "off": [off[g] for g in gidx],
            "n": [0] * len(gidx), "t0_seed": t0_seed, "t0_uses": defaultdict(int), "t0_fit": {},
            "step": self.step_k + self._step_offset(),
            "kind_stats0": {k: list(v) for k, v in self.kind_stats.items()},
            "kind_of0": set(self._kind_of)}
        return self._ep_ctx

    def _erng(self, i: int) -> random.Random:
        """Episode i's stream (episode streams), else the trainer RNG (the original draws)."""
        return self._ep_ctx["erng"][i] if self._ep_ctx is not None else self.rng

    def _ep_tag(self, i: int) -> dict:
        """Turns-to-beat records carry their episode under episode streams (merge order)."""
        return {"_ep": i} if self._ep_ctx is not None else {}

    def _ep_kw(self, i: int) -> dict:
        """_episode_refit kwargs under episode streams: the episode's stream and the next
        deterministic probe name (names seed the probe fits)."""

        ctx = self._ep_ctx
        if ctx is None:
            return {}
        ctx["n"][i] += 1
        return {"rng": ctx["erng"][i],
                "name": f"ep_refit_s{ctx['step']}_e{ctx['gidx'][i]}_{ctx['n'][i]}"}

    @staticmethod
    def _slim(tr: Trajectory) -> Trajectory:
        """For the all-gather: convs[t] is the final conversation's prefix through turn t."""
        return Trajectory(tr.spec, [tr.convs[-1]] if tr.convs else [], tr.results, tr.feedback)

    @staticmethod
    def _unslim(tr: Trajectory) -> Trajectory:
        last = tr.convs[0] if tr.convs else []
        return Trajectory(tr.spec, [[dict(m) for m in last[: 2 * t + 2]]
                                    for t in range(len(tr.results))], tr.results, tr.feedback)

    def _merge_rollout(self, trajs, passed_turn, ep_stats, pending, seg_pending, last_refit,
                       deploy_start, seg_start):
        """Every rank's rollout records -> the step's records in episode order, applied
        identically on every rank (one rank: the same records, canonically ordered). Returns
        (all trajectories, passes, ep_stats, censored refits); sets the merged deploy / segment /
        refit logs, kind_stats / kind map, the pursuit acts (_g_acts, ordered as one process
        inserts them: turn, then episode) and ctx["bank_size"]."""

        ctx = self._ep_ctx
        gidx, multi = ctx["gidx"], self.world > 1

        def tagged(log, start, open_):
            recs = log[start:] + [{**v, "beaten": False} for v in open_.values()]
            del log[start:]
            return [{**r, "_ep": gidx[r["_ep"]]} for r in recs]

        k0 = ctx["kind_stats0"]
        part = {
            "gidx": gidx,
            "trajs": [self._slim(tr) for tr in trajs] if multi else trajs,
            "passed": passed_turn, "stats": ep_stats, "cens": len(last_refit),
            "dep": tagged(self.deploy_log, deploy_start, pending),
            "seg": tagged(self.seg_log, seg_start, seg_pending),
            "refits": [{**r, "episode": gidx[r["episode"]]} for r in self._refit_log],
            "kind_delta": {k: [v[0] - k0.get(k, [0, 0])[0], v[1] - k0.get(k, [0, 0])[1]]
                           for k, v in self.kind_stats.items()},
            "kind_of": {n: k for n, k in self._kind_of.items() if n not in ctx["kind_of0"]},
            "acts": {(gidx[i], t): (a.cpu() if multi else a)
                     for (i, t), a in self._ep_acts.items()},
            "acts_c": {(gidx[i], t): c for (i, t), c in self._ep_concept.items()},
            "names": [e.monitor.name for e in self.pool.entries],
            **({"stay": {(gidx[i], t): (a.cpu() if multi else a)
                         for (i, t), a in self._ep_stay.items()}} if self._ep_stay else {}),
            "t0_fit": dict(ctx["t0_fit"]), "t0_uses": dict(ctx["t0_uses"]),
            "read_sec": getattr(self, "_refit_read_sec", 0.0)}
        parts = dp.gather(part) if multi else [part]
        g_trajs: list = [None] * len(ctx["gspecs"])
        for pt in parts:
            for g, tr in zip(pt["gidx"], pt["trajs"], strict=True):
                g_trajs[g] = self._unslim(tr) if multi else tr
        stats: dict = {}
        for pt in parts:
            for k, v in pt["stats"].items():
                stats[k] = stats.get(k, [] if isinstance(v, list) else 0) + v
        # turns-to-beat records: episode order (stable: each episode's own order kept)
        for key, log in (("dep", self.deploy_log), ("seg", self.seg_log)):
            recs = sorted((r for pt in parts for r in pt[key]), key=lambda r: r["_ep"])
            log += [{k: v for k, v in r.items() if k != "_ep"} for r in recs]
        refits, seen_t0 = [], set()
        for r in sorted((r for pt in parts for r in pt["refits"]),
                        key=lambda r: (r["episode"], r["turn"], r["attempt"])):
            if "t0_key" in r:  # a shared turn-0 probe: one fit (several ranks may have fit it)
                if r["t0_key"] in seen_t0:
                    continue
                seen_t0.add(r["t0_key"])
                r = {k: v for k, v in r.items() if k != "t0_key"}
            refits.append(r)
        self._refit_log = refits
        ks: dict = {k: list(v) for k, v in k0.items()}
        for pt in parts:
            for k, (a, b) in pt["kind_delta"].items():
                cur = ks.setdefault(k, [0, 0])
                cur[0], cur[1] = cur[0] + a, cur[1] + b
        self.kind_stats = ks
        for pt in parts:
            self._kind_of.update(pt["kind_of"])
        acts = {k: v for pt in parts for k, v in pt["acts"].items()}
        conc = {k: v for pt in parts for k, v in pt["acts_c"].items()}
        order = sorted(acts, key=lambda k: (k[1], k[0]))
        self._g_acts = {k: acts[k] for k in order}
        self._g_concept = {k: conc[k] for k in order}
        if self.cfg.adaptive.stay_standalone:
            stay = {k: v for pt in parts for k, v in pt.get("stay", {}).items()}
            self._g_stay = {k: stay[k] for k in order if k in stay}
        ctx["bank_size"] = len({n for pt in parts for n in pt["names"]})
        # shared turn-0 probes: one fit per key, the other episodes opening on it share it
        t0_fit = {k: v for pt in parts for k, v in pt["t0_fit"].items()}
        uses: dict = defaultdict(int)
        for pt in parts:
            for k, v in pt["t0_uses"].items():
                uses[k] += v
        stats["first_att"] = stats.get("first_att", 0) + len(t0_fit)
        stats["first_acc"] = stats.get("first_acc", 0) + sum(v is not None for v in t0_fit.values())
        stats["first_shared"] = sum(uses[k] - 1 for k, v in t0_fit.items() if v is not None)
        self._refit_read_sec = sum(pt["read_sec"] for pt in parts)
        passed = [t for pt in parts for t in pt["passed"]]
        return g_trajs, passed, stats, sum(pt["cens"] for pt in parts)

    @torch.no_grad()
    def rollout(self, specs):
        """Teacher-forced lockstep rollout with a per-episode defender. Returns trajectories whose
        results[t] carry the ACTIVE monitor's read and name, plus metrics."""

        acfg, pol, pool = self.cfg.adaptive, self.policy, self.pool
        T = self.cfg.turns
        # episode_rng / data-parallel: per-episode streams; this rank runs its share of the specs
        ctx = self._episode_streams(specs)
        spec0 = specs[0]  # the run's first spec (refit buffer concept, pursuit layer)
        if ctx is not None:
            specs = ctx["specs"]
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

        def switch_refits(i, s, prefix, shown, c, t, bank, info, tried, refit=None):
            # refit_switch_max (10-15g): a refit that fails its quality gate is NOT replaced by
            # the old (already dodged) probe; another probe TYPE is fit instead (same sample
            # sizes), up to refit_switch_max more types.
            refit = refit or self._episode_refit
            while bank is None and len(tried) <= acfg.refit_switch_max:
                k2 = self._pick_kind(exclude=tried, rng=self._erng(i))
                if k2 is None:
                    break
                t_sw = time.time()
                bank, info = refit(s, prefix, shown, c, t + 1, kind=k2, **self._ep_kw(i))
                self._log_refit(info, i, t + 1, c, len(tried), time.time() - t_sw)
                tried.append(k2)
                ep_stats["switch_att"] += 1
            return bank, info

        def finish_refit(i, c, t, dfd, deployed_i, res, bank, info, tried):
            if bank is None and len(tried) > 1:
                ep_stats["all_failed"] += 1
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
                dump = os.environ.get("TICA_DUMP_REFITS")  # debug: every accepted refit's tensors
                if dump:
                    Path(dump).mkdir(parents=True, exist_ok=True)
                    torch.save(_probe_tensors(bank), Path(dump) / f"{bank.name}.pt")
                dfd.members.append(bank)
                dfd.by_name[bank.name] = bank
                dfd.active, dfd.stage_turn = bank.name, 0
                deployed_i.append(bank.name)
                self._kind_of[bank.name] = info.get("kind", "linear")
                ep_stats["acc"] += 1
                ep_stats["auroc"].append(info["auroc"])
                last_refit[(i, c)] = t
                pending[(i, c)] = {"kind": info.get("kind", "linear"),
                                   "turns": 0, "concept": c, **self._ep_tag(i)}

        # adaptive.refit_overlap (10-21e): this turn's refits read on the main thread and fit on
        # ONE worker thread (FIFO = the serial fit order); ovl_join finishes them in episode order
        # (switch fallback, deploy) before anything uses them. Exactness: see _iso_fit / ovl_join.
        overlap = (ep_mode and acfg.refit_overlap and ctx is not None
                   and not (acfg.episode_refit_kind == "mixed" and acfg.refit_switch_max > 0
                            and acfg.episode_kind_sampling == "prioritized"))
        ovl: list[dict] = []  # this turn's pending refits (episode order)
        ovl_recs: list = []  # RNG end states of the turn's effective fits, serial order
        if overlap:
            ep_stats.update({"ovl_main": 0.0, "ovl_work": 0.0, "ovl_wait": 0.0, "ovl_waste": 0.0,
                             "ovl_redo": 0, "ovl_redo_same": 0})
            if getattr(self, "_ovl_pool", None) is None:
                self._ovl_pool = (_StackFitWorker() if acfg.refit_stack_episodes
                                  else ThreadPoolExecutor(max_workers=1,
                                                          thread_name_prefix="tica-refit-fit"))
            if acfg.refit_cpu_fits and getattr(self, "_cpu_pool", None) is None:
                import multiprocessing as mp

                from next_chameleons.tica import probe_zoo

                self._cpu_pool = ProcessPoolExecutor(
                    max_workers=acfg.refit_cpu_fits, mp_context=mp.get_context("spawn"),
                    initializer=_cpu_fit_init,
                    initargs=(acfg.refit_cpu_threads, dict(probe_zoo.ENS_LAYERS),
                              dict(probe_zoo.TOKENS_ENS_LAYERS)))

        side = overlap and acfg.refit_overlap_stream and torch.cuda.is_available()

        def ovl_fit(job):
            cpu_pool = getattr(self, "_cpu_pool", None) if acfg.refit_cpu_fits else None
            if cpu_pool is not None and _cpu_fit_ok(job):  # pooled kinds: a CPU worker process
                return cpu_pool.submit(_iso_fit, _job_to_cpu(job), torch.is_grad_enabled(), None)
            ready = None
            if side:  # the job's reads are queued on the main stream: the fit waits for them
                ready = torch.cuda.Event()
                ready.record()
            return self._ovl_pool.submit(_iso_fit, job, torch.is_grad_enabled(), ready)

        def np_snap():  # the no-prompt read cache (refit_np_cache) after a read
            cch = getattr(self, "_np_cache", None)
            return (None if cch is None else dict(cch), getattr(self, "_np_ver", None),
                    getattr(self, "_np_hits", 0))

        def np_restore(sn):
            cch, ver, hits = sn
            self._np_cache = {} if cch is None else dict(cch)
            self._np_ver, self._np_hits = ver, hits

        def ovl_submit(i, s, c, t, shown, dfd, deployed_i, res):
            t_ep = time.time()
            kw = self._ep_kw(i)
            prefix = list(convs[i])  # the conversation through the passing answer
            plan = self._refit_plan(s, prefix, shown, c, None, None, kw["rng"])
            job = self._refit_reads(plan)
            job["name"], job["excl"] = kw["name"], self._heldout_layers()
            ovl.append({"i": i, "s": s, "c": c, "t": t, "shown": shown, "dfd": dfd,
                        "dep": deployed_i, "res": res, "prefix": prefix, "plan": plan,
                        "job": job, "snap": np_snap(), "fut": ovl_fit(job)})
            ep_stats["ovl_main"] += time.time() - t_ep

        def ovl_refit(s, prefix, shown, c, turn, kind=None, rng=None, name=None):
            # a switch attempt at the join: the serial reads, the fit on the worker (FIFO)
            plan = self._refit_plan(s, prefix, shown, c, kind, None, rng)
            job = self._refit_reads(plan)
            job["name"], job["excl"] = name, self._heldout_layers()
            fut = ovl_fit(job)
            if hasattr(self._ovl_pool, "flush"):
                self._ovl_pool.flush()
            out, rec, _ = fut.result()
            ovl_recs.append(rec)
            return out

        def ovl_join():
            """Finish the turn's pending refits in episode order = the serial order. A refit that
            needs the switch fallback rolls the no-prompt read cache back to its own read, runs
            its switch reads + fits, then redoes the later episodes' reads (their serial place);
            a later fit is kept when its redone reads are bit-identical, else refit. The global
            torch RNG ends in the state the serial fits leave."""

            if not ovl:
                return
            t_j = time.time()
            p = 0
            while p < len(ovl):
                e = ovl[p]
                i, c, t = e["i"], e["c"], e["t"]
                tw = time.time()
                if hasattr(self._ovl_pool, "flush"):
                    self._ovl_pool.flush()
                (bank, info), rec, fsec = e["fut"].result()
                ep_stats["ovl_wait"] += time.time() - tw
                excl = self._heldout_layers()
                if e["job"]["kind"] == "linear" and excl != e["job"]["excl"]:
                    e["job"]["excl"] = excl  # an earlier deploy changed it: fit as serial would
                    fut = ovl_fit(e["job"])
                    if hasattr(self._ovl_pool, "flush"):
                        self._ovl_pool.flush()
                    (bank, info), rec, fsec = fut.result()
                info = {**info, "sec_read": e["job"]["sec_read"]}
                sec = e["job"]["read_dt"] + fsec
                self._log_refit(info, i, t + 1, c, 0, sec)
                ovl_recs.append(rec)
                tried = [info.get("kind")]
                if bank is None and len(tried) <= acfg.refit_switch_max:
                    later = ovl[p + 1 :]
                    for f in later:
                        f["fut"].cancel()
                    np_restore(e["snap"])  # the cache as the serial switch reads see it
                    t_sw = time.time()
                    bank, info = switch_refits(i, e["s"], e["prefix"], e["shown"], c, t, bank,
                                               info, tried, refit=ovl_refit)
                    sec += time.time() - t_sw
                    for f in later:  # redo the later reads in their serial place
                        old = f["job"]
                        job = self._refit_reads(f["plan"], t0=time.time())
                        job["name"], job["excl"] = old["name"], old["excl"]
                        self._refit_read_sec -= old["read_dt"]
                        ep_stats["ovl_waste"] += old["read_dt"]
                        ep_stats["ovl_redo"] += 1
                        f["snap"] = np_snap()
                        if _same_reads(old, job):
                            ep_stats["ovl_redo_same"] += 1
                            f["job"] = {**old, "read_dt": job["read_dt"],
                                        "sec_read": job["sec_read"]}
                            if f["fut"].cancelled():
                                f["fut"] = ovl_fit(old)
                        else:
                            f["job"], f["fut"] = job, ovl_fit(job)
                ep_stats["sec"] += sec
                ep_stats["ovl_work"] += sec
                finish_refit(i, c, t, e["dfd"], e["dep"], e["res"], bank, info, tried)
                p += 1
            last = [r for r in ovl_recs if r is not None]
            if last:  # the serial fits' RNG end state (the worker restores its own)
                _rng_set(last[-1])
            ovl.clear()
            ovl_recs.clear()
            ep_stats["ovl_main"] += time.time() - t_j

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
        v5 = getattr(self.render_feedback, "v5", False)  # layout tagged_v5 (10-23)
        v5_last: dict = {}  # (episode, concept, probe) -> score shown at its last judged turn
        v5_fix: list = []  # this turn's v5 messages: "New monitor" re-checked after the refits

        def defender(i, s, c):
            if c not in def_by[i]:
                ovl_join()  # refit_overlap: first probes read + fit after the pending refits
                start = _start(s) if c == s.monitored else pool.concept_monitor(c).name
                # "switch_fork" (10-17h): the episode's FIRST concept keeps its base probe; every
                # newly switched-to concept gets a fork-fit first probe
                fp = self._first_probe_mode()
                fork = fp == "fork" or (fp in ("switch_fork", "base_panel")
                                        and c != s.monitored_at(0))
                if fp == "base_panel" and not fork and ep_mode:
                    # 10-20: turn 0 = ONE fixed base probe drawn from the panel
                    name = self._panel_first(i, s, c, umsg[i], pending, ep_stats)
                    if name is not None:
                        start = name
                elif fork and ep_mode:
                    name = self._fork_first(i, s, c, convs[i], umsg[i], pending, ep_stats)
                    if name is not None:
                        start = name
                elif acfg.first_probe not in ("base", "fork", "switch_fork"):
                    raise ValueError(acfg.first_probe)
                def_by[i][c] = AdaptiveDefender(pool, c, acfg, start=start,
                                                rng=random.Random(self._erng(i).randrange(1 << 30)))
                dep_by[i][c] = [def_by[i][c].active]
            return def_by[i][c]

        dodging = [s.cond in DODGE_CONDS for s in specs]
        if self.refit_buf is None:
            self.refit_buf = RefitData(layers=pool.layers, concept=spec0.monitored)
        # Track the centroid / per-turn acts always (turn_move is reported for the pursuit-OFF
        # control too); the pursuit LOSS term is gated on acfg.pursuit below. One centroid per
        # concept (multi-concept runs): concept clouds sit in different places.
        plp = self._plp = pool.concept_monitor(spec0.monitored).layer
        self._ep_acts, self._ep_concept, self._kind_first = {}, {}, {}
        stay_L = self._stay_L = self._stay_layers(plp) if acfg.stay_standalone else []
        self._ep_stay = {}
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
        off = ctx["off"] if ctx is not None else [
            [d and t >= 1 and self.rng.random() < acfg.off_turn_prob for t in range(T)]
            for d in dodging]
        # Turn budget (final.turn_budget, 10-19): switch points follow the race (_budget_step);
        # tail[i][t] = turns after the episode's last probe (shown "Monitor: OFF"; no loss, no
        # stats). Specs without a budget never change either.
        tail = [[False] * T for _ in specs]
        free = getattr(self, "free_answers", None)  # eval only: the model writes its answers
        gen_fin = None
        roll_tok = self._refit_token_layers(plp)
        # PASS ROBUSTNESS (eval, --pass-forks; 10-22): at every pass of the active probe, fork the turn
        # (same prefix), sample pass_forks[0] answers at pass_forks[1] temperature and score them with
        # the SAME probe; results[t]["pass_forks"] = share of the samples that also pass
        pf_n, pf_temp = (getattr(self, "pass_forks", None) or (0, 0.0)) if free else (0, 0.0)
        fork_panel = getattr(self, "fork_panel", None) if free else None
        for t in range(T):
            pf_items = []
            fp_items, fp_umsg = [], (list(umsg) if fork_panel else None)  # turn t's renderers
            if free:  # tica_switch_eval --answers free (10-19b): greedy, read as a forced one
                gens = pol.generate(convs, free["max_new"])
                gen_fin = [g["finished"] for g in gens]
                for c, g in zip(convs, gens, strict=True):
                    c.append({"role": "assistant", "content": g["text"]})
            else:
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
            rl = sorted(set(pool.layers) | set(roll_tok) | set(stay_L))
            pooled = (pol.pooled_incremental(convs, rl, inc_state, token_out=tok)
                      if acfg.incremental_rollout or roll_tok else pol.pooled(
                          convs, sorted(set(pool.layers) | set(stay_L)) if stay_L
                          else pool.layers))
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
                    if tail[i][t]:
                        res["tail"] = True
                    else:
                        ep_stats["off"] += 1
                        ep_stats["off_det"] += det
                    self._budget_step(s, i, t, off, tail, judged=False, passed=False)
                    shown, shown_cond = (0.0, 1.0), "none"
                elif dfd is not None:
                    self._ep_acts[(i, t)] = pooled[plp][i : i + 1].detach()
                    self._ep_concept[(i, t)] = c
                    if stay_L:  # standalone stay: this turn's reads at the stay layers
                        self._ep_stay[(i, t)] = torch.cat(
                            [pooled[L][i : i + 1] for L in stay_L]).detach()
                    deployed_i = dep_by[i][c]
                    active_name = dfd.active  # the newest probe (an older one may catch below)
                    read = dfd.score(active_name, pooled, i)
                    active_p = read.get("p")  # the active probe's own p (an older catcher may
                    # replace `read` below); the next turn's progress term compares against it
                    self._kind_outcome(i, active_name, t, read["detected"])
                    if cumulative and not read["detected"]:
                        # No circularity: an OLDER probe of this episode catching a fallback to
                        # an old region counts as caught (the strongest catcher is shown).
                        olds = [dfd.score(n, pooled, i) for n in deployed_i if n != dfd.active]
                        olds = [r for r in olds if r["detected"]]
                        if olds:
                            read = max(olds, key=lambda r: r["score"] - r["thr"])
                            ep_stats["old_catch"] += 1
                    rb_share = None
                    if acfg.pass_robust_n and not free and not read["detected"]:
                        rb = self._robust_pass(s, t, c, convs[i], list(deployed_i), dfd)
                        rb_share = rb["share"]
                        if rb["catch"] is not None and (
                                rb_share * len(rb["rows"]) < acfg.pass_robust_min):
                            read = {**rb["catch"], "robust_fail": True}  # not a robust pass
                    obs = dfd.observe(read, pooled, i)
                    if pf_n and obs["passed"]:  # eval: re-sample this turn (pass robustness)
                        pf_items.append((i, active_name, c))
                    if fork_panel and t >= 1:  # eval: fresh-problem forks judged by THIS pool
                        fp_items.append((i, c, list(deployed_i)))
                    if ep_mode and (i, c) in pending:
                        pending[(i, c)]["turns"] += 1
                        if obs["passed"]:
                            self.deploy_log.append({**pending.pop((i, c)), "beaten": True})
                    if (i, c) not in seg_seen:  # first judged turn of concept c in episode i
                        seg_seen.add((i, c))
                        seg_pending[(i, c)] = {"concept": c, "kind": "segment", "turns": 0,
                                               **self._ep_tag(i)}
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
                        "active_p": active_p,
                    }
                    if rb_share is not None:
                        res["robust_share"] = rb_share
                    if obs["passed"]:
                        passed_turn.append(t)
                        if t >= 1:
                            self.refit_buf.add("pos", pooled, i, cap)
                            self._fresh_escapes += 1
                    shown = (read["score"], read["thr"])
                    if self.falsify:  # eval control: show the OPPOSITE verdict (refits still
                        # follow TRUE passes; forks carry the verdict the model actually saw);
                        # "pass" (10-22): ALWAYS show a pass (does the hiding hold without
                        # corrective feedback?)
                        th = read["thr"]
                        if self.falsify == "pass":
                            shown = (th - 1.0, th)
                        elif self.falsify == "score_noise":  # 10-23: TRUE verdict, random
                            # score on its side of the threshold (is the NUMBER used?)
                            u = random.Random(f"noise|{s.id}|{i}|{t}").uniform(1.0, 40.0)
                            shown = (th + u, th) if read["detected"] else (th - u, th)
                        else:
                            shown = (th - 1.0, th) if read["detected"] else (th + 1.0, th)
                    shown_cond = s.cond
                    bact = self._budget_step(s, i, t, off, tail, judged=True,
                                             passed=obs["passed"])
                    # eval-only exogenous schedule (forced_refit_turns, tica_switch_eval
                    # --refit-every): refit after these turns whatever the verdict, never on a pass
                    forced = self.forced_refit_turns
                    if ep_mode and (obs["passed"] if forced is None else t in forced):
                        if (i, c) in last_refit:
                            ep_stats["recov"].append(t - last_refit.pop((i, c)))
                        catcher = None
                        if t < T - 1 and acfg.episode_bank_first and bact != "end":
                            catcher = dfd.best_catcher(pooled, i)  # a member that catches it
                        if bact == "end":  # turn budget: no new probe fits; the rest is a tail
                            dfd.active = read["name"]
                            res["episode_refit"] = "end"
                        elif catcher is not None:  # quick mode: switch, no fork
                            dfd.active, dfd.stage_turn = catcher, 0
                            ep_stats["switch"] += 1
                            last_refit[(i, c)] = t
                            pending[(i, c)] = {"kind": "bank", "turns": 0, "concept": c,
                                               **self._ep_tag(i)}
                            res["episode_refit"] = "bank"
                        elif t < T - 1 and s.monitored_at(t + 1) != c:
                            # the next turn monitors another concept: no fork now; if the episode
                            # returns to c, its last-passed probe judges first (and a pass there
                            # refits at that point)
                            dfd.active = read["name"]
                            res["episode_refit"] = "switch"
                        elif t < T - 1 and self._erng(i).random() >= acfg.episode_refit_prob:
                            dfd.active = read["name"]  # no refit this time: same probe next turn
                            res["episode_refit"] = "skip"
                        elif t < T - 1:  # fork at the turn after the pass -> active from t+1
                            dfd.active = read["name"]  # refit, or keep the dodged probe
                            if overlap:  # read now, fit on the worker, finish at the turn's join
                                ovl_submit(i, s, c, t, shown, dfd, deployed_i, res)
                            else:
                                t_ep = time.time()
                                bank, info = self._episode_refit(s, convs[i], shown, c, t + 1,
                                                                 **self._ep_kw(i))
                                self._log_refit(info, i, t + 1, c, 0, time.time() - t_ep)
                                tried = [info.get("kind")]
                                bank, info = switch_refits(i, s, convs[i], shown, c, t, bank,
                                                           info, tried)
                                ep_stats["sec"] += time.time() - t_ep
                                finish_refit(i, c, t, dfd, deployed_i, res, bank, info, tried)
                else:
                    # neg_text (concept-free text under the brief): the monitored concept's base
                    # and held-out probes are read too, i.e. their IN-CONTEXT false-positive rate
                    # (10-21q; was hard-coded False / None). Logged only; shown + losses unchanged.
                    fp = s.cond == "neg_text" and s.text_concept is None
                    base = (pool.concept_monitor(s.monitored)
                            if (s.text_concept or fp) else None)
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
                    ) if (heldout and (s.text_concept or fp)) else None
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
                if gen_fin is not None:
                    res["gen_finished"] = gen_fin[i]
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
                    if v5 and res.get("active_monitor") and not res.get("off"):
                        # tagged_v5 (10-23): a different probe judges t + 1 (refit / bank switch,
                        # same concept; serial refits are final here) -> "New monitor"; the score
                        # change since the last turn judged by the SAME probe
                        nd = def_by[i].get(c_next)
                        kw["new_monitor"] = (c_next == c and nd is not None
                                             and nd.active != res["active_monitor"])
                        key = (i, c, res["active_monitor"])
                        kw["prev_score"] = v5_last.get(key)
                        v5_last[key] = shown[0]
                        v5_fix.append((i, s, c, c_next, res["active_monitor"], shown_cond, shown,
                                       nxt, kw))
                    fb = self.render_feedback(shown_cond, c, shown, nxt,
                                              feedback=s.feedback, named=s.named, **kw)
                    umsg[i] = (lambda x, sc=shown_cond, cc=c, sh=shown, kw=kw, s=s:
                               self.render_feedback(sc, cc, sh, x, feedback=s.feedback,
                                                    named=s.named, **kw))
                    tr.feedback.append(fb)
                    convs[i].append({"role": "user", "content": fb})
            ovl_join()  # refit_overlap: the turn's refits, before the next turn uses them
            for i, s, c, c_next, name_t, sc, sh, nxt, kw in v5_fix:
                # tagged_v5 + refit_overlap: a refit finished at the join -> the message already
                # in the conversation gets its "New monitor" notice (umsg[i] shares kw)
                nd = def_by[i].get(c_next)
                want = c_next == c and nd is not None and nd.active != name_t
                if want != kw["new_monitor"]:
                    kw["new_monitor"] = want
                    fb = self.render_feedback(sc, c, sh, nxt, feedback=s.feedback, named=s.named,
                                              **kw)
                    convs[i][-1] = {"role": "user", "content": fb}
                    trajs[i].feedback[-1] = fb
            v5_fix.clear()
            if pf_items:
                self._pass_forks(pf_items, trajs, t, def_by, free["max_new"], pf_n, pf_temp, rl,
                                 roll_tok)
            if fp_items:
                self._fork_panel(fp_items, trajs, specs, t, def_by, fp_umsg, free["max_new"], rl,
                                 roll_tok)
        n_cens, trajs_all = len(last_refit), trajs
        if ctx is not None:  # episode streams: every rank's records merged (dp: all-gathered)
            trajs_all, passed_turn, ep_stats, n_cens = self._merge_rollout(
                trajs, passed_turn, ep_stats, pending, seg_pending, last_refit, deploy_start,
                seg_start)
            pending, seg_pending = {}, {}  # censored records are already in the merged logs
            dodging = [s.cond in DODGE_CONDS for s in ctx["gspecs"]]
        self._g_trajs = trajs_all  # every episode of the step (cap_gen, logs)
        n_dodge = sum(dodging)
        # Concept switches: detection on the first turn of each new segment (caught by
        # construction ~1) and the pass rate on the later turns of switched-to segments; returns =
        # segments that come back to an earlier concept (all of its old probes judge).
        sw = {"seg_first": [], "seg_later": [], "ret_first": []}
        for tr in trajs_all:
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
        n_tail = sum(bool(r.get("tail")) for tr in trajs_all for r in tr.results)
        metrics = {
            "pass_rate": len(passed_turn) / max(1, n_dodge * T - n_tail),
            "mean_pass_turn": (sum(passed_turn) / len(passed_turn))
            if passed_turn
            else float("nan"),
            "bank_size": len(pool.entries) if ctx is None else ctx["bank_size"],
        }
        bud = [tr.spec.budget for tr in trajs_all if tr.spec.budget is not None]
        if bud:  # turn budget: share of unjudged tail turns, switches made
            metrics["budget_tail_frac"] = n_tail / max(1, n_dodge * T)
            metrics["budget_switches"] = sum(b["k"] for b in bud)
        metrics.update({k: v for k, v in ep_stats.items() if k.startswith("sw_")})
        for kind, (cg, jd) in self.kind_stats.items():
            metrics[f"kind_catch_{kind}"] = round(cg / max(1, jd), 3)
        if ep_mode:
            ep_stats["cens"] = n_cens  # accepted refits never passed again this episode
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
            metrics.update(probe_index_metrics(trajs_all))
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
            rbs = self.__dict__.pop("_rb_stats", None)
            if rbs and rbs[0]:  # pass_robust_n: raw passes checked, share not robust, mean share
                metrics.update({"robust_checks": rbs[0], "robust_fail_frac": rbs[1] / rbs[0],
                                "robust_share_mean": rbs[2] / rbs[0]})
            if "ovl_main" in ep_stats:
                # refit_overlap, pass refits only (first probes stay serial): read + fit work
                # (refits.jsonl sec), main-thread time on them (reads incl. any slow-down from the
                # concurrent fit, join waits, switch fallback) and the difference (time saved)
                metrics.update({
                    "sec_ep_refit_work": round(ep_stats["ovl_work"], 1),
                    "sec_ep_refit_main": round(ep_stats["ovl_main"], 1),
                    "sec_ep_refit_join_wait": round(ep_stats["ovl_wait"], 1),
                    "ep_refit_stacked": getattr(self._ovl_pool, "n_stacked", 0),
                    "sec_ep_refit_overlap_saved": round(ep_stats["ovl_work"] - ep_stats["ovl_main"],
                                                        1),
                    "sec_ep_refit_redo_waste": round(ep_stats["ovl_waste"], 1),
                    "ep_refit_overlap_redo": ep_stats["ovl_redo"],
                    "ep_refit_overlap_redo_same": ep_stats["ovl_redo_same"]})
            metrics.update(self._flush_refit_log())
        self._flush_turn_log(trajs_all)
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

        pol, pool, ctx = self.policy, self.pool, self._ep_ctx
        t0_key = (c, s.cond, bool(getattr(s, "named", True))) if len(conv) == 2 else None
        if t0_key is not None and t0_key in self._t0_first:
            # TURN 0 has no history: the fork set (opening(x) + x for the concept's texts) depends
            # only on the concept and the current weights, so one fork-fit first probe per concept
            # per rollout serves every episode that opens on it (10-16o speed-up; same probe
            # distribution, one fit instead of one per episode)
            if ctx is not None:
                ctx["t0_uses"][t0_key] += 1
            if self._t0_first[t0_key] is None:  # episode streams: a failed shared fit stays failed
                return None
            name, kind = self._t0_first[t0_key]
            pending[(i, c)] = {"kind": kind, "turns": 0, "concept": c, "first": True,
                               **self._ep_tag(i)}
            if ctx is None:  # episode streams: counted when the ranks' records merge
                ep_stats["first_shared"] = ep_stats.get("first_shared", 0) + 1
            return name
        kw, ks = {}, None
        if ctx is not None and t0_key is not None:
            # episode streams: the shared turn-0 probe has its own stream and name, so every rank
            # that needs it fits the same probe (no-prompt reads uncached: same batches everywhere)
            ks = "|".join(map(str, t0_key))
            kw = {"rng": random.Random(f"{ctx['t0_seed']}|{ks}"), "np_cache": False,
                  "name": f"ep_first_s{ctx['step']}_{zlib.crc32(ks.encode())}"}
            ctx["t0_uses"][t0_key] += 1
        elif ctx is not None:
            kw = self._ep_kw(i)
        t_ep = time.time()
        bank, info = self._episode_refit(s, conv[:-2], None, c, None, fork_msg=render, **kw)
        self._log_refit({**info, "first_probe": True}, i, len(conv) // 2 - 1, c, 0,
                        time.time() - t_ep)
        if ks is not None:
            self._refit_log[-1]["t0_key"] = ks  # one fit per key once the ranks merge
        ep_stats["sec"] += time.time() - t_ep
        count = ks is None  # shared turn-0 fits under episode streams: counted at the merge
        if count:
            ep_stats["first_att"] = ep_stats.get("first_att", 0) + 1
        if bank is None:
            if ks is not None:
                self._t0_first[t0_key] = ctx["t0_fit"][t0_key] = None
            return None
        bank = bank.to(pol.device)
        pool.add(bank, round=self.step_k, family="episode_refit")
        self._transient.append(bank.name)
        self._kind_of[bank.name] = info.get("kind", "linear")
        if count:
            ep_stats["first_acc"] = ep_stats.get("first_acc", 0) + 1
        pending[(i, c)] = {"kind": info.get("kind", "linear"), "turns": 0, "concept": c,
                           "first": True, **self._ep_tag(i)}
        if t0_key is not None:
            self._t0_first[t0_key] = (bank.name, info.get("kind", "linear"))
            if ks is not None:
                ctx["t0_fit"][t0_key] = bank.name
        return bank.name

    def _panel_kinds(self) -> list[str]:
        """adaptive.first_probe_panel_kinds, [] = the in-training zoo: every single-layer kind of
        episode_refit_kinds at every kind_layers depth + the multi-layer kinds as they are."""

        ac = self.cfg.adaptive
        if ac.first_probe_panel_kinds:
            return list(ac.first_probe_panel_kinds)
        out = []
        for k in ac.episode_refit_kinds:
            arch, L = split_kind(k)
            if L is not None or arch in ZOO_ENS or arch in ZOO_TOKENS_ENS or not ac.kind_layers:
                out.append(k)
            else:
                out.extend(f"{arch}@{int(Lk)}" for Lk in ac.kind_layers)
        return out

    def _panel_first(self, i, s, c, render, pending, ep_stats):
        """first_probe_panel_from_step (10-20): concept c's turn-0 probe in episode i = ONE fixed
        BASE-model probe drawn uniformly (episode stream) from the panel. Each (concept, condition,
        named, kind) probe is fit once with the turn-0 fork construction (as _fork_first) under
        policy.reference() (adapters off: the base model's states), with a deterministic stream
        and name, so every data-parallel rank and every resume chunk fits the same probe; it then
        stays in the pool for the run (family "turn0_panel"). None (base probe) if its gate fails.
        Turns-to-beat records carry kind "base:<kind>" (ignored by the refit-kind bandit)."""

        kind = self._erng(i).choice(self._panel_kinds())
        name, info, sec = self._panel_probe(s, c, kind, render)
        if info is not None:
            self._log_refit({**info, "first_probe": True, "panel": True}, i, 0, c, 0, sec)
            ep_stats["sec"] += sec
        if name is None:
            return None
        pending[(i, c)] = {"kind": f"base:{kind}", "turns": 0, "concept": c, "first": True,
                           **self._ep_tag(i)}
        return name

    def _panel_probe(self, s, c, kind, render):
        """One turn-0 panel probe (concept c, s's condition / naming, kind): fit once under
        policy.reference() with the deterministic stream and name (disk cache if configured), then
        kept in the pool (family "turn0_panel"). Returns (name or None if its gate failed, fit info
        or None when it was already known, seconds)."""

        pol, pool = self.policy, self.pool
        key = (c, s.cond, bool(getattr(s, "named", True)), kind)
        cache = self.__dict__.setdefault("_panel0", {})
        if key in cache:
            return cache[key], None, 0.0
        ks = "|".join(map(str, key))
        name = f"panel0_{zlib.crc32(ks.encode())}"
        try:
            pool.get(name)  # restored with the pool (resume): keep it
            cache[key] = name
            return name, None, 0.0
        except KeyError:
            pass
        t_ep = time.time()
        with pol.reference():
            if self.cfg.adaptive.first_probe_panel_cache:  # TamIA 10-21: disk cache
                bank, info = self._panel_cached(s, c, kind, render, ks, name)
            else:
                bank, info = self._episode_refit(
                    s, [], None, c, None, kind=kind, fork_msg=render,
                    rng=random.Random(f"panel0|{ks}"), name=name, np_cache=False)
        if bank is not None:
            pool.add(bank.to(pol.device), round=self.step_k, family="turn0_panel")
            self._kind_of[name] = f"base:{kind}"
        cache[key] = name if bank is not None else None
        return cache[key], info, time.time() - t_ep

    def _anchor_names(self, s, c) -> list[str]:
        """adaptive.match_anchor "panel" (10-22): every turn-0 panel probe of concept c for s's
        condition (all kinds of _panel_kinds; gate failures dropped), through _panel_probe with the
        render precompute_panel uses (cache hits when the panel cache is filled)."""

        named = bool(getattr(s, "named", True))
        key = (c, s.cond, named)
        cache = self.__dict__.setdefault("_anchors", {})
        if key not in cache:
            brief = "real_score" if CONDITION_SPECS[s.cond]["brief"] else "nobrief"

            def render(x, c=c, nm=named, b=brief):  # the turn-0 user message (as precompute)
                return self.opening(b, x, c, named=nm)

            model = self.policy.model
            was = model.training
            model.eval()  # fits read with the KV cache (dropped in train mode + grad ckpt)
            try:
                names = []
                for kind in self._panel_kinds():
                    name, info, sec = self._panel_probe(s, c, kind, render)
                    if info is not None and hasattr(self, "_refit_log"):  # (log exists after a rollout)
                        self._log_refit({**info, "anchor": True, "panel": True}, -1, 0, c, 0, sec)
                    if name is not None:
                        names.append(name)
            finally:
                model.train(was)
            if not self.cfg.adaptive.match_anchor_tokens:  # pooled kinds only (cost, 10-22)
                names = [n for n in names
                         if not getattr(self.pool.get(n), "needs_tokens", False)]
            cache[key] = names
        return cache[key]

    def _pass_forks(self, items, trajs, t, def_by, max_new, n, temp, layers, tok_layers):
        """--pass-forks (eval, 10-22): for each (episode, probe, concept) that passed at turn t,
        sample n answers from the same prefix (temperature temp) and score each with that probe.
        Writes trajs[i].results[t]["pass_forks"] = passing share (1.0 = a robust pass)."""

        pol = self.policy
        prefixes, owner = [], []
        for k, (i, _, _) in enumerate(items):
            pre = [dict(x) for x in trajs[i].convs[t][:-1]]  # through turn t's user message
            prefixes.extend([pre] * n)
            owner.extend([k] * n)
        gens = pol.generate(prefixes, max_new, temperature=temp, seed=1000 + t)
        convs = [[*pre, {"role": "assistant", "content": g["text"]}]
                 for pre, g in zip(prefixes, gens, strict=True)]
        tok = {} if tok_layers else None
        pooled = pol.pooled_incremental(convs, layers, {}, token_out=tok)
        for L in tok_layers:
            pooled[("tok", L)] = tok[L]
        ok = [0] * len(items)
        for row, k in enumerate(owner):
            i, name, c = items[k]
            ok[k] += int(not def_by[i][c].score(name, pooled, row)["detected"])
        for k, (i, _, _) in enumerate(items):
            trajs[i].results[t]["pass_forks"] = ok[k] / n

    def _fork_panel(self, items, trajs, specs, t, def_by, umsg_t, max_new, layers, tok_layers):
        """FORK PANEL (eval, 10-27; tica_switch_eval --fork-panel M): performance at a given refit
        level with many samples. For each monitored episode at turn t >= 1: M forks with the same
        history and the same verdict message as turn t but a NEW request of the concept (a text
        never used in the episode so far), answered by the model (greedy unless fork_panel["temp"])
        and judged like the main turn by every probe deployed AT turn t (cumulative: caught = any
        fires; the turn's refit, if any, comes after). Records -> fork_panel["log"] (raw text)."""

        fp, pol = self.fork_panel, self.policy
        if not hasattr(self, "_fp_texts"):
            self._fp_texts = {ct.concept: list(ct.texts)
                              for ct in getattr(self.sampler, "concepts", [])}
        convs, owner = [], []
        for i, c, judges in items:
            sp = specs[i]
            seen = {sp.text(u) for u in range(t + 1)}
            cand = [x for x in self._fp_texts.get(c, []) if x not in seen]
            rng = random.Random(f"fork|{sp.id}|{t}")
            pre = [dict(x) for x in trajs[i].convs[t][:-2]]  # through turn t - 1
            for x in rng.sample(cand, min(fp["m"], len(cand))):
                convs.append([*pre, {"role": "user", "content": umsg_t[i](x)}])
                owner.append((i, c, judges))
        if not convs:
            return
        gens = pol.generate(convs, max_new, temperature=fp.get("temp", 0.0),
                            seed=fp.get("seed", 0) + t)
        full = [[*cv, {"role": "assistant", "content": g["text"]}]
                for cv, g in zip(convs, gens, strict=True)]
        tok = {} if tok_layers else None
        pooled = pol.pooled_incremental(full, layers, {}, token_out=tok)
        for L in tok_layers:
            pooled[("tok", L)] = tok[L]
        for row, ((i, c, judges), cv, g) in enumerate(zip(owner, full, gens, strict=True)):
            caught = any(def_by[i][c].score(n, pooled, row)["detected"] for n in judges)
            fp["log"].append({"ep": specs[i].id, "turn": t, "concept": c,
                              "n_deployed": len(judges), "passed": not caught,
                              "finished": g["finished"], "user": cv[-2]["content"],
                              "answer": g["text"]})

    def _robust_pass(self, s, t, c, conv, names, dfd) -> dict:
        """adaptive.pass_robust_n (10-22): read turn t's prefix with pass_robust_n other texts of
        concept c (shared-prefix forks; deterministic per global step / episode / turn) and score
        each with every deployed probe `names`. Returns {"share": passing share, "rows": n,
        "catch": the strongest catching read (or None)}."""

        acfg, pol = self.cfg.adaptive, self.policy
        if not hasattr(self, "_ctexts"):
            self._ctexts = {x.concept: x.texts for x in self.branch_concepts or []}
        cur = conv[-1]["content"]
        texts = [x for x in self._ctexts.get(c, []) if x != cur]
        rng = random.Random(f"robust|{self.step_k + self._step_offset()}|{s.id}|{t}")
        pick = rng.sample(texts, min(acfg.pass_robust_n, len(texts)))
        if not pick:
            return {"share": 1.0, "rows": [], "catch": None}
        pre = [dict(m) for m in conv[:-1]]  # through turn t's user message
        forks = [[*pre, {"role": "assistant", "content": x}] for x in pick]
        mons = [self.pool.get(n) for n in names]
        tok_L = sorted({L for m in mons if getattr(m, "needs_tokens", False)
                        for L in (getattr(m, "layers_used", None) or [m.layer])})
        Ls = sorted({m.layer for m in mons} | {L for m in mons
                                                for L in (getattr(m, "layers_used", None) or [])}
                    | set(tok_L))
        ft = {} if tok_L else None
        rall = pol.pooled_shared_prefix(forks, Ls, token_out=ft,
                                        batch=acfg.episode_fork_batch or None)
        pooled = {L: rall[L].to(pol.device) for L in Ls}
        for L in tok_L:
            pooled[("tok", L)] = ft[L]
        ok, catch = 0, None
        for row in range(len(forks)):
            det = [r for r in (dfd.score(n, pooled, row) for n in names) if r["detected"]]
            if not det:
                ok += 1
                continue
            w = max(det, key=lambda r: r["score"] - r["thr"])
            if catch is None or w["score"] - w["thr"] > catch["score"] - catch["thr"]:
                catch = w
        st = self.__dict__.setdefault("_rb_stats", [0, 0, 0.0])
        st[0] += 1
        st[1] += int(ok < acfg.pass_robust_min)
        st[2] += ok / len(forks)
        return {"share": ok / len(forks), "rows": list(range(len(forks))), "catch": catch}

    def precompute_panel(self, shard: int = 0, nshard: int = 1) -> dict:
        """Fill adaptive.first_probe_panel_cache ahead of training (TamIA 10-21): every turn-0 panel
        probe the run can draw = train concept x condition of the train mix x panel kind, each
        through the SAME path as _panel_first (its key string, stream, name and render), this
        process taking keys shard::nshard. Returns hit / miss / gate-failed counts."""

        from types import SimpleNamespace

        acfg = self.cfg.adaptive
        if not acfg.first_probe_panel_cache:
            raise ValueError("precompute_panel needs adaptive.first_probe_panel_cache")
        conds = [c for c, w in zip(self.sampler.conds, self.sampler.weights, strict=True) if w > 0]
        concepts = [c.concept for c in self.branch_concepts or []]
        named = not getattr(self.sampler, "hidden", False)
        keys = [(c, cond, named, k) for c in concepts for cond in conds
                for k in self._panel_kinds()][shard::nshard]
        out = {"keys": len(keys), "hit": 0, "miss": 0, "failed": 0}
        t0 = time.time()
        # eval mode as in the rollout: HF drops the KV cache (shared-prefix reads) in train mode
        # with gradient checkpointing
        self.policy.model.eval()
        for j, (c, cond, nm, kind) in enumerate(keys):
            s = SimpleNamespace(cond=cond, named=nm, monitored=c, feedback=None)
            brief = "real_score" if CONDITION_SPECS[cond]["brief"] else "nobrief"

            def render(x, c=c, nm=nm, b=brief):  # the turn-0 user message (rollout's umsg)
                return self.opening(b, x, c, named=nm)

            ks = "|".join(map(str, (c, cond, nm, kind)))
            name = f"panel0_{zlib.crc32(ks.encode())}"
            with self.policy.reference():
                bank, info = self._panel_cached(s, c, kind, render, ks, name)
            out[info["panel_cache"]] += 1
            out["failed"] += bank is None
            if j % 20 == 0:
                print(json.dumps({"panel_precompute": j, **out,
                                  "sec": round(time.time() - t0, 1)}), flush=True)
        print(json.dumps({"panel_precompute_done": True, **out,
                          "sec": round(time.time() - t0, 1)}), flush=True)
        return out

    def _panel_cached(self, s, c, kind, render, ks, name):
        """_panel_first's fit through adaptive.first_probe_panel_cache (TamIA 10-21): the same plan
        -> reads -> fit as _episode_refit (same stream, same name), but the result (probe or None,
        info) is stored under a hash of everything the fit depends on and loaded on a hit."""

        import dataclasses
        import hashlib

        acfg, pol = self.cfg.adaptive, self.policy
        plan = self._refit_plan(s, [], None, c, kind, render, random.Random(f"panel0|{ks}"),
                                False)
        src = Path(__file__).resolve().parents[1]
        # fitting code only (10-22): probe_zoo.py + monitors.py whole, and from adaptive.py the refit
        # fit functions -- hashing all of adaptive.py made every config-field edit a full cache miss
        import inspect

        from next_chameleons.tica import adaptive as _ad

        fit_src = "".join(inspect.getsource(o) for o in (
            _ad.RefitData, _ad._split, _ad.fit_refit_bank, _ad.MLPMonitor, _ad.fit_refit_mlp))
        code = hashlib.sha256(b"".join((src / f).read_bytes() for f in (
            "probe_zoo.py", "monitors.py")) + fit_src.encode()).hexdigest()[:16]
        model = Path(str(getattr(getattr(pol.model, "config", None), "_name_or_path", ""))).name
        key = json.dumps({
            "v": 1, "code": code, "model": model, "name": name, "kind": plan["full_kind"],
            "L": plan["L"], "Ls": list(plan["Ls"]), "excl": sorted(self._heldout_layers()),
            "labels": [k for k, _ in plan["picks"]], "forks": plan["forks"],
            "np": plan["np_convs"], "ids0": pol.prompt_ids(plan["forks"][0]),
            "np_ids0": pol.prompt_ids(plan["np_convs"][0]) if plan["np_convs"] else [],
            # scheduling-only fields (fits run on a worker thread / side stream: same probe) are
            # keyed at their defaults, so switching them on keeps the precomputed cache (10-27)
            "refit": {f.name: (False if f.name in ("refit_overlap", "refit_overlap_stream")
                               else getattr(acfg, f.name)) for f in dataclasses.fields(acfg)
                      if f.name.startswith("refit_")
                      and f.name not in ("refit_cpu_fits", "refit_cpu_threads",
                                         "refit_stack_episodes")}},
            sort_keys=True, default=str)
        d = Path(acfg.first_probe_panel_cache)
        path = d / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.pt"
        if path.exists():
            st = torch.load(path, map_location="cpu", weights_only=False)
            return st["bank"], {**st["info"], "panel_cache": "hit"}
        job = self._refit_reads(plan)
        job["name"], job["excl"] = name, self._heldout_layers()
        bank, info = _refit_fit(job)
        d.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save({"bank": bank.to("cpu") if bank is not None else None, "info": info}, tmp)
        os.replace(tmp, path)  # atomic: concurrent ranks / jobs never read a partial file
        return bank, {**info, "panel_cache": "miss"}

    def _episode_refit(self, s, prefix, shown, concept=None, turn=None, kind=None,
                       fork_msg=None, rng=None, name=None, np_cache=True):
        """refit_mode="episode": fork this conversation at the turn AFTER its pass (prefix ends with
        the passing answer; each fork = prefix + the shown "not caught" verdict + one follow-up) and
        fit a fresh probe at the concept layer. Shared-prefix reads (one cached forward of the
        history). Positives: in-context concept follow-ups + half the no-prompt concept turns;
        negatives: in-context other-concept + concept-free follow-ups; the other no-prompt half is
        the out-of-context TPR gate. Same construction as scripts/tica_frozen_race.py.
        rng / name / np_cache (episode streams): the episode's stream, a deterministic probe name
        (it seeds the fit) and whether no-prompt reads may come from the rollout cache.
        Stages: _refit_plan (every draw) -> _refit_reads (GPU reads) -> _refit_fit (no trainer
        state; adaptive.refit_overlap runs it on a worker thread)."""

        plan = self._refit_plan(s, prefix, shown, concept, kind, fork_msg, rng, np_cache)
        job = self._refit_reads(plan)
        if name is None:
            self.refit_k += 1
            name = f"ep_refit_{self.refit_k}"
        job["name"], job["excl"] = name, self._heldout_layers()
        return _refit_fit(job)

    def _evdet_buffer(self, L: int) -> EvDetBuffer:
        """The run's generic-detector buffer (refit kinds evdet@L / pcares@L), seeded once per layer
        from adaptive.evdet_seed_model / evdet_seed_base (train-concept panel dumps) when set."""

        if not hasattr(self, "_evdet_buf"):
            self._evdet_buf, self._evdet_seeded = EvDetBuffer(), {}
        acfg = self.cfg.adaptive
        if L not in self._evdet_seeded and acfg.evdet_seed_model:
            self._evdet_seeded[L] = self._evdet_buf.seed(
                acfg.evdet_seed_model.split(","), acfg.evdet_seed_base.split(","), L,
                device=self.policy.device)
            print(f"[evdet] seeded layer {L}: {self._evdet_seeded[L]} pairs", flush=True)
        return self._evdet_buf

    def _refit_plan(self, s, prefix, shown, concept=None, kind=None, fork_msg=None, rng=None,
                    np_cache=True) -> dict:
        """_episode_refit's draws (texts, kind, no-prompt texts) + fork conversations; no reads."""

        acfg = self.cfg.adaptive
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
        rng = rng if rng is not None else self.rng
        hard = self.hard_neg.get(concept) or []
        n_hard = min(len(hard), ef["other"] // 2)
        # close negatives (adaptive.family_neg_frac): part of the other-concept draws come from
        # the concept's own family
        fam = self._families() if acfg.family_neg_frac > 0 else {}
        near = [c for c in others if fam.get(concept) and fam.get(c) == fam[concept]]
        pos_p = [("pos", x) for x in rng.sample(pos_t, min(ef["pos"], len(pos_t)))]
        hard_p = [("neg", x) for x in rng.sample(hard, n_hard)]
        picks = (
            pos_p + hard_p
            + [("neg", x) for x in other_negatives(rng, texts, others, near,
                                                   ef["other"] - n_hard, acfg.family_neg_frac)]
            + [("neg", x) for x in rng.sample(free_negs, min(ef["free"], len(free_negs)))]
        )
        pre = [dict(m) for m in prefix]

        def fork(x):
            msg = fork_msg(x) if fork_msg is not None else self.render_feedback(
                s.cond, concept, shown, x, feedback=s.feedback, named=s.named,
                **({"new_monitor": True}  # v5: a refit's forks precede it
                   if getattr(self.render_feedback, "v5", False) else {}))
            return [*pre, {"role": "user", "content": msg}, {"role": "assistant", "content": x}]

        forks = [fork(x) for _, x in picks]
        full_kind = kind or self._pick_kind(rng=rng)  # e.g. "linear@22": kind at layer 22
        kind, Lo = split_kind(full_kind)
        if Lo is not None:
            L = Lo
        if kind in DETECTOR_KINDS:  # generic hiding detectors (eval only): the concept's forks
            keep = [q for q, (k_, _) in enumerate(picks) if k_ == "pos"]
            # selective kinds: the policy's reads of the other-concept / concept-free follow-ups
            # under the same brief (nothing to hide) are honest negatives + calibration
            mis_forks = ([f for f, (k_, _) in zip(forks, picks, strict=True) if k_ == "neg"]
                         if kind.endswith("sel") else [])
            picks, forks = [picks[q] for q in keep], [forks[q] for q in keep]
            used = {x for _, x in picks}
            rest = [x for x in pos_t if x not in used]
            cal = rng.sample(rest, min(EVDET_CAL, len(rest)))  # base-only calibration reads
            cal_forks = [fork(x) for x in cal]
            return {"t0": t_read0, "concept": concept, "L": L, "kind": kind,
                    "full_kind": full_kind, "Ls": [L], "picks": picks, "forks": forks,
                    "cal_forks": cal_forks, "mis_forks": mis_forks, "np_convs": [], "nps": [],
                    "np_cache": np_cache}
        Ls = kind_layers(kind, L) if kind in ZOO_ENS or kind in ZOO_TOKENS_ENS else [L]
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
        return {"t0": t_read0, "concept": concept, "L": L, "kind": kind, "full_kind": full_kind,
                "Ls": Ls, "picks": picks, "forks": forks, "np_convs": np_convs, "nps": nps,
                "np_cache": np_cache}

    def _refit_reads(self, plan: dict, t0: float | None = None) -> dict:
        """_episode_refit's activation reads for a plan -> the fit job (everything _refit_fit
        needs; the caller adds "name" and "excl"). Re-running it on the same plan repeats the same
        reads (adaptive.refit_overlap redoes them after a rollback)."""

        acfg, pol = self.cfg.adaptive, self.policy
        t_read0 = plan["t0"] if t0 is None else t0
        kind, L, Ls, picks = plan["kind"], plan["L"], plan["Ls"], plan["picks"]
        if kind in DETECTOR_KINDS:  # the same forks read by the policy and by the base model
            batch = acfg.episode_fork_batch or None
            nf = len(plan["forks"])
            rp = pol.pooled_shared_prefix(plan["forks"] + plan["mis_forks"], [L],
                                          batch=batch)[L].float()
            rp, rm = rp[:nf], rp[nf:]
            with pol.reference():  # forks + calibration forks in one shared-prefix read
                rb = pol.pooled_shared_prefix(plan["forks"] + plan["cal_forks"], [L],
                                              batch=batch)[L].float()
            rb, rc = rb[:nf], rb[nf:]
            dt = time.time() - t_read0
            self._refit_read_sec = getattr(self, "_refit_read_sec", 0.0) + dt
            return {"kind": kind, "full_kind": plan["full_kind"], "L": L, "Ls": [L], "L0": L,
                    "picks": picks, "concept": plan["concept"], "r_pol": rp, "r_base": rb,
                    "r_cal": rc, "r_mis": rm if len(rm) else None,
                    "evdet_buf": self._evdet_buffer(L), "acfg": acfg, "device": pol.device,
                    "sec_read": round(dt, 2), "read_dt": dt}
        ft = {} if kind in ZOO_TOKEN else None  # per-token fork states (token-pooled kinds)
        keep = ZOO_TOK_KEEP if kind in ZOO_TOKENS_ENS else None
        rall = pol.pooled_shared_prefix(plan["forks"], Ls, token_out=ft, token_keep=keep,
                                        batch=acfg.episode_fork_batch or None)
        L0 = L if L in Ls else Ls[0]  # an ensemble may not read the primary layer (ensdeep4)
        r, rl = rall[L0].float().cpu(), {Lk: rall[Lk].float() for Lk in Ls}
        nps, np_convs = plan["nps"], plan["np_convs"]
        nt = {} if kind in ZOO_TOKEN else None
        if acfg.refit_np_cache and plan["np_cache"]:  # shared across the rollout's refits
            nall = self._np_reads(nps, np_convs, Ls, nt, keep)
        else:
            nall = (pol.pooled_incremental(np_convs, Ls, {}, token_out=nt, token_keep=keep)
                    if nt is not None
                    else pol.pooled(np_convs, Ls))
        npr, npl = nall[L0].float().cpu(), {Lk: nall[Lk].float() for Lk in Ls}
        if pol.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t_read0
        self._refit_read_sec = getattr(self, "_refit_read_sec", 0.0) + dt
        return {"kind": kind, "full_kind": plan["full_kind"], "L": L, "Ls": Ls, "L0": L0,
                "picks": picks, "concept": plan["concept"], "r": r, "rl": rl, "ft": ft,
                "npr": npr, "npl": npl, "nt": nt, "acfg": acfg, "device": pol.device,
                "sec_read": round(dt, 2), "read_dt": dt}  # sec_read: refits.jsonl

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
            t = self._fork_turn(tr)
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
                         "q": q, "kind": "gsm8k"})
            forks.append(fork)
        free_forks, free_meta = self._cap_free_forks(trajs)
        forks, meta = forks + free_forks, meta + free_meta
        # refit level of the capability forks (10-27): probes deployed at the forked turn (every
        # rank draws the same forks, so these are step-wide numbers before sharding)
        fork_probes = [len(m["names"]) for m in meta if m["kind"] != "nobrief"]
        if not forks:
            return {"cap_gen/n_forks": 0}
        gen_seed = None
        shard = self.world > 1 and acfg.cap_gen_shard
        if self.world > 1:  # data-parallel: rank 0 generates and backpropagates for every rank
            gen_seed = self.rng.randrange(1 << 30)  # drawn on every rank (shared RNG)
            self._share_monitors({x for m in meta for x in m["names"]}, to_all=shard)
            if not shard and self.rank != 0:
                return {}
        n_forks_all = len(forks)
        if shard:  # cap_gen_shard: this rank's forks (rank, rank + world, ...)
            forks, meta = forks[self.rank :: self.world], meta[self.rank :: self.world]
            gen_seed += self.rank
        n = acfg.cap_gen_samples
        convs = [f for f in forks for _ in range(n)]
        gens = []
        if convs:
            was, bs = pol.model.training, pol.batch_size
            pol.model.eval()  # KV cache (HF drops it in train mode with gradient checkpointing)
            pol.batch_size = acfg.cap_gen_gen_batch or len(convs)  # one batched call by default
            try:
                with (pol.merged() if acfg.cap_gen_merged else nullcontext()):  # TamIA 10-21
                    gens = pol.generate(convs, acfg.cap_gen_max_new,
                                        temperature=acfg.cap_gen_temperature,
                                        seed=(self.rng.randrange(1 << 30) if gen_seed is None
                                              else gen_seed))
            finally:
                pol.batch_size = bs
                pol.model.train(was)
        t_gen = time.time() - t0
        # items: (conversation, meta, NLL on it?). KL-only items = wrong samples (cap_gen_kl_all).
        kl_w = acfg.cap_gen_kl_weight
        items, n_ok, n_work, n_fb = [], 0, 0, 0
        fork_ok = [False] * len(forks)
        free = {"n": 0, "valid": 0, "nb": 0}  # cap_free_* samples (logged apart from GSM8K)
        # 10-28 (user: "is capability still preserved as refits increase?"): per refit level of the
        # fork (probes judging it: 1, 2, 3, 4, 5+) [GSM8K samples, correct, free samples, valid]
        lvl = [[0, 0, 0, 0] for _ in range(5)]
        for j, g in enumerate(gens):
            m = meta[j // n]
            b = min(max(len(m["names"]), 1), 5) - 1
            if m["kind"] == "gsm8k":
                ok = g["finished"] and is_correct(g["text"], m["gold"])
                wk = work_shown(g["text"])
                n_ok += ok and wk
                n_work += wk
                good = ok and wk
                lvl[b][0] += 1
                lvl[b][1] += good
            elif m["kind"] == "free":  # does the task: functional check (SQL runs, JSON is the
                # requested function schema) where one exists, else the form check (validity.py)
                good = bool(g["finished"]
                            and check_valid(m["concept"], g["text"], m["q"], m.get("ref")))
                free["n"] += 1
                free["valid"] += good
                lvl[b][2] += 1
                lvl[b][3] += good
            else:  # "nobrief": KL to the base model only
                good = False
                free["nb"] += 1
            if good:
                fork_ok[j // n] = True
                items.append((convs[j] + [{"role": "assistant", "content": g["text"]}], m, True))
            elif kl_w > 0 and (acfg.cap_gen_kl_all or m["kind"] == "nobrief") and g["text"].strip():
                items.append((convs[j] + [{"role": "assistant", "content": g["text"]}], m,
                              False))
        if acfg.cap_gen_log and gens:  # 10-20: the samples, for hand checks (scratch only)
            self.out.mkdir(parents=True, exist_ok=True)
            with open(self.out / f"cap_gen_rank{self.rank}.jsonl", "a") as fh:
                for j, g in enumerate(gens):
                    m = meta[j // n]
                    fh.write(json.dumps({
                        "step": self.step_k + self._step_offset(), "fork": j // n,
                        "q": m["q"], "gold": m["gold"], "msg": m["msg"], "kind": m["kind"],
                        "verdict_msg": convs[j][-1]["content"][: -len(m["q"])].strip(),
                        "text": g["text"], "finished": g["finished"],
                        "correct": bool(g["finished"] and (
                            is_correct(g["text"], m["gold"]) if m["kind"] == "gsm8k"
                            else m["kind"] == "free" and check_valid(
                                m["concept"], g["text"], m["q"], m.get("ref")))),
                        "work": bool(work_shown(g["text"])), "turn": len(convs[j]) // 2}) + "\n")
        for f, ok in enumerate(fork_ok):  # every sample wrong: train on the base solution
            if not ok and meta[f]["base"]:
                n_fb += 1
                items.append((forks[f] + [{"role": "assistant", "content": meta[f]["base"]}],
                              meta[f], True))
        ow = self.cfg.objective.hide_weight
        hv = acfg.cap_gen_hide_valid_only  # hide only on passing samples (flag), else KL only
        n_hide = sum(len([x for x in m["names"]
                          if not getattr(self.pool.get(x), "needs_tokens", False)])
                     for _, m, f in items if f or not hv) if acfg.cap_gen_hide else 0
        n_flag, n_items = sum(f for _, _, f in items), len(items)
        if shard:  # the step's normalisers: every rank's items
            n_hide, n_flag, n_items = (int(x) for x in dp.sum_floats(
                [n_hide, n_flag, n_items]))
        n_nll = max(1, n_flag)
        s_nll, s_hide, s_caught, s_kl, n_klm = 0.0, 0.0, 0, 0.0, 0
        if items:
            layers = self.pool.layers
            refs = []  # KL anchor: the base model answering the BARE problem (no history)
            if kl_w > 0:
                with pol.reference(), torch.no_grad():
                    for a in range(0, len(items), 2):
                        ch = items[a : a + 2]
                        ex = {"topk": acfg.kl_topk}
                        pol.forward([pol.view([{"role": "user", "content": m["q"]}, c[-1]])
                                     for c, m, _ in ch], [], grad=False, with_nll=True,
                                    extra=ex)
                        refs += ex["ref"]
            for a in range(0, len(items), 2):
                chunk = items[a : a + 2]
                views = [pol.view(c) for c, _, _ in chunk]
                extra = None
                if kl_w > 0:
                    rr = refs[a : a + 2]
                    n_tok = [len(v.ids) - v.start for v in views]
                    rr = [r if r[0].shape[0] == k else None
                          for r, k in zip(rr, n_tok, strict=True)]
                    n_klm += sum(r is None for r in rr)
                    extra = {"kl_ref": rr}
                pooled, nll = pol.forward(views, layers, grad=True, with_nll=True, extra=extra)
                w = torch.tensor([float(f) for _, _, f in chunk], device=nll.device)
                # adaptive.cap_level_adapt: 1 + gain x deficit of the item's refit level
                lm = torch.tensor([1.0 + acfg.cap_level_gain * self._cap_deficit(
                    min(max(len(m["names"]), 1), 5) - 1) if acfg.cap_level_adapt
                                   and m["kind"] != "nobrief" else 1.0 for _, m, _ in chunk],
                                  device=nll.device)
                loss = acfg.cap_gen_weight * self._cap_mult * (lm * w * nll).sum() / n_nll
                s_nll += float((w * nll).detach().sum())
                if extra is not None:
                    loss = loss + kl_w * self._cap_mult * (lm * extra["kl"]).sum() / n_items
                    s_kl += float(extra["kl"].detach().sum())
                if n_hide:
                    tg, rows = [], []
                    for r, (_, m, f) in enumerate(chunk):
                        if hv and not f:
                            continue
                        for x in m["names"]:
                            if not getattr(self.pool.get(x), "needs_tokens", False):
                                tg.append(Target(r, 0, "cap", x, m["concept"], "hide"))
                                rows.append(r)
                    if tg:
                        form = self.cfg.objective.hide_loss
                        if form == "mse":
                            p = probe_prob(self.pool, pooled, rows, tg)
                            h_term = ow * (p**2).sum() / n_hide
                        else:
                            p, z = probe_prob(self.pool, pooled, rows, tg, return_z=True)
                            h_term = ow * hide_per_target(
                                p, z, form, self.cfg.objective.hide_gce_q).sum() / n_hide
                        if self.cfg.hide_grad_clip:  # separate hide buffer (10-21r)
                            self._hide_grad_acc(h_term)
                        else:
                            loss = loss + h_term
                        s_hide += float((p**2).detach().sum())
                        s_caught += int((p > 0.5).sum())
                loss.backward()
        gsm = [j for j in range(len(gens)) if meta[j // n]["kind"] == "gsm8k"]
        c = {"forks": sum(m["kind"] == "gsm8k" for m in meta), "gens": len(gsm), "ok": n_ok,
             "work": n_work, "fin": sum(gens[j]["finished"] for j in gsm),
             "msg": sum(m["msg"] for m in meta if m["kind"] == "gsm8k"),
             "solved": sum(ok for ok, m in zip(fork_ok, meta, strict=True)
                           if m["kind"] == "gsm8k"), "fb": n_fb, "items": len(items),
             "kl_only": sum(not f for _, _, f in items), "nll": s_nll, "kl": s_kl, "klm": n_klm,
             "hide": s_hide, "caught": s_caught}
        if shard:  # logged: sums over every rank
            c = dict(zip(c, dp.sum_floats(list(c.values())), strict=True))
        g_n, f_n = max(1, c["gens"]), max(1, c["forks"])
        if shard:
            free = dict(zip(free, dp.sum_floats(list(free.values())), strict=True))
            flat = dp.sum_floats([x for row in lvl for x in row])
            lvl = [flat[4 * k : 4 * k + 4] for k in range(5)]
        by_lvl = {}
        if acfg.cap_level_adapt:  # update after this step's terms used the previous EMAs
            self._cap_level_update(lvl)
            for b in range(5):
                tag = f"k{b + 1}" if b < 4 else "k5plus"
                by_lvl[f"cap_level/deficit_{tag}"] = round(self._cap_deficit(b), 3)
        for k, (gn, gok, fn, fok) in enumerate(lvl):
            tag = f"k{k + 1}" if k < 4 else "k5plus"
            if gn:
                by_lvl.update({f"cap_gen/acc_{tag}": gok / gn, f"cap_gen/n_{tag}": int(gn)})
            if fn:
                by_lvl.update({f"cap_free/valid_{tag}": fok / fn, f"cap_free/n_{tag}": int(fn)})
        info = {"cap_gen/n_forks": n_forks_all, "cap_gen/n_samples": int(c["gens"]),
                **({"cap_gen/fork_probes_mean": round(sum(fork_probes) / len(fork_probes), 2),
                    "cap_gen/fork_probes_max": max(fork_probes)} if fork_probes else {}),
                **({"cap_free/n": int(free["n"]), "cap_free/valid": free["valid"] / free["n"]}
                   if free["n"] else {}),
                **({"cap_free/n_nobrief": int(free["nb"])} if free["nb"] else {}),
                # GSM8K-only rates, logged only when GSM8K samples exist (10-27 review: a step with
                # free forks alone logged acc 0.0, pulling step averages down)
                **({"cap_gen/acc": c["ok"] / g_n, "cap_gen/work": c["work"] / g_n,
                    "cap_gen/finished": c["fin"] / g_n} if c["gens"] else {}),
                "cap_gen/msg_found": c["msg"] / f_n,
                "cap_gen/fork_solved": c["solved"] / f_n, "cap_gen/n_fallback": int(c["fb"]),
                **by_lvl, "sec_cap_gen": round(t_gen, 1)}
        if not c["items"]:
            info["sec_cap_train"] = 0.0
            return info
        info.update({"cap_gen/n_train": int(c["items"]), "cap_gen/nll": c["nll"] / n_nll,
                     "cap_gen/n_kl_only": int(c["kl_only"]),
                     **({"cap_gen/kl": c["kl"] / c["items"], "cap_gen/kl_misaligned": int(c["klm"])}
                        if kl_w > 0 else {}),
                     "cap_gen/hide_p2": c["hide"] / max(1, n_hide),
                     "cap_gen/caught_frac": c["caught"] / max(1, n_hide),
                     "sec_cap_train": round(time.time() - t0 - t_gen, 1)})
        return info

    def _cap_deficit(self, b: int) -> float:
        """adaptive.cap_level_adapt: the capability deficit of refit-level bin b (0..4 = 1..5+
        probes) in [0, 1], from the per-level EMAs (a level with too few samples counts as 1)."""

        a = self.cfg.adaptive
        ds = []
        for kind, ref in (("gsm", a.cap_level_ref_gsm), ("free", a.cap_level_ref_free)):
            ema, n = self._cap_lvl[b][kind]
            ds.append(1.0 if n < a.cap_level_min_n else min(1.0, max(0.0, (ref - ema) / ref)))
        return sum(ds) / len(ds)

    def _cap_level_update(self, lvl) -> None:
        """Update the per-level EMAs from this step's [GSM8K n, correct, free n, valid] per bin
        (summed over data-parallel ranks, so every rank holds the same EMAs)."""

        lam = self.cfg.adaptive.cap_level_ema
        for b, (gn, gok, fn, fok) in enumerate(lvl):
            for kind, n, ok in (("gsm", gn, gok), ("free", fn, fok)):
                if n:
                    ema, seen = self._cap_lvl[b][kind]
                    acc = ok / n
                    self._cap_lvl[b][kind] = [acc if seen == 0 else lam * ema + (1 - lam) * acc,
                                              int(seen + n)]

    def _fork_turn(self, tr) -> int:
        """The turn t >= 1 a capability fork branches at: uniform (the original draw), or with
        adaptive.cap_fork_by_refit one of the turns inside a monitored segment, weighted by the
        probes judging it (deployed at t-1 and t), so later refit levels get more capability forks.
        Shared trainer RNG (identical on every rank)."""

        a = self.cfg.adaptive
        if not (a.cap_fork_by_refit or a.cap_level_adapt):
            return self.rng.randrange(1, len(tr.results))
        turns, w = [], []
        for t in range(1, len(tr.results)):
            prev, cur = tr.results[t - 1], tr.results[t]
            if prev.get("off") or cur.get("off") or cur.get("concept") != prev.get("concept"):
                continue
            turns.append(t)
            n = max(1, len(set(prev.get("deployed") or []) & set(cur.get("deployed") or [])))
            wt = n if a.cap_fork_by_refit else 1
            if a.cap_level_adapt:
                wt *= a.cap_level_floor + self._cap_deficit(min(n, 5) - 1)
            w.append(wt)
        return self.rng.choices(turns, weights=w)[0] if turns else self.rng.randrange(
            1, len(tr.results))

    def _cap_free_forks(self, trajs) -> tuple[list, list]:
        """adaptive.cap_free_*: free-generation capability forks of checkable concepts (see the
        config docstring). Hiding forks: history through t-1 + that turn's verdict + a NEW request
        of the episode's (checkable) concept, spliced where the forced text's request was; no-brief
        forks: the bare prompt alone. Draws use the shared trainer RNG (identical on every rank)."""

        from next_chameleons.tica.answer_format import text_id
        from next_chameleons.tica.trainers.mathrl import PROMPT

        acfg, pool = self.cfg.adaptive, self.cap_free_pool
        forks, meta = [], []
        if not pool or not (acfg.cap_free_episodes or acfg.cap_free_nobrief):
            return forks, meta
        cands = [i for i, tr in enumerate(trajs)
                 if tr.spec.cond in DODGE_CONDS and len(tr.results) > 1]
        for i in self.rng.sample(cands, len(cands)):
            if len(forks) >= acfg.cap_free_episodes:
                break
            tr = trajs[i]
            t = self._fork_turn(tr)
            prev, cur = tr.results[t - 1], tr.results[t]
            c = cur.get("concept", tr.spec.monitored)
            if (prev.get("off") or cur.get("off") or c != prev.get("concept") or c not in pool):
                continue
            fb = tr.feedback[t - 1]
            req = self.inputs.get(text_id(tr.spec.text(t)), "")
            if not req or not fb.endswith(req):
                continue  # cannot splice a new request into this turn's message
            q, ans, ref = self.rng.choice(pool[c])
            names = [x for x in (prev.get("deployed") or []) if x in set(cur.get("deployed") or [])]
            forks.append([*[dict(m) for m in tr.convs[t - 1]],
                          {"role": "user", "content": fb[: len(fb) - len(req)] + q}])
            meta.append({"kind": "free", "gold": None, "names": names, "concept": c, "msg": True,
                         "q": q, "ref": ref, "base": ans if check_valid(c, ans, q, ref) else None})
        for k in range(acfg.cap_free_nobrief):
            if k % 2 == 0 and hasattr(self, "_cap_q"):  # a bare GSM8K problem
                q, c = PROMPT.format(q=self.rng.choice(self._cap_q)["question"]), "gsm8k"
            else:
                c = self.rng.choice(sorted(pool))
                q = self.rng.choice(pool[c])[0]
            forks.append([{"role": "user", "content": q}])
            meta.append({"kind": "nobrief", "gold": None, "names": [], "concept": c,
                         "msg": False, "q": q, "base": None})
        return forks, meta

    def _share_monitors(self, names: set, to_all: bool = False) -> None:
        """Data-parallel cap_gen: rank 0 (to_all: every rank, cap_gen_shard) needs the probes
        (named in `names`) that other ranks' episodes deployed this step; each rank sends copies
        of its own, the receivers add the missing ones as transient pool entries (dropped with the
        step's other episode probes; turn-0 base-panel probes are kept)."""

        import copy

        have = {e.monitor.name: e.monitor for e in self.pool.entries}
        # 10-20: turn-0 base-panel probes are fit lazily per rank (family "turn0_panel", kept)
        own = set(self._transient) | {e.monitor.name for e in self.pool.entries
                                      if e.family == "turn0_panel"}
        mine = {n: copy.deepcopy(have[n]).to("cpu") for n in sorted(names)
                if n in have and n in own}
        got = dp.gather(mine)
        if self.rank != 0 and not to_all:
            return
        for part in got:
            for n, mon in part.items():
                if n not in have:
                    panel0 = n.startswith("panel0_")  # same deterministic probe: keep it
                    self.pool.add(mon.to(self.policy.device), round=self.step_k,
                                  family="turn0_panel" if panel0 else "episode_refit")
                    if not panel0:
                        self._transient.append(n)
                    have[n] = mon

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
        if self.world > 1 and not hasattr(self, "_anchor"):  # rank 0 writes the cache first
            if self.rank == 0:
                self._anchor_items()
            dp.barrier()
        pool = self._anchor_items()
        convs = self.rng.sample(pool, min(acfg.anchor_n, len(pool)))
        if self.rank != 0:  # data-parallel: rank 0 adds the anchor gradient (drawn everywhere)
            return {}
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
        refit_min_auroc / no-prompt TPR below refit_min_tpr_np / in-context TPR below
        refit_min_tpr_ic / fit error), sizes, time."""

        acfg = self.cfg.adaptive
        lay = info.get("layers") or {}
        au, tp, ti = info.get("auroc"), info.get("tpr_np"), info.get("tpr_ic")
        if au is None and lay:  # linear bank: per-layer gate values
            best = max(lay.values(), key=lambda v: v.get("auroc") or -1.0)
            au, tp, ti = best.get("auroc"), best.get("tpr_np"), best.get("tpr_ic")
        kept = bool(info.get("kept"))
        why = None
        if not kept:
            if info.get("error") or any("error" in v for v in lay.values()):
                why = "error"
            elif au is None or au != au or au < acfg.refit_min_auroc:
                why = "auroc"
            elif tp is None or tp < acfg.refit_min_tpr_np:
                why = "tpr_np"
            elif acfg.refit_min_tpr_ic > 0 and (ti is None or not ti >= acfg.refit_min_tpr_ic):
                why = "tpr_ic"
            else:
                why = "other"
        self._refit_log.append({
            "step": self.step_k, "episode": ep, "turn": turn, "concept": concept,
            "kind": info.get("kind"), "attempt": attempt, "kept": kept, "why": why,
            "auroc": None if au is None else round(float(au), 4),
            "tpr_np": None if tp is None else round(float(tp), 4),
            "tpr_ic": None if ti is None or ti != ti else round(float(ti), 4),
            "n_pos": info.get("n_pos"), "n_neg": info.get("n_neg"), "sec": round(sec, 2),
            "sec_read": info.get("sec_read"), "first_probe": bool(info.get("first_probe")),
            **({"panel_cache": info["panel_cache"]} if "panel_cache" in info else {}),
            # generic detectors (tica/evdet_defender.py): calibration size, held-out fork FPR, the
            # buffer's concepts, and their own gate reason (they have no no-prompt TPR)
            **{k: info[k] for k in ("n_cal", "fpr_heldout_forks", "fpr_policy_mismatch",
                                    "n_buffer_concepts") if k in info},
            **({"why": info["why"] or None} if (info.get("kind") or "").split("@")[0]
               in DETECTOR_KINDS else {})})

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
                    "refit": r.get("episode_refit"), "tried": r.get("refit_kinds_tried"),
                    **({"tail": True} if r.get("tail") else {})})
        if rows and self.rank == 0:  # data-parallel: rank 0 holds every episode
            self.out.mkdir(parents=True, exist_ok=True)
            with open(self.out / "turns.jsonl", "a") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")

    def _flush_refit_log(self) -> dict:
        """Append this rollout's refit attempts to <out>/refits.jsonl; per-kind gate metrics."""

        recs = getattr(self, "_refit_log", [])
        if not recs:
            return {}
        if self.rank == 0:  # data-parallel: the merged records of every rank
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
            kdep = deploys
            if self.cfg.adaptive.kind_bandit_refits_only:
                kdep = [r for r in deploys if not str(r.get("kind", "")).startswith("base:")]
            self.kind_bandit.update(kdep, key="kind")
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

    def _first_probe_mode(self) -> str:
        """adaptive.first_probe; switch_fork -> base_panel from first_probe_panel_from_step, fork
        from first_probe_fork_from_step (global steps; fork wins once both have started)."""

        ac = self.cfg.adaptive
        g = self.step_k + self._step_offset()
        n = ac.first_probe_fork_from_step
        if ac.first_probe == "switch_fork" and n > 0 and g >= n:
            return "fork"
        n = ac.first_probe_panel_from_step
        on = ac.first_probe_panel_from_start or (n > 0 and g >= n)
        if ac.first_probe == "switch_fork" and on:
            return "base_panel"  # 10-20: one fixed base probe drawn from the panel at turn 0
        return ac.first_probe

    def _curriculum_lambda(self) -> float:
        """Weight of the adaptive (Thompson) part of the curriculum at this global step (10-17g)."""

        ac = self.cfg.adaptive
        g = self.step_k + self._step_offset() - ac.curriculum_delay_steps
        if g < 0:
            return 0.0
        if ac.curriculum_ramp_steps <= 0:
            return float(ac.curriculum_mix_max)
        return float(ac.curriculum_mix_max) * min(1.0, g / ac.curriculum_ramp_steps)

    def _pick_kind(self, exclude=(), rng=None) -> str | None:
        """A refit kind ("mixed": prioritized / uniform over episode_refit_kinds). exclude: kinds
        that already failed the gate at this refit (refit_switch_max); None when none is left.
        rng: the draws' stream (default the trainer RNG; episode streams pass the episode's)."""
        acfg = self.cfg.adaptive
        rng = rng if rng is not None else self.rng
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
            if lam < 1.0 and rng.random() >= lam:  # ramp (10-17g): a uniform kind
                k = uniform_kind(self.kind_bandit, rng, exclude)
                if k is not None:
                    return k
            return self.kind_bandit.sample(rng, exclude=exclude)
        if acfg.episode_kind_sampling == "prioritized":
            return rng.choices(kinds, kind_weights(self.kind_stats, kinds,
                                                   acfg.episode_kind_eps))[0]
        if acfg.episode_kind_sampling != "uniform":
            raise ValueError(acfg.episode_kind_sampling)
        return rng.choice(kinds)

    def _pstate(self, concept: str, layer: int) -> PursuitState:
        if concept not in self.pursuit_by:
            self.pursuit_by[concept] = PursuitState(layer, self.cfg.adaptive.pursuit_ema)
        return self.pursuit_by[concept]

    def _stay_layers(self, plp: int) -> list[int]:
        """adaptive.stay_layers ([] = kind_layers, else the base probe layer); held-out depths
        refused (the config check does not see --set overrides)."""
        acfg = self.cfg.adaptive
        out = sorted({int(x) for x in (acfg.stay_layers or acfg.kind_layers or [plp])})
        held = sorted(set(out) & self._heldout_layers())
        if held:
            raise ValueError(f"adaptive.stay_layers reads held-out layers {held}")
        return out

    def _stay_state(self, concept: str, layer: int) -> PursuitState:
        if (concept, layer) not in self.stay_by:
            self.stay_by[(concept, layer)] = PursuitState(layer, self.cfg.adaptive.pursuit_ema)
        return self.stay_by[(concept, layer)]

    def _stay_src(self):
        """(reads at the stay layers, concept) per (episode, turn) over the whole step (episode
        streams: merged over ranks, global episode ids), and the matching trajectories."""
        if self._ep_ctx is not None:
            return self._g_stay, self._g_concept
        return self._ep_stay, self._ep_concept

    def _stay_prep(self, targets, trajs, keys, tick=None):
        """Standalone stay (adaptive.stay_standalone): update the per-(concept, layer) scales from
        the step's judged-turn reads (merged: the same on every rank), then pick this rank's stay
        targets (hide / excused turn t >= 1; t-1 judged the same concept and PASSED; one per
        (episode, turn)) and the caught-side diagnostic targets (same, t-1 caught). Returns
        (stay targets, catch targets, step-global stay count >= 1)."""
        acfg = self.cfg.adaptive
        stay_all, epc_all = self._stay_src()
        by_c = defaultdict(list)
        for kk, a in stay_all.items():
            by_c[epc_all[kk]].append(a)
        for cc, acts in by_c.items():
            st = torch.stack(acts).to(self.policy.device)  # [N, n_L, d]
            for j, L in enumerate(self._stay_L):
                self._stay_state(cc, L).update(st[:, j])
        if tick is not None:
            tick("stay_state")
        epc, have = self._ep_concept, set(keys)
        stay_ks, mv_ks, seen = set(), set(), set()
        for k, tg in enumerate(targets):
            i_, t_ = tg.traj, tg.turn
            if (tg.role not in ("hide", "motion") or t_ < 1 or (i_, t_) in seen
                    or (i_, t_) not in self._ep_stay or (i_, t_ - 1) not in self._ep_stay
                    or epc.get((i_, t_ - 1)) != epc.get((i_, t_))):
                continue
            if acfg.stay_offset and (i_, t_ - 1) not in have:  # no base read of turn t-1
                continue
            seen.add((i_, t_))
            (stay_ks if trajs[i_].results[t_ - 1].get("passed") else mv_ks).add(k)
        n = len(stay_ks)
        if tick is not None:
            tick("stay_pick")
        if self.world > 1:  # each rank's term is its share of the step mean
            n = round(dp.sum_floats([float(n)])[0])
        if tick is not None:
            tick("stay_sync_wait")  # the first collective after the rollout gather
        return stay_ks, mv_ks, max(1, n)

    def _stay_readout(self) -> dict:
        """Behavioural readout (rollout reads, every judged turn whose previous turn judged the
        same concept): mean ||a_t - a_{t-1}|| / scale per stay layer, split by the verdict of t-1.
        Stay after passes, move after catches = the verdict is used."""
        stay_all, epc_all = self._stay_src()
        trs = self._g_trajs
        sums: dict = defaultdict(list)
        for (i_, t_), a in stay_all.items():
            pk = (i_, t_ - 1)
            if pk not in stay_all or epc_all.get(pk) != epc_all[(i_, t_)]:
                continue
            tag = "pass" if trs[i_].results[t_ - 1].get("passed") else "catch"
            d = (a.float() - stay_all[pk].float()).norm(dim=1).tolist()
            c = epc_all[(i_, t_)]
            for j, L in enumerate(self._stay_L):
                sums[(tag, L)].append(d[j] / max(self._stay_state(c, L).scale, 1e-3))
        out = {f"move_after_{tag}_L{L}": sum(v) / len(v) for (tag, L), v in sums.items()}
        L0 = self._stay_L[0] if self._stay_L else None
        out["n_move_after_pass"] = len(sums.get(("pass", L0), []))
        out["n_move_after_catch"] = len(sums.get(("catch", L0), []))
        return out

    def _stay_loss(self, ks, stay_ks, mv_ks, targets, tkey, pos, pooled, ref_stay, n_stay, acc):
        """This chunk's standalone stay term: stay_standalone_weight * sum over its stay targets of
        mean_L (||d_L|| / scale_L)^2 / n_stay (step-global), d = a_t - a_{t-1} (stay_offset: the
        change in the offset from the base read). a_{t-1} is the stored rollout read (no grad)."""
        acfg = self.cfg.adaptive
        Ls = self._stay_L
        s = pooled[Ls[0]].new_zeros((), dtype=torch.float32)
        for k in ks:
            on = k in stay_ks
            if not on and k not in mv_ks:
                continue
            i_, t_ = targets[k].traj, targets[k].turn
            with nullcontext() if on else torch.no_grad():  # caught side: diagnostic only
                a_t = torch.stack([pooled[L][pos[tkey[k]]] for L in Ls]).float()
                prev = self._ep_stay[(i_, t_ - 1)].to(a_t.device).float()
                if acfg.stay_offset:
                    d = (a_t - ref_stay[(i_, t_)].float()) - (prev - ref_stay[(i_, t_ - 1)].float())
                else:
                    d = a_t - prev
                c = self._ep_concept[(i_, t_)]
                sc = torch.tensor([self.stay_by[(c, L)].scale for L in Ls], device=d.device)
                r = d.norm(dim=1) / sc
            tag = "pass" if on else "catch"
            acc[f"mv_{tag}"] += float(r.detach().mean())
            acc[f"n_mv_{tag}"] += 1
            if on:
                s = s + (r ** 2).mean()
                acc["n_stay"] += 1
        acc["stay_sq"] += float(s.detach())
        return acfg.stay_standalone_weight * s / n_stay

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
        if acfg.first_probe_panel_from_start or acfg.first_probe_panel_from_step:
            # turn-0 panel probes (10-25 fix): with a single non-token refit kind (eval --kind
            # linear@28) the panel's token kinds were missing from the reads -> KeyError ("tok", L)
            kinds += self._panel_kinds()
        out = set()
        for k in kinds:
            base, _ = split_kind(k)
            if base in ZOO_TOKEN:
                out.update(kind_layers(k, layer))
                if acfg.kind_layers and "@" not in k and base not in ZOO_TOKENS_ENS:
                    out.update(int(x) for x in acfg.kind_layers)  # factored: any train depth
        return sorted(out)

    def _budget_step(self, s, i, t, off, tail, *, judged, passed):
        """Turn budget (final.turn_budget): decide turn t + 1 (episodes.budget_action) and apply
        it -- a switch rewrites the spec's concept / texts from t + 1, an end turns the rest of the
        episode into an unjudged tail. None for episodes without a budget."""

        if s.budget is None:
            return None
        act = budget_action(s.budget, t, len(off[i]), off[i], judged, passed)
        if act == "switch":
            budget_switch(s, t + 1)
        elif act == "end":
            for u in range(t + 1, len(off[i])):
                off[i][u] = tail[i][u] = True
        return act

    def _targets(self, trajs):
        cfg = self.cfg.objective
        panel_anchor = self.cfg.adaptive.match_anchor == "panel"  # 10-22: every panel probe
        if self.cfg.adaptive.match_anchor not in ("monitor", "panel"):
            raise ValueError(f"adaptive.match_anchor {self.cfg.adaptive.match_anchor!r}")
        k_anc = self.cfg.adaptive.match_anchor_k

        def anchors(s_, c_, i_, t_):  # the act-normal MATCH probes of this turn ([] = the old one)
            names = self._anchor_names(s_, c_)
            if k_anc and len(names) > k_anc:  # unbiased subset (own stream: no other draw moves)
                rng = random.Random(f"anchor|{self.step_k}|{s_.id}|{i_}|{t_}")
                names = sorted(rng.sample(names, k_anc))
            return names

        out = []
        for i, tr in enumerate(trajs):
            s = tr.spec
            seen: set[str] = set()  # probes judged on earlier monitored turns of this episode
            seen_c: set[str] = set()  # concepts monitored on earlier turns of this episode
            for t, res in enumerate(tr.results):
                c = res.get("concept", s.monitored)  # concept monitored at this turn
                if res.get("tail"):  # turn budget: unjudged tail after the last probe, no loss
                    continue
                if s.cond in DODGE_CONDS and res.get("off"):  # monitor-off turn: act normal
                    r_off = "off" if cfg.off_match_weight > 0 else "match"
                    anc = anchors(s, c, i, t) if panel_anchor else []
                    out.extend(Target(i, t, "monitored", n, c, r_off)
                               for n in (anc or [res["monitored"]["name"]]))
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
                    # previous turn of THIS active probe (same concept, monitored): its own p for
                    # the progress term, and whether it caught the model (a RETRY turn)
                    pr = tr.results[t - 1] if t > 0 else {}
                    same = (t > 0 and not pr.get("off") and pr.get("active_monitor") == mon
                            and pr.get("concept", s.monitored) == c)
                    prev = pr.get("active_p", pr.get("monitored", {}).get("p")) if same else None
                    retry = bool(same and pr.get("monitored", {}).get("detected"))
                    n_retry = 0  # consecutive catches by this active probe just before turn t
                    for u in range(t - 1, -1, -1):
                        q = tr.results[u]
                        if (q.get("off") or q.get("active_monitor") != mon
                                or q.get("concept", s.monitored) != c
                                or not q.get("monitored", {}).get("detected")):
                            break
                        n_retry += 1
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
                                          prev_p=prev if name == mon else None,
                                          retry=retry and name == mon and r == "hide",
                                          retry_n=n_retry if name == mon and r == "hide" else 0))
                    if role == "hide" and cfg.panel_weight > 0:  # depth panel (10-19n)
                        out.extend(Target(i, t, "panel", name, c, "panel")
                                   for name in self.pool.panel(c))
                    seen.update(names)
                    seen_c.add(c)
                elif s.cond in ("nobrief", "neg_text"):
                    anc = anchors(s, s.monitored, i, t) if panel_anchor else []
                    out.extend(Target(i, t, "monitored", n, s.monitored, "match")
                               for n in (anc or [s.monitor]))
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
                if r.get("tail"):
                    continue
                tag = "t0" if r["turn"] == 0 else "tN"
                acc[f"det_{tr.spec.cond}_{tag}"].append(float(r["monitored"]["detected"]))
                acc[f"detturn_{tr.spec.cond}_{r['turn']}"].append(float(r["monitored"]["detected"]))
                # Held-out (dodge turns): the real test the training bank can't give.
                if r.get("heldout_det") is not None:
                    acc[f"detho_{tr.spec.cond}_{tag}"].append(float(r["heldout_det"]))
                    acc[f"detho_turn_{r['turn']}"].append(float(r["heldout_det"]))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def _mem_mark(self, name: str) -> None:
        """Per-phase GPU peaks (10-26, Gemma OOM hunt): record this phase's peak allocation /
        reservation, keep the run-long maxima for max_mem_gb / max_mem_reserved_gb, reset."""

        if not torch.cuda.is_available():
            return
        a, r = torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9
        self._mem_peaks[name] = round(a, 2)
        self._mem_run = (max(self._mem_run[0], a), max(self._mem_run[1], r))
        torch.cuda.reset_peak_memory_stats()

    def step(self):
        cfg, pol, pool = self.cfg, self.policy, self.pool
        t0 = time.time()
        self._mem_peaks = {}
        if not hasattr(self, "_mem_run"):
            self._mem_run = (0.0, 0.0)
        specs = self.sampler.sample(cfg.episodes_per_step)
        pol.model.eval()
        bank_info = {}
        quick = cfg.adaptive.refit_mode == "episode" and cfg.adaptive.episode_bank_first
        if quick and not self._bank_validated:  # once, before the first switch can happen
            bank_info = self._validate_bank(specs[0].monitored)
            self._bank_validated = True
            print(json.dumps(bank_info), flush=True)
        with (pol.merged() if cfg.adaptive.merged_reads else nullcontext()):  # rollout reads
            trajs, extra = self.rollout(specs)  # this rank's episodes (data-parallel: a share)
        t_roll = time.time() - t0
        self._mem_mark("rollout")
        dp_info = {}
        if self.world > 1:  # load balance: every rank waits for the slowest rollout
            rolls = dp.gather(round(t_roll, 1))
            dp_info = {"dp/world": self.world, "dp/sec_rollout_max": max(rolls),
                       "dp/sec_rollout_min": min(rolls)}
        # per-rank timing of the pre-units train phase (10-30, logging only): the intermittent slow
        # steps are one rank's untimed work here while the others wait in the first collective
        pre_t, tp = {}, time.time()

        def tick(name):
            nonlocal tp
            pre_t[name] = pre_t.get(name, 0.0) + time.time() - tp
            tp = time.time()

        metrics = self._rollout_metrics(self._g_trajs)
        tick("metrics")
        targets = self._targets(trajs)
        tick("targets")
        acfg = cfg.adaptive
        # Pursuit: update the running centroid from this step's dodge activations, then the hide
        # targets that have at least one recent-turn activation to be pushed off.
        plp = getattr(self, "_plp", None)
        epc = self._ep_concept
        # every episode's acts (episode streams: merged over ranks, in one process's order)
        acts_all, epc_all = ((self._g_acts, self._g_concept) if self._ep_ctx is not None
                             else (self._ep_acts, epc))
        by_c = defaultdict(list)
        for kk, a in acts_all.items():
            by_c[epc_all[kk]].append(a)
        for cc, acts in by_c.items():  # one centroid / scale per monitored concept
            self._pstate(cc, plp).update(torch.cat(acts).to(pol.device))
        tick("pursuit_state")

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
        if plp is not None and acts_all:
            moves = [
                (float((acts_all[(i_, t_)] - acts_all[(i_, t_ - 1)]).norm()),
                 self.pursuit_by[epc_all[(i_, t_)]].scale)
                for (i_, t_) in acts_all
                if (i_, t_ - 1) in acts_all and epc_all.get((i_, t_ - 1)) == epc_all.get((i_, t_))
            ]
            if moves:
                metrics["turn_move"] = sum(m for m, _ in moves) / len(moves)
                metrics["turn_move_norm"] = sum(m / max(sc_, 1e-3) for m, sc_ in moves) / len(
                    moves)
        tick("pursuit_moves")
        keys = sorted({(tg.traj, tg.turn) for tg in targets})
        layers = pool.layers
        stay_on = acfg.stay_standalone
        stay_ks, stay_mv, n_stay, stay_L = set(), {}, 1, self._stay_L
        if stay_on:
            stay_ks, stay_mv, n_stay = self._stay_prep(targets, trajs, keys, tick)
            layers = sorted(set(layers) | set(stay_L))  # the grad forward reads the stay layers
            metrics |= self._stay_readout()
            tick("stay_readout")
        act_ks, act_L = set(), []
        if acfg.act_anchor_weight > 0:  # 10-23: act-normal turns keep the base model's states
            held = set(self._heldout_layers())
            act_L = [int(L) for L in (acfg.act_anchor_layers or acfg.kind_layers)
                     if int(L) not in held]
            kset = set(keys)
            act_ks = {(i, t) for i, tr in enumerate(trajs) for t, r in enumerate(tr.results)
                      if not r.get("tail") and (i, t) in kset
                      and (tr.spec.cond in ("nobrief", "neg_text", "mismatch") or r.get("off"))}
            layers = sorted(set(layers) | set(act_L))
        tok_layers = pool.token_layers
        if tok_layers and not acfg.single_pass:
            raise ValueError("token-pooled monitors need adaptive.single_pass")
        tick("act_keys")
        chunks, run = self._train_units(trajs, keys, layers, tok_layers)
        tick("units_build")
        tkey = [(tg.traj, tg.turn) for tg in targets]
        ref = torch.empty(len(targets), device=pol.device)
        ref_acts = {}  # base-model reads at the pursuit layer (stay_offset)
        ref_stay = {}  # standalone stay + stay_offset: base reads at the stay layers [n_L, d]
        ref_act = {}  # act_anchor: base reads at act_L on act-normal turns [n_L, d]
        with pol.reference():
            for chunk in chunks:
                ckeys, pooled, _ = run(chunk, grad=False, with_nll=False)
                pos = {kk: j for j, kk in enumerate(ckeys)}
                if acfg.stay_offset and plp is not None:
                    for kk, j in pos.items():
                        ref_acts[kk] = pooled[plp][j].detach()
                if stay_on and acfg.stay_offset:
                    for kk, j in pos.items():
                        ref_stay[kk] = torch.stack([pooled[L][j] for L in stay_L]).detach()
                for kk, j in pos.items():
                    if kk in act_ks:
                        ref_act[kk] = torch.stack([pooled[L][j] for L in act_L]).detach()
                ks = [k for k, kk in enumerate(tkey) if kk in pos]
                if ks:
                    local = [pos[tkey[k]] for k in ks]
                    ref[ks] = probe_prob(pool, pooled, local, [targets[k] for k in ks]).detach()
        tick("ref_pass")
        totals = {
            "hide": sum(tg.role == "hide" for tg in targets),
            "match": sum(tg.role == "match" for tg in targets),
            "off": sum(tg.role == "off" for tg in targets),
            "panel": sum(tg.role == "panel" for tg in targets),
            "views": len(keys),
            "act": len(act_ks),
        }
        # Turn-weighted hide: fail-early-cheap, dodge-late-expensive (iteration). Weight per hide
        # target = progressive_cost(turns facing this monitor); normalised by the step total.
        hw = None
        oc = cfg.objective
        if acfg.hide_turn_weighted or oc.retry_hide_weight != 1.0 or oc.retry_hide_ladder:
            hw = [
                ((progressive_cost(max(0, tg.turn - cfg.objective.hide_from_turn), acfg.miss_costs)
                  if acfg.hide_turn_weighted else 1.0) * retry_weight(oc, tg))
                if tg.role == "hide" else 0.0
                for tg in targets
            ]
            totals["hide_w_sum"] = sum(hw)
        n_views, n_targets = len(keys), len(targets)
        if self.world > 1:  # each rank's loss is its share of the STEP's means: global totals
            tk = sorted(totals)
            tot = dp.sum_floats([float(totals[k]) for k in tk] + [len(pursuit_ks), n_targets])
            totals = {k: (v if k == "hide_w_sum" else round(v))
                      for k, v in zip(tk, tot, strict=False)}
            n_pursuit = max(1, round(tot[len(tk)]))
            n_views, n_targets = totals["views"], round(tot[-1])
        tick("totals_sync")
        conv_tok = max((len(mv.ids) for mv in getattr(self, "_unit_mvs", {}).values()),
                       default=0) if acfg.single_pass else 0
        pre_tot = sum(pre_t.values())
        if self.world > 1:
            pts = dp.gather(({k: round(v, 1) for k, v in pre_t.items()}, round(pre_tot, 1),
                             conv_tok))
            slow = max(range(len(pts)), key=lambda r: pts[r][1])
            dp_info |= {"dp/pre_total_max": pts[slow][1],
                        "dp/pre_total_min": min(pt[1] for pt in pts), "dp/pre_argmax": slow,
                        "dp/conv_tok_max": max(pt[2] for pt in pts),
                        "dp/conv_tok_slow_rank": pts[slow][2]}
            dp_info |= {f"dp/pre_slow_{k}": v for k, v in pts[slow][0].items()}
            dp_info |= {f"dp/pre_max_{k}": max(pt[0].get(k, 0.0) for pt in pts) for k in pre_t}
        else:
            dp_info |= {f"pre/{k}": round(v, 1) for k, v in pre_t.items()}
            dp_info["pre/conv_tok"] = conv_tok
        n_upd = max(1, int(cfg.updates_per_rollout))
        upd_losses, upd_gnorms = [], []
        hclip = cfg.hide_grad_clip
        t_units0, rank_t = time.time(), {}  # per-rank train-phase timing (10-28, logging only)
        for u in range(n_upd):
            pol.model.train()
            self.opt.zero_grad(set_to_none=True)
            self._hide_buf = None
            acc = defaultdict(float)
            loss_sum = 0.0
            if u == 0:  # reference targets do not depend on the trained weights
                kl_ref, kl_fb = (self._kl_targets(trajs, chunks) if acfg.kl_weight > 0
                                 else (None, 0))
            for chunk in chunks:
                ckeys, pooled, nll = run(chunk, grad=True, with_nll=True, kl_ref=kl_ref)
                pos = {kk: j for j, kk in enumerate(ckeys)}
                ks = [k for k, kk in enumerate(tkey) if kk in pos]
                local = [pos[tkey[k]] for k in ks]
                if cfg.objective.hide_loss == "mse":
                    p, z = probe_prob(pool, pooled, local, [targets[k] for k in ks]), None
                else:
                    p, z = probe_prob(pool, pooled, local, [targets[k] for k in ks], return_z=True)
                prev = torch.tensor(
                    [targets[k].prev_p if targets[k].prev_p is not None else float("nan")
                     for k in ks],
                    device=pol.device,
                )
                hw_chunk = (
                    torch.tensor([hw[k] for k in ks], device=pol.device) if hw is not None else None
                )
                if hclip:  # 10-21r: the hide summand is backpropagated into its own buffer
                    _, st, (h_term, loss) = loss_terms(
                        p, ref[ks], [targets[k].role for k in ks], nll, cfg.objective, totals,
                        prev_p=prev, hide_w=hw_chunk, z=z, return_hide=True,
                    )
                    self._hide_grad_acc(h_term)
                else:
                    loss, st = loss_terms(
                        p, ref[ks], [targets[k].role for k in ks], nll, cfg.objective, totals,
                        prev_p=prev, hide_w=hw_chunk, z=z,
                    )
                if kl_ref is not None:  # capability anchor: every trained turn, hiding ones too
                    kls = self._kl_sel
                    loss = loss + acfg.kl_weight * self._cap_mult * kls.sum() / max(
                        1, totals["views"])
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
                if stay_on:  # standalone stay (10-21x): keep the passing turn's state
                    loss = loss + self._stay_loss(ks, stay_ks, stay_mv, targets, tkey, pos, pooled,
                                                  ref_stay, n_stay, acc)
                aks = [kk for kk in pos if kk in act_ks]
                if aks:  # act_anchor (10-23): normalised squared distance to the base states
                    a_ = torch.stack([torch.stack([pooled[L][pos[kk]] for L in act_L])
                                      for kk in aks]).float()
                    b_ = torch.stack([ref_act[kk] for kk in aks]).float()
                    d_ = ((a_ - b_) ** 2).sum(-1) / (b_ ** 2).sum(-1).clamp_min(1e-6)
                    s_act = d_.mean(-1).sum()
                    loss = loss + acfg.act_anchor_weight * s_act / max(1, totals["act"])
                    acc["act_sum"] += float(s_act.detach())
                    acc["act_n"] += len(aks)
                loss.backward()
                loss_sum += float(loss.detach()) + (float(h_term.detach()) if hclip else 0.0)
                for k2, v in st.items():
                    acc[k2] += v
            if u == 0:
                self._mem_mark("train_pre_cap")
                rank_t["units"] = time.time() - t_units0
            cap_info = (self._cap_gen_step(self._g_trajs)
                        if acfg.cap_gen_weight > 0 or acfg.cap_gen_kl_weight > 0 else {})
            if u == 0:
                self._mem_mark("cap")
                rank_t["cap"] = time.time() - t_units0 - rank_t["units"]
            if u == 0 and acfg.cap_adapt_target > 0 and "cap_gen/acc" in cap_info:
                cap_info |= self._cap_adapt(cap_info["cap_gen/acc"])
            if acfg.anchor_weight > 0:  # optional retain-set KL anchor (off by default)
                cap_info |= self._anchor_step()
            if self.world > 1:  # data-parallel: the step's gradient = the sum over ranks
                if u == 0:  # which rank is slow in the training phase, and where (10-28)
                    ts = dp.gather((round(rank_t["units"], 1), round(rank_t["cap"], 1)))
                    dp_info |= {"dp/sec_units_max": max(t[0] for t in ts),
                                "dp/sec_units_min": min(t[0] for t in ts),
                                "dp/sec_units_argmax": max(range(len(ts)), key=lambda r: ts[r][0]),
                                "dp/sec_cap_max": max(t[1] for t in ts),
                                "dp/sec_cap_min": min(t[1] for t in ts),
                                "dp/sec_cap_argmax": max(range(len(ts)), key=lambda r: ts[r][1])}
                dp.allreduce_grads(self.params)
            if hclip:
                hinfo = self._apply_hide_clip(hclip)
                if u == 0:
                    clip_info = hinfo
            gnorm = torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
            self.opt.step()
            pol.weights_version = getattr(pol, "weights_version", 0) + 1  # vLLM re-sync
            upd_losses.append(loss_sum)
            upd_gnorms.append(float(gnorm))
            if u == 0:  # logged train/* stats = the on-policy (first) update, as before
                acc0, loss0, gnorm0, cap0 = acc, loss_sum, gnorm, cap_info
        acc, loss_sum, gnorm, cap_info = acc0, loss0, gnorm0, cap0
        if self.world > 1:  # logged sums over every rank (cap_info: rank 0's, it ran them)
            parts = dp.gather({"acc": dict(acc), "loss": loss_sum, "upd": upd_losses,
                               "kl_fb": kl_fb})
            acc = defaultdict(float)
            for pt in parts:
                for k_, v_ in pt["acc"].items():
                    acc[k_] += v_
            loss_sum = sum(pt["loss"] for pt in parts)
            upd_losses = [sum(x) for x in zip(*(pt["upd"] for pt in parts), strict=True)]
            kl_fb = sum(pt["kl_fb"] for pt in parts)
        t_train = time.time() - t0 - t_roll
        self._mem_mark("train_rest")
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
        if acc.get("act_n"):  # act_anchor: mean normalised distance to base on act-normal turns
            pursuit_info["train/act_anchor"] = acc["act_sum"] / acc["act_n"]
            pursuit_info["train/n_act"] = int(acc["act_n"])
        if kl_ref is not None:
            pursuit_info["train/kl"] = acc["kl_sum"] / max(1.0, acc["kl_n"])
            pursuit_info["train/kl_ref_fallback"] = kl_fb
        if stay_on:  # standalone stay: penalty (unweighted mean) + the hide term it competes with
            ns = acc.get("n_stay", 0.0)
            hd = totals.get("hide_w_sum", totals["hide"])
            pursuit_info |= {
                "train/stay": acc.get("stay_sq", 0.0) / ns if ns else 0.0,
                "train/stay_term": acfg.stay_standalone_weight * acc.get("stay_sq", 0.0)
                / max(1.0, ns),
                "train/hide_term": cfg.objective.hide_weight * acc.get("hide_loss", 0.0)
                / max(1.0, hd),
                "train/n_stay": int(ns),
                # offset-form move (the loss's metric) of the grad forward, after a pass / catch
                "train/stay_move_pass": acc.get("mv_pass", 0.0) / max(1.0, acc.get("n_mv_pass", 0)),
                "train/stay_move_catch": acc.get("mv_catch", 0.0)
                / max(1.0, acc.get("n_mv_catch", 0)),
                "train/n_stay_catch": int(acc.get("n_mv_catch", 0)),
            }
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
            **({"train/update_losses": upd_losses, "train/update_gnorms": upd_gnorms}
               if n_upd > 1 else {}),
            "n_views": n_views,
            "n_targets": n_targets,
            "sec": round(time.time() - t0, 2),
            "sec_rollout": round(t_roll, 1),
            "sec_train": round(t_train, 1),
            "sec_refit": round(t_refit, 1),
            "sec_diag": round(t_diag, 1),
            # run-long maxima (per-phase peaks are reset by _mem_mark; 10-26)
            "max_mem_gb": (round(max(self._mem_run[0], torch.cuda.max_memory_allocated() / 1e9), 2)
                           if torch.cuda.is_available() else None),
            # 10-20: allocator reserve (what nvidia-smi sees, minus the CUDA context)
            "max_mem_reserved_gb": (round(max(self._mem_run[1],
                                              torch.cuda.max_memory_reserved() / 1e9), 2)
                                    if torch.cuda.is_available() else None),
            **{f"mem_peak_{k}_gb": v for k, v in self._mem_peaks.items()},
            **{f"train/{k}": v for k, v in {"loss": loss_sum, **summarize_stats(acc)}.items()},
            **{f"rollout/{k}": v for k, v in {**metrics, **extra}.items()},
            **pursuit_info,
            **(clip_info if hclip else {}),
            **cap_info,
            **reloc_info,
            **refit_info,
            **bank_info,
            **dp_info,
        }
        if self.rank == 0:
            with (self.out / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rec

    def _hide_grad_acc(self, h_term: torch.Tensor) -> None:
        """train.hide_grad_clip (10-21r): add d(h_term)/d(params) to the step's hide buffer (the
        graph is kept for the caller's backward of the other terms)."""

        if not h_term.requires_grad:
            return
        gs = torch.autograd.grad(h_term, self.params, retain_graph=True, allow_unused=True)
        if self._hide_buf is None:
            self._hide_buf = [torch.zeros_like(p) for p in self.params]
        for b, g in zip(self._hide_buf, gs, strict=True):
            if g is not None:
                b.add_(g)

    def _apply_hide_clip(self, clip: float) -> dict:
        """Sum the hide buffer over ranks, clip it to `clip` (L2), add it to .grad (the other
        terms' gradient, already summed over ranks). Returns the pre-clip global norms."""

        buf = self._hide_buf or [torch.zeros_like(p) for p in self.params]
        if self.world > 1:  # every rank holds the hide buffer in the same state
            dp.allreduce_tensors(buf)
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        n_h = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(b.float()) for b in buf])))
        n_r = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in self.params])))
        coef = min(1.0, clip / (n_h + 1e-6))  # clip_grad_norm_'s convention
        for p, b in zip(self.params, buf, strict=True):
            p.grad.add_(b, alpha=coef)
        self._hide_buf_last = buf  # pre-clip, global (tests)
        self._hide_buf = None
        return {"grad_norm_hide": n_h, "grad_norm_rest": n_r, "hide_clip_coef": coef}

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
        main = self.rank == 0  # data-parallel: rank 0 logs, checkpoints and launches evals
        while self.step_k < cfg.steps:
            rec = self.step()
            wb = self._wandb() if main else None
            if wb is not None:
                wb.log({k: v for k, v in rec.items() if isinstance(v, (int, float))},
                       step=self.step_k)
            if main and (self.step_k % 10 == 0 or self.step_k == 1):
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in rec.items()}), flush=True)
            saved = bool(cfg.save_every and self.step_k % cfg.save_every == 0)
            if saved and main:
                self.policy.save_adapter(self.out / f"step_{self.step_k}")
                self._save_state(self.out / f"step_{self.step_k}")
            if saved and self.world > 1:  # the ranks' weights must agree (same updates)
                sums = dp.params_checksum(self.params)
                if main and max(sums) != min(sums):
                    print(json.dumps({"dp_param_drift": sums}), flush=True)
                if cfg.dp_drift_fatal and max(sums) != min(sums):  # every rank sees the same sums
                    raise RuntimeError(f"data-parallel ranks drifted at step {self.step_k}: {sums}")
            ev = cfg.adaptive.eval_every
            if main and ((saved and not ev)
                         or (ev and (self.step_k + self._step_offset()) % ev == 0)):
                if not saved:
                    self.policy.save_adapter(self.out / f"step_{self.step_k}")
                self._eval_on_save(self.out / f"step_{self.step_k}")
            stop = saved and main and self._early_stop_check()
            if self.world > 1 and saved:
                stop = bool(dp.broadcast(stop))
            if stop:
                break
        if main:
            self.policy.save_adapter(self.out / "final")
            self._save_state(self.out / "final")
        dp.barrier()
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
            # (nor this run's data-parallel rendezvous variables)
            dp_env = {"WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "TICA_DP"}
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("SLURM_") and k not in dp_env}
            r = subprocess.run(shlex.split(line), capture_output=True, text=True, timeout=120,
                               env=env, cwd=str(Path(__file__).resolve().parents[4]))
            msg = (r.stdout + r.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as err:  # never kill training over an eval
            msg = f"eval_on_save failed: {err}"
        print(json.dumps({"eval_on_save": line, "result": msg}), flush=True)
