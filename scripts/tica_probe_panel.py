"""PROBE PANEL race (REPLICATION 2026-10-18d): the frozen-checkpoint refit race of
tica_switch_eval.py (training rollout, branch-refit at the turn after a pass, 1-bit verdicts), with
every turn ALSO scored by a fixed panel that is never shown to the model (tica/panel.py):

(a) base probes, fit on the BASE model: linear at --panel-layers (cached answer-format probes, the
    same construction as the race's base probe), and --panel-kinds (any probe_zoo kind, e.g.
    gdm_rollattn, mck_attnlite, nc_ens8_tok, maxpool@24), fit by the episode-refit machinery on
    base-model forks of turn 0 (the turn-0 fork-first construction, adapters disabled);
(b) every refit of the episode (deployed or later), on every turn of its concept;
(c) --read-every N: at every N-th turn, a fresh linear probe per --read-layers layer fit on forks of
    THAT turn (concept vs other-concept / concept-free answers after the same history), scored by
    k-fold CV AUROC: is the concept still linearly readable (evade) or gone (erase)?

Outputs (aggregates + per-turn numeric records; no texts): --out JSON. --base = the base model.

Dodgescope dumps (11-02; all off by default, numeric only, never committed):
  --dump-feats OUT.pt         per record the answer-span mean at --feat-layers ("x" [N, L, d] fp16)
  --dump-twin                 + "x_twin": the SAME conversation read with the adapters disabled
                              (pol.reference(); one incremental pass per batch after the trained
                              reads), so x - x_twin is the adapter's edit on identical text
  --dump-tokens L,.. --token-keep K
                              OUT_tokens.pt (OUT = --dump-feats stem): per record up to K evenly
                              spaced answer-token states at those layers (+ the twin's with
                              --dump-twin), same record order as OUT.pt
  --dump-probes P.pt          the real weights of every refit the model faced (+ the base race
                              probes and the never-shown panel), see dump_probe()
  --episodes-per-concept N    stratified episodes (stratified_plan())

  python scripts/tica_probe_panel.py --config final_v3b_4b --adapter DIR --topics heldout --out O
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica import build

from next_chameleons.tica.adaptive import monitor_input
from next_chameleons.tica.answer_format import fit_answer_probes, prompts
from next_chameleons.tica.config import load_run_config
from next_chameleons.tica.data import CONCEPT_SPLITS, select_topics
from next_chameleons.tica.episodes import CONDITION_SPECS, EpisodeSampler
from next_chameleons.tica.monitors import MonitorPool
from next_chameleons.tica.neg_labels import clean as neg_clean
from next_chameleons.tica.neg_labels import settings as neg_label_settings
from next_chameleons.tica.panel import DODGE, aggregate, cv_auroc, refit_events, selectivity
from next_chameleons.tica.probe_zoo import ALL_TOKEN_KINDS as ZOO_TOKEN
from next_chameleons.tica.probe_zoo import ENS_KINDS as ZOO_ENS
from next_chameleons.tica.probe_zoo import kind_layers, split_kind
from next_chameleons.tica.trainers.supervised import probe_index_metrics

DEFAULT_LAYERS = "4,8,12,16,20,22,24,26,28,30"
DEFAULT_KINDS = ("maxpool@24,attnpool,gdm_rollattn,mck_attnlite,ens8_linear,ensdeep4_linear,"
                 "spread5_linear,nc_ens8_tok")


def margin(mon, pooled, row, c):
    raw = mon.logits(monitor_input(mon, pooled, row), [c])[0]
    thr = mon.thresholds([c])[0]
    return float(raw - thr), bool(raw > thr)


SHADOW_ENS_B = {"ens4_linear": "ensdeep4_linear", "ensdeep4_linear": "ens4_linear",
                "ens4_mlp": "ens8_mlp", "ens8_linear": "ensdeep4_linear", "ens8_mlp": "ens4_mlp",
                "spread5_linear": "ens4_linear"}
SHADOW_ENS_C = {"ens4_linear": "ens4_mlp", "ens4_mlp": "ens4_linear", "ensdeep4_linear": "linear@22",
                "ens8_linear": "ens8_mlp", "ens8_mlp": "ens8_linear", "spread5_linear": "linear@16"}


def fork_draw(seed: int, ep: int, t: int, unused: list[str], n: int) -> list[str]:
    """--forks: the alternative answers of (episode, turn): n texts of the concept not used in the episode, from a
    stream seeded per (run seed, episode, turn) -- identical across models and reruns."""
    return random.Random(f"fork|{seed}|{ep}|{t}").sample(unused, min(n, len(unused)))


def grid_slug(c: str) -> str:
    """file name of a concept's grid probes (--grid-fit DIR/<slug>.pt)."""
    return "".join(ch if ch.isalnum() else "_" for ch in c).strip("_").lower()


def shadow_specs(full_kind: str) -> list[tuple[str, str]]:
    """--shadows of a refit of kind full_kind ("linear@12", "maxpool@8", "ens4_linear"):
    (a) the same kind and layer (resampled data, another seed), (b) the same kind at another layer
    (L + 4, or L - 4 above 26; multi-layer kinds: the sibling ensemble at other depths, e.g.
    ens4_linear <-> ensdeep4_linear), (c) another kind at the same layer (linear <-> maxpool,
    attnpool / rlobf_attn -> linear; ens4_linear <-> ens4_mlp; ensdeep4_linear -> linear@22)."""

    k, L = split_kind(full_kind)
    if L is None:
        if k not in SHADOW_ENS_B:
            return []
        return [("a", k), ("b", SHADOW_ENS_B[k]), ("c", SHADOW_ENS_C[k])]
    L2 = L + 4 if L + 4 <= 30 else L - 4
    other = "maxpool" if k == "linear" else "linear"
    return [("a", f"{k}@{L}"), ("b", f"{k}@{L2}"), ("c", f"{other}@{L}")]


def stratified_plan(concepts: list[str], per_concept: int, mix: dict[str, float],
                    seed: int, batch: int = 0) -> list[tuple[str, str]]:
    """--episodes-per-concept: [(text concept, condition)] in run order. Every DODGE condition of
    the mix (real_score / real_verdict) gets per_concept * w / w_dodge episodes PER CONCEPT. The
    control conditions (the rest of the mix) together get round(n * per_concept * w_ctrl / w_dodge)
    episodes in this shard (the --mix ratio to the dodge episodes). batch > 0 (FILL THE BATCHES):
    the shard total is moved to the NEAREST multiple of batch (ties: up; at least enough for the
    dodge episodes + one per control condition) by changing the number of CONTROL episodes (no
    controls in the mix: extra dodge episodes, concepts in a seeded order). The control total is
    split over the control conditions by largest remainder of their weights (ties seeded), each
    condition's episodes over the concepts as evenly as possible (floor(k / n) each, the remainder
    to concepts in a seeded random order). The list is shuffled (seeded) so batches mix concepts
    and conditions. E.g. 0.6/0.2/0.2, per_concept 2, batch 8: 5 concepts -> 10 + 3 + 3 = 16;
    4 concepts -> 8 + 4 + 4 = 16."""

    rng = random.Random(seed + 13)
    w_dodge = sum(w for k, w in mix.items() if k in DODGE and w > 0)
    if w_dodge <= 0:
        raise ValueError("--episodes-per-concept needs a dodge condition (real_score) in --mix")
    ctrl = {k: w for k, w in mix.items() if k not in DODGE and w > 0}
    n, plan = len(concepts), []

    def spread(cond, total):
        order = list(concepts)
        rng.shuffle(order)
        return [(c, cond) for j, c in enumerate(order)
                for _ in range(total // n + (1 if j < total % n else 0))]

    for cond, w in mix.items():
        if cond in DODGE and w > 0:
            plan += [(c, cond) for c in concepts for _ in range(round(per_concept * w / w_dodge))]
    n_dodge = len(plan)
    n_ctrl = round(n * per_concept * sum(ctrl.values()) / w_dodge)
    if batch > 0:
        tot = n_dodge + n_ctrl
        lo = max(batch * (tot // batch), batch * -(-(n_dodge + len(ctrl)) // batch))
        hi = batch * -(-tot // batch)
        tot = lo if tot - lo < hi - tot and lo >= n_dodge + len(ctrl) else max(hi, lo)
        if not ctrl:  # nothing to adjust: fill with dodge episodes
            dc = next(k for k in mix if k in DODGE)
            order = list(concepts)
            rng.shuffle(order)
            plan += [(order[j % n], dc) for j in range(tot - n_dodge)]
        n_ctrl = tot - n_dodge
    if ctrl:
        wt = sum(ctrl.values())
        share = {k: n_ctrl * w / wt for k, w in ctrl.items()}
        cnt = {k: int(v) for k, v in share.items()}
        rest = sorted(ctrl, key=lambda k: (-(share[k] - cnt[k]), rng.random()))
        for k in rest[: n_ctrl - sum(cnt.values())]:
            cnt[k] += 1
        for k in ctrl:
            plan += spread(k, cnt[k])
    rng.shuffle(plan)
    return plan


def plan_specs(sampler, plan: list[tuple[str, str]]) -> list:
    """One EpisodeSpec per (concept, condition) through the sampler's own draws (texts, mismatch
    partner, shown monitor), with the concept pinned by concept_weights and the condition forced."""

    specs = []
    orig = sampler._conditions
    try:
        for c, cond in plan:
            sampler.concept_weights = {c: 1.0}
            sampler._conditions = lambda k, cond=cond: [cond] * k
            (s,) = sampler.sample(1)
            assert s.text_concept == c and s.cond == cond, (s.id, c, cond)
            specs.append(s)
    finally:
        sampler.concept_weights, sampler._conditions = None, orig
    return specs


def dump_probe(m, kind=None, extra=None, tensors=True) -> dict:
    """Everything needed to re-evaluate a monitor offline: its class, layers, threshold, every
    tensor (_probe_tensors), the bank state (ConceptProbeBank.state()) and a pickled CPU copy
    ("obj": monitor.logits / thresholds work on it directly; torch.load(weights_only=False))."""

    import copy
    import pickle

    import torch

    from next_chameleons.tica.trainers.supervised import _probe_tensors

    try:
        obj = copy.deepcopy(m).to("cpu")
        pickle.dumps(obj)  # a probe that cannot be pickled must not break the torch.save
    except Exception as err:  # never lose the dump over one probe
        obj = None
        print(f"[dump-probes] no CPU copy of {getattr(m, 'name', '?')}: {err!r}", flush=True)
    concepts = getattr(m, "concepts", None)
    thr = getattr(m, "thr", None)
    return {"name": m.name, "kind": kind, "cls": f"{type(m).__module__}.{type(m).__qualname__}",
            "concept": getattr(m, "concept", None) or (
                concepts[0] if concepts and len(concepts) == 1 else concepts),
            "layer": getattr(m, "layer", None),
            "layers": list(getattr(m, "layers_used", None) or [m.layer]),
            "needs_tokens": bool(getattr(m, "needs_tokens", False)),
            "thr": thr.detach().cpu().clone() if isinstance(thr, torch.Tensor) else thr,
            "scale_": getattr(m, "scale_", None),
            "tensors": _probe_tensors(m) if tensors else None,  # (grid: in obj only)
            "state": m.state() if hasattr(m, "state") else None,
            "obj": obj, **(extra or {})}


def _flag(s: str) -> bool:
    """argparse bool: true / false (also 1/0, yes/no, on/off)."""
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {s!r}")


def activations_dir(out: str, root: str) -> Path:
    """Where --activations writes: ROOT/<path of --out after its last 'runs' dir, minus .json> (ROOT given),
    else <out stem>_acts/ next to --out."""
    o = Path(out).resolve()
    if not root:
        return o.parent / (o.stem + "_acts")
    parts = o.with_suffix("").parts
    rel = parts[len(parts) - parts[::-1].index("runs"):] if "runs" in parts else parts[-2:]
    return Path(root).joinpath(*rel)


def default_feat_layers(depth: int) -> str:
    """every 2nd layer from 2, plus the last one (32 layers: 2,4,...,30,31)."""
    return ",".join(str(x) for x in [*range(2, depth - 1, 2), depth - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--topics", default="heldout")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--switch-prob", type=float, default=0.0)
    ap.add_argument("--min-seg", type=int, default=4)
    ap.add_argument("--mix", default="real_score:1.0",
                    help="episode conditions, e.g. real_score:0.6,mismatch:0.2,nobrief:0.2 "
                         "(selectivity: is an UNMONITORED concept hidden too?)")
    ap.add_argument("--kind", default="mixed", help="race refit kind (default: the config mix)")
    ap.add_argument("--panel-layers", default=DEFAULT_LAYERS)
    ap.add_argument("--panel-kinds", default=DEFAULT_KINDS)
    ap.add_argument("--read-every", type=int, default=2, help="readability every N turns (0 off)")
    ap.add_argument("--read-layers", default="8,12,16,20,24,28")
    ap.add_argument("--read-pos", type=int, default=40)
    ap.add_argument("--read-neg", type=int, default=56)
    ap.add_argument("--max-topics", type=int, default=0, help="random subset of the topics")
    ap.add_argument("--dump-feats", help="save per-turn pooled features (.pt) for "
                    "scripts/tica_evasion_detector.py")
    ap.add_argument("--feat-layers", default="8,12,16,20,24")
    ap.add_argument("--dump-twin", action="store_true",
                    help="with --dump-feats: also x_twin, the same text read with adapters off")
    ap.add_argument("--dump-tokens", default="",
                    help="with --dump-feats: layers whose subsampled answer-token states go to "
                         "<dump-feats stem>_tokens.pt")
    ap.add_argument("--token-keep", type=int, default=32,
                    help="--dump-tokens: at most this many evenly spaced tokens per answer")
    ap.add_argument("--dump-probes", help="save every refit / base race probe / panel probe (.pt)")
    ap.add_argument("--episodes-per-concept", type=int, default=0,
                    help="stratified: N dodge episodes per concept + --mix controls "
                         "(overrides --episodes; see stratified_plan)")
    ap.add_argument("--load-panel", help="never-shown panel probes from a --dump-probes file "
                    "(same topics; e.g. the trained shard for its --base twin): no panel fit")
    ap.add_argument("--empty-cache", action="store_true",
                    help="release cached CUDA blocks after the panel fit and each rollout / read "
                         "phase (several processes per GPU: reserved ~= live memory)")
    # dodgescope stage 4 (11-03)
    ap.add_argument("--fork-batch", type=int, default=0,
                    help="adaptive.episode_fork_batch (forks per shared-prefix read; 0 = config); also the "
                         "batch of the grid / shadow reads (default 16 there): caps the read memory")
    ap.add_argument("--force", choices=["catch", "pass", "real"],
                    help="FORCED FEEDBACK: from the turn an episode has --force-k accepted refits, show "
                         "CAUGHT (catch) / PASSED (pass) every turn and fit no more refits; real = the "
                         "unchanged race with the switch turn recorded (rec force_switch / forced)")
    ap.add_argument("--force-k", type=int, default=0)
    ap.add_argument("--dump-nomon", action="store_true",
                    help="with --dump-feats: x_nomon = the SAME conversation read with the adapter ON but the "
                         "monitoring context removed (no brief, no verdicts: the kl_ref nobrief construction, "
                         "_nobrief_conv); + tok_nomon, rec panel_nomon")
    ap.add_argument("--slim-probes", action="store_true",
                    help="--dump-probes without the duplicate _probe_tensors copy (the pickled obj + bank state "
                         "hold the weights) and without re-saving a --load-panel panel (panel_source = its files)")
    ap.add_argument("--dump-text", help="episode texts (JSON): per (ep, turn) the user message as shown, its request, the "
                    "brief part, the verdict / feedback message before the turn, the assistant answer "
                    "(teacher-forced from the data pool) and its source (data episode split)")
    ap.add_argument("--forks", type=int, default=0,
                    help="FORK-ROBUST: per monitored dodge turn, N forks with another unused same-concept answer "
                         "(history + real feedback kept), scored by every deployed probe: rec[\"forks\"]")
    ap.add_argument("--twin-refits", action="store_true",
                    help="STALENESS: every accepted refit also gets a base twin probe (same plan, adapters OFF reads, "
                         "same recipe + seed), scored on the twin reads (needs --dump-twin): rec twin_refits")
    ap.add_argument("--fixed-refits", default="", help="STALENESS (base runs): fit refit-style probes at these "
                    "turns (e.g. 2,5,10,20) on the run's own reads, score every turn: rec fixed_refits")
    ap.add_argument("--pin-kind", help="GAUNTLET: the first probe AND every refit are this kind only "
                    "(e.g. maxpool@12, ens4_linear): episode_refit_kind + first_probe_panel_kinds; no "
                    "switch to another kind on a gate failure")
    ap.add_argument("--no-panel0-cache", action="store_true",
                    help="fit the turn-0 probes in-process (no shared disk cache; shadows need the reads)")
    ap.add_argument("--shadows", action="store_true",
                    help="SHADOW refits: at every accepted refit (and turn-0 probe) also fit, on the same "
                         "forks, (a) same kind + layer on an 80%% resample with another seed, (b) same "
                         "kind at another layer, (c) another kind at the same layer; never shown, scored "
                         "every turn as refits with refit_meta deployed=false, shadow_of")
    ap.add_argument("--grid-fit", help="PASSIVE probe grid: fit every --grid-kinds x layer per topic on ONE "
                    "shared draw of the panel-fit data (turn-0 forks, base model) and save it in the "
                    "--dump-probes format (load it with --load-panel)")
    ap.add_argument("--grid-only", action="store_true", help="stop after --grid-fit")
    ap.add_argument("--grid-kinds", default="linear,maxpool,attnpool,rlobf_attn,ens4_linear,ens4_mlp,"
                    "ensdeep4_linear")
    ap.add_argument("--grid-layers", default="2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31",
                    help="layers of the pooled single-layer grid kinds (linear)")
    ap.add_argument("--grid-tok-layers", default="4,8,12,16,20,22,24,26,30",
                    help="layers of the token grid kinds (maxpool, attnpool, rlobf_attn)")
    ap.add_argument("--refit-k-offset", type=int, default=0,
                    help="REPLAY: advance the refit counter by the zoo-panel fits a --load-panel run "
                         "skips, so refit names (which seed the fits) match the original run")
    ap.add_argument("--twin-panel", action="store_true",
                    help="with --dump-twin: also score the panel on the twin read (full tokens): "
                         "rec[\"panel_twin\"]")
    ap.add_argument("--activations", type=_flag, default=True, metavar="true|false",
                    help="DEFAULT ON (11-04): store per-turn activations for the dodge viewer / analyses, unless "
                         "--dump-feats is given explicitly: pooled reads at every 2nd layer + the last (adapter on; "
                         "--feat-layers overrides), x_twin (adapters off, same text), x_nomon (adapter on, "
                         "monitoring context removed), the deployed probes (slim) and the episode texts, in "
                         "--activations-dir. --activations false: none of it")
    ap.add_argument("--activations-dir", default=os.environ.get("TICA_ACTIVATIONS_DIR", ""),
                    help="root for --activations (env TICA_ACTIVATIONS_DIR; empty: <out stem>_acts/ next to --out)")
    ap.add_argument("--model-path")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    auto_acts = a.activations and not a.dump_feats
    auto_feat_layers = auto_acts and a.feat_layers == "8,12,16,20,24"
    if auto_acts:
        acts = activations_dir(a.out, a.activations_dir)
        a.dump_feats = str(acts / "feats.pt")
        a.dump_twin = a.dump_twin or not a.base
        a.dump_text = a.dump_text or str(acts / "text.json")
        if not a.dump_probes:
            a.dump_probes, a.slim_probes = str(acts / "probes.pt"), True
        print(f"[activations] on (--activations false to skip): {acts}", flush=True)
    if (a.dump_twin or a.dump_tokens) and not a.dump_feats:
        raise SystemExit("--dump-twin / --dump-tokens need --dump-feats")

    import torch

    from next_chameleons.tica.policy import _keep_tokens

    if os.environ.get("TICA_MEM_FRACTION") and torch.cuda.is_available():
        # several processes per GPU: the caching allocator frees its cached blocks at this cap
        # instead of crowding the neighbour (scripts/slurm/dodgescope_node.sh)
        torch.cuda.set_per_process_memory_fraction(float(os.environ["TICA_MEM_FRACTION"]))
    from next_chameleons.tica.trainers.supervised import (
        AdaptiveSupervisedConfig,
        AdaptiveSupervisedTrainer,
        SupervisedConfig,
    )

    cfg = load_run_config(a.config)
    if cfg.model.depth and int(cfg.model.depth) != 32:  # 10-24: 32-layer defaults, rescaled
        from next_chameleons.tica.probe_zoo import scale_kind, scale_layer

        n = int(cfg.model.depth)
        for arg, default in (("panel_layers", DEFAULT_LAYERS), ("read_layers", "8,12,16,20,24,28"),
                             ("feat_layers", "8,12,16,20,24")):
            if getattr(a, arg) == default:
                setattr(a, arg, ",".join(str(scale_layer(int(x), n)) for x in default.split(",")))
        if a.panel_kinds == DEFAULT_KINDS:
            a.panel_kinds = ",".join(scale_kind(k, n) for k in DEFAULT_KINDS.split(","))
    if auto_feat_layers:
        a.feat_layers = default_feat_layers(int(cfg.model.depth or 32))
    if auto_acts and not a.base and getattr(cfg.episodes, "inputs", None):
        a.dump_nomon = True
    if a.model_path:
        cfg.model.path = a.model_path
    if a.pin_kind:  # GAUNTLET: one kind / layer for the first probe and every refit
        a.kind = a.pin_kind
        cfg.rl.adaptive.first_probe_panel_kinds = [a.pin_kind]
    if a.no_panel0_cache or a.shadows:
        cfg.rl.adaptive.first_probe_panel_cache = ""
    cfg.rl.adaptive.episode_refit_kind = a.kind
    if a.fork_batch:
        cfg.rl.adaptive.episode_fork_batch = a.fork_batch
    data, _, factory = build(cfg)
    # "train+heldout": the union of several concept splits (evasion detector: train / test)
    topics = [c for sp in a.topics.split("+") for c in select_topics(data, sp)]
    if a.max_topics and len(topics) > a.max_topics:  # fewer concepts = fewer panel fits
        topics = sorted(random.Random(a.seed).sample(sorted(topics), a.max_topics))
    train = set(cfg.episode_concepts)
    if a.base and a.adapter:
        raise SystemExit("--base and --adapter are exclusive")
    pol = factory(None if a.base else a.adapter, True)
    opening, _ = fmt = prompts("answer", cfg.episodes.inputs, cfg.episodes.layout)
    held = [int(x) for x in cfg.probes.held_out_layers]
    prim = cfg.primary_layer()
    layers = [prim] + [h for h in held if h != prim]
    pan_layers = [int(x) for x in a.panel_layers.split(",") if x]
    pool, panel = MonitorPool(), {}  # panel: concept -> {name: monitor}
    failed = {}
    t_start = time.time()
    timing: dict = collections.defaultdict(float)  # per-component seconds (JSON "timing")
    loaded = None
    if a.load_panel:  # the panel probes of another run of the same topics (pickled CPU copies)
        loaded, lp = {}, {"panel_failed": {}}
        for fpath in a.load_panel.split(","):  # several files (e.g. a 1102 panel + the probe grid): merged
            part = torch.load(fpath, weights_only=False)
            for c, ms in part["panel"].items():
                loaded.setdefault(c, {}).update({p: d["obj"] for p, d in ms.items()})
            lp["panel_failed"].update(part.get("panel_failed", {}))
            del part
        miss = [c for c in topics if c not in loaded]
        if miss or any(m is None for ms in loaded.values() for m in ms.values()):
            raise SystemExit(f"--load-panel {a.load_panel}: no (complete) panel for {miss}")
        failed.update({f"loaded:{k}": v for k, v in lp.get("panel_failed", {}).items()})
    for k, c in enumerate(list(topics)):
        neg = ("train",) if c in train else None
        try:
            mons, _ = fit_answer_probes(cfg.rl.adaptive, data, pol, c, layers, opening,
                                        neg_splits=neg, rename=f"answer_base_c{k}")
        except RuntimeError as err:
            failed[c] = str(err)[-160:]
            topics.remove(c)
            continue
        for L, m in zip(layers, mons, strict=False):
            pool.add(m, round=0, family="concept_linear", held_out=L in held)
        panel[c] = {}
        if loaded is not None:
            panel[c] = {p: m.to(pol.device) for p, m in loaded[c].items()}
            continue
        if not pan_layers:
            continue
        # (a) linear base probes at every panel depth: one multi-layer read (cached); per layer
        # only if some layer fails its gate
        try:
            pm, _ = fit_answer_probes(cfg.rl.adaptive, data, pol, c, pan_layers, opening,
                                      neg_splits=neg, rename=f"panel_c{k}")
            got = dict(zip(pan_layers, pm, strict=True))
        except RuntimeError:
            got = {}
            for L in pan_layers:
                try:
                    (m,), _ = fit_answer_probes(cfg.rl.adaptive, data, pol, c, [L], opening,
                                                neg_splits=neg, rename=f"panel_c{k}_L{L}")
                    got[L] = m
                except RuntimeError as err:
                    failed[f"{c} linear@{L}"] = str(err)[-120:]
        for L, m in got.items():
            panel[c][f"base:linear@{L}"] = m.to(pol.device)
    pool = pool.to(pol.device)
    timing["race_base_and_linear_panel_fit"] = time.time() - t_start
    sampler = EpisodeSampler(
        data.concepts("episode", CONCEPT_SPLITS, only=set(topics)), data.negatives("episode"),
        pool, mix={k: float(v) for k, v in (x.split(":") for x in a.mix.split(","))},
        turns=a.turns, seed=a.seed,
        feedback=cfg.episodes.feedback, switch_prob=a.switch_prob, switch_min_segment=a.min_seg)
    plan_specs_all = None
    if a.episodes_per_concept:  # stratified episodes (drawn before the trainer exists: same rng)
        plan = stratified_plan([cc.concept for cc in sampler.concepts], a.episodes_per_concept,
                               {k: float(v) for k, v in (x.split(":") for x in a.mix.split(","))},
                               a.seed, batch=a.batch)
        plan_specs_all = plan_specs(sampler, plan)
        a.episodes = len(plan_specs_all)
    tcfg = dataclasses.replace(cfg.train, turns=a.turns, episodes_per_step=a.batch, seed=a.seed)
    asc = AdaptiveSupervisedConfig(
        **{f.name: getattr(tcfg, f.name) for f in dataclasses.fields(SupervisedConfig)},
        adaptive=cfg.rl.adaptive)
    out = Path(a.out)
    from next_chameleons.tica.vllm_gen import maybe_attach

    maybe_attach(pol, cfg.model.path, out.parent / (out.stem + "_vllm"))  # TICA_VLLM=1: colocated vLLM generation
    tr = AdaptiveSupervisedTrainer(
        pol, pool, sampler, asc, out.parent / (out.stem + "_work"),
        branch_concepts=data.concepts("fit", CONCEPT_SPLITS),
        branch_negatives=data.negatives("fit"), roots_only=set(topics), prompts=fmt,
        hard_neg={c: data.hard_negatives(c, "fit") for c in topics})
    tr._heldout_layers = lambda: set()  # eval: refits may sit at held-out layers
    if a.force:
        tr.force_mode, tr.force_k = a.force, a.force_k
    if a.dump_text and getattr(tr, "inputs", None) is None:
        from next_chameleons.tica.answer_format import load_inputs as _li

        tr.inputs = _li(cfg.episodes.inputs)
    texts_out: dict = {}
    if a.dump_nomon:  # the request map of the answer format (training builds the kl_ref context from it)
        from next_chameleons.tica.answer_format import load_inputs

        tr.inputs = load_inputs(cfg.episodes.inputs)
    shadow_pending: list = []  # --shadows: (deployed job, shadow reads) per refit event of the batch
    shadow_mon: dict = {}  # shadow name -> (monitor, meta)
    if a.shadows:
        orig_reads = tr._refit_reads

        def reads_with_shadows(plan, t0=None):
            """the deployed refit's reads + ONE extra read of the same forks / no-prompt texts at the
            shadow layers (full tokens where a shadow kind needs them); fits after the rollout."""
            job = orig_reads(plan, t0)
            specs = shadow_specs(plan["full_kind"] if "@" in plan["full_kind"]
                                 else (plan["full_kind"] if plan["kind"] in ZOO_ENS
                                       else f'{plan["kind"]}@{plan["L"]}'))
            if not specs:
                return job
            kl = {fk: (kind_layers(split_kind(fk)[0], prim) if split_kind(fk)[0] in ZOO_ENS
                       else [split_kind(fk)[1]]) for _, fk in specs}
            Ls = sorted({x for v in kl.values() for x in v})
            tokL = sorted({x for fk, v in kl.items() if split_kind(fk)[0] in ZOO_TOKEN for x in v})
            fb = a.fork_batch or 16  # shadow reads: bounded batch, tokens 2 layers at a time
            rall = pol.pooled_shared_prefix(plan["forks"], Ls, batch=fb)
            nall = pol.pooled(plan["np_convs"], Ls)
            ft = {} if tokL else None
            nt = {} if tokL else None
            for j in range(0, len(tokL), 2):
                pol.pooled_shared_prefix(plan["forks"], tokL[j : j + 2], token_out=ft, batch=fb)
                pol.pooled_incremental(plan["np_convs"], tokL[j : j + 2], {}, token_out=nt)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            shadow_pending.append((job, {
                "specs": specs, "kl": kl, "picks": plan["picks"], "concept": plan["concept"],
                "rall": {x: rall[x].float().cpu() for x in Ls},
                "nall": {x: nall[x].float().cpu() for x in Ls},
                "ft": {x: ft[x] for x in tokL} if tokL else None,
                "nt": {x: nt[x] for x in tokL} if tokL else None}))
            return job

        tr._refit_reads = reads_with_shadows

    twin_pending: list = []  # --twin-refits: (trained job, adapter-off job of the SAME plan)
    twin_mon: dict = {}  # parent refit name -> base-twin probe
    if a.twin_refits:
        orig_tw = tr._refit_reads

        def reads_with_twin(plan, t0=None):
            """the refit's reads + the same plan read with the adapters OFF (the twin context); the base
            twin probe is fit after the rollout with the same recipe and name seed (+ "~base")."""
            job = orig_tw(plan, t0)
            if getattr(pol, "_ref_depth", 0) > 0 or "r_pol" in job:  # base-panel turn-0 fits / detectors
                return job
            with pol.reference():
                bjob = orig_tw({**plan, "np_cache": False})  # never touch the rollout's np-read cache
            twin_pending.append((job, bjob))
            return job

        tr._refit_reads = reads_with_twin

    def fit_twins(accepted) -> None:
        from next_chameleons.tica.trainers.supervised import _refit_fit

        while twin_pending:
            job, bjob = twin_pending.pop(0)
            name = job.get("name")
            if name not in accepted:
                continue
            bjob = {**bjob, "name": f"{name}~base", "excl": job.get("excl", set())}
            try:
                mon, _ = _refit_fit(bjob)
            except (RuntimeError, ValueError):
                mon = None
            twin_mon[name] = mon.to(pol.device) if mon is not None else None

    FIXED_KINDS = ["linear@8", "linear@12", "linear@16", "linear@22", "linear@26", "maxpool@12", "attnpool@16",
                   "rlobf_attn@22", "ens4_linear", "ensdeep4_linear", "ens4_mlp", "maxpool@26"]
    fixed_turns = [int(x) for x in a.fixed_refits.split(",") if x]

    def fit_fixed(trajs, ep0):
        """--fixed-refits (base runs): refit-style probes fit at fixed turns on this run's own reads (same
        recipe as a race refit: _refit_plan forks after turn t with the verdict shown at t), kinds rotated."""
        from next_chameleons.tica.trainers.supervised import _refit_fit

        out = {}
        for i, trj in enumerate(trajs):
            if trj.spec.cond not in DODGE:
                continue
            for j, t in enumerate(fixed_turns):
                if t >= len(trj.results) - 1:
                    continue
                r = trj.results[t]
                kind = FIXED_KINDS[(ep0 + i + j) % len(FIXED_KINDS)]
                shown = (r["monitored"].get("score", 0.0), r["monitored"].get("thr", 1.0))
                c = r.get("concept", trj.spec.monitored)
                plan = tr._refit_plan(trj.spec, [dict(m) for m in trj.convs[t]], shown, c, kind, None,
                                      random.Random(f"fixed|{a.seed}|{ep0 + i}|{t}"), False)
                job = tr._refit_reads(plan)
                job["name"], job["excl"] = f"fixed_e{ep0 + i}_t{t}", set()
                try:
                    mon, _ = _refit_fit(job)
                except (RuntimeError, ValueError):
                    mon = None
                if mon is not None:
                    out[job["name"]] = (mon.to(pol.device), i, t, kind)
        return out

    def scale_of(m, c):
        try:
            return float(m.scale([c]).reshape(-1)[0])
        except Exception:
            return None

    def fit_shadows(accepted: dict) -> None:
        """fit the pending shadows of every ACCEPTED parent (name in accepted); same data, never shown."""
        from next_chameleons.tica.trainers.supervised import _refit_fit

        while shadow_pending:
            job, sh = shadow_pending.pop(0)
            parent = job.get("name")
            if parent not in accepted:
                continue
            n = len(sh["picks"])
            for typ, fk in sh["specs"]:
                k, Lo = split_kind(fk)
                lk = sh["kl"][fk]
                L0 = lk[0] if Lo is None else Lo
                rows = list(range(n))
                if typ == "a":  # same kind + layer: 80% resample of the forks, another seed
                    rows = sorted(random.Random(zlib.crc32(f"{parent}~a".encode())).sample(rows,
                                                                                          int(0.8 * n)))
                dev = pol.device
                sj = {"kind": k, "full_kind": fk, "L": L0, "Ls": lk, "L0": L0,
                      "picks": [sh["picks"][q] for q in rows], "concept": sh["concept"],
                      "r": sh["rall"][L0][rows], "rl": {x: sh["rall"][x][rows].to(dev) for x in lk},
                      "ft": ({x: [sh["ft"][x][q] for q in rows] for x in lk} if k in ZOO_TOKEN
                             else None),
                      "npr": sh["nall"][L0], "npl": {x: sh["nall"][x].to(dev) for x in lk},
                      "nt": {x: sh["nt"][x] for x in lk} if k in ZOO_TOKEN else None,
                      "acfg": acfg, "device": dev, "sec_read": 0.0, "read_dt": 0.0,
                      "name": f"{parent}~{typ}", "excl": set()}
                try:
                    mon, info = _refit_fit(sj)
                except (RuntimeError, ValueError) as err:
                    mon, info = None, {"why": str(err)[-120:]}
                shadow_mon[f"{parent}~{typ}"] = (mon.to(dev) if mon is not None else None, {
                    "shadow_of": parent, "shadow_type": typ, "kind": fk, "deployed": False,
                    "fit": {x: info.get(x) for x in ("auroc", "tpr_np", "why")}})
    pol.model.eval()  # KV-cached reads (grad checkpointing disables the cache in train mode)
    acfg = cfg.rl.adaptive
    t0 = time.time()
    if a.grid_fit:  # PASSIVE probe grid (never shown to the model): one shared draw per concept
        from next_chameleons.tica.trainers.supervised import _refit_fit

        gk = [k for k in a.grid_kinds.split(",") if k]
        glin = [int(x) for x in a.grid_layers.split(",") if x]
        gtok = [int(x) for x in a.grid_tok_layers.split(",") if x]
        cells = []
        for k in gk:
            if k in ZOO_ENS:
                cells.append((k, None))
            else:
                cells += [(k, L) for L in (glin if k == "linear" else gtok)]
        grid, gfail, gsec = {}, {}, {}
        for c in topics:
            if (Path(a.grid_fit) / f"{grid_slug(c)}.pt").exists():  # resume: already fitted
                continue
            tg = time.time()
            spec = SimpleNamespace(monitored=c, cond="real_score", feedback=cfg.episodes.feedback,
                                   named=True)
            with pol.reference(), torch.no_grad():  # the panel-fit construction: turn-0 forks, base
                plan = tr._refit_plan(spec, [], None, c, "linear",
                                      lambda x, c=c: tr.opening("real_score", x, c, named=True),
                                      random.Random(f"grid|{c}|{a.seed}"), False)
                kl = {(k, L): (kind_layers(k, prim) if L is None else [L]) for k, L in cells}
                Ls = sorted({x for v in kl.values() for x in v})
                tokL = sorted({x for (k, L), v in kl.items() if k in ZOO_TOKEN for x in v})
                ft, nt = {}, {}  # pooled reads at every layer (no tokens), then full-token reads
                # in chunks of 2 token layers (one 16-layer token read exceeded 38 GB)
                rall = pol.pooled_shared_prefix(plan["forks"], Ls, batch=a.fork_batch or 16)
                nall = pol.pooled(plan["np_convs"], Ls)
                for j in range(0, len(tokL), 2):
                    ch = tokL[j : j + 2]
                    pol.pooled_shared_prefix(plan["forks"], ch, token_out=ft,
                                             batch=a.fork_batch or 16)
                    pol.pooled_incremental(plan["np_convs"], ch, {}, token_out=nt)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            ft = {L: ft[L] for L in tokL}  # full-token states only where a token kind reads
            nt = {L: nt[L] for L in tokL}
            grid[c] = {}
            for k, L in cells:
                lk = kl[(k, L)]
                L0 = lk[0] if L is None else L
                fk = k if L is None else f"{k}@{L}"
                job = {"kind": k, "full_kind": fk, "L": L0, "Ls": lk, "L0": L0,
                       "picks": plan["picks"], "concept": c, "r": rall[L0].float().cpu(),
                       "rl": {x: rall[x].float() for x in lk},
                       "ft": {x: ft[x] for x in lk} if k in ZOO_TOKEN else None,
                       "npr": nall[L0].float().cpu(), "npl": {x: nall[x].float() for x in lk},
                       "nt": {x: nt[x] for x in lk} if k in ZOO_TOKEN else None,
                       "acfg": acfg, "device": pol.device, "sec_read": 0.0, "read_dt": 0.0,
                       "name": f"grid_{zlib.crc32(f'{c}|{fk}|{a.seed}'.encode())}", "excl": set()}
                try:
                    mon, info = _refit_fit(job)
                except (RuntimeError, ValueError) as err:
                    mon, info = None, {"why": str(err)[-120:]}
                if mon is None:
                    gfail[f"{c} {fk}"] = {x: info.get(x) for x in ("why", "auroc", "tpr_np")}
                    continue
                grid[c][f"grid:{fk}"] = dump_probe(mon, fk, {
                    "passive": "never shown to the model (probe grid)",
                    "fit": {x: info.get(x) for x in ("auroc", "tpr_np")}}, tensors=False)
            gsec[c] = round(time.time() - tg, 1)
            gd = Path(a.grid_fit)  # one file per concept: <dir>/<slug>.pt (runs load their concepts)
            gd.mkdir(parents=True, exist_ok=True)
            torch.save({"panel": {c: grid[c]}, "panel_failed": {k: v for k, v in gfail.items()
                                                                if k.startswith(c + " ")},
                        "grid_sec": gsec[c], "cells": cells, "seed": a.seed, "topics": [c],
                        "label": "PASSIVE probe grid: never shown to the model"},
                       gd / f"{grid_slug(c)}.pt")
            nok = len(grid[c])
            grid[c] = None  # saved; free the CPU copies
            print(json.dumps({"grid": c, "sec": gsec[c], "ok": nok,
                              "failed": sum(1 for x in gfail if x.startswith(c + " "))}), flush=True)
            del rall, nall, ft, nt
        (Path(a.grid_fit) / "_index.json").write_text(json.dumps({"sec": gsec, "failed": gfail}) + "\n")
        if a.grid_only:
            return
    # (a) zoo kinds on the BASE model: turn-0 forks, adapters off; own rng (race draws unchanged)
    race_rng = tr.rng
    tr.rng = random.Random(a.seed + 7)
    for c in [] if loaded is not None else topics:
        spec = SimpleNamespace(monitored=c, cond="real_score", feedback=cfg.episodes.feedback,
                               named=True)
        for kind in [k for k in a.panel_kinds.split(",") if k]:
            with pol.reference(), torch.no_grad():
                try:
                    mon, info = tr._episode_refit(
                        spec, [], None, c, None, kind=kind,
                        fork_msg=lambda x, c=c: tr.opening("real_score", x, c, named=True))
                except (RuntimeError, ValueError) as err:
                    mon, info = None, {"why": str(err)[-120:]}
            if mon is None:
                failed[f"{c} {kind}"] = info.get("why")
                continue
            panel[c][f"base:{kind}"] = mon.to(pol.device)
    tr.rng = race_rng
    tr._np_cache, tr._np_ver = {}, None  # base-weight reads must not serve the policy's refits
    tr.refit_k += a.refit_k_offset  # REPLAY: the names (fit seeds) of the original run
    timing["panel_zoo_fit"] = time.time() - t0
    free_cache = (torch.cuda.empty_cache if a.empty_cache and torch.cuda.is_available()
                  else (lambda: None))
    free_cache()
    print(json.dumps({"panel_fit_sec": round(time.time() - t0),
                      "panel_sizes": {c: len(v) for c, v in panel.items()},
                      "failed": len(failed)}), flush=True)
    # every layer / token layer any panel probe reads
    need = set(pan_layers)
    feat_layers = [int(x) for x in a.feat_layers.split(",") if x] if a.dump_feats else []
    need |= set(feat_layers)
    feats: list[torch.Tensor] = []  # per record: [n_feat_layers, d] fp16 (--dump-feats)
    twin = a.dump_twin and not a.base  # --base: the read IS the base model (x_twin == x)
    feats_twin: list[torch.Tensor] = []  # per record, same shape (--dump-twin)
    dtok_layers = [int(x) for x in a.dump_tokens.split(",") if x]
    need |= set(dtok_layers)
    toks_dump: list[torch.Tensor] = []  # per record: [n_tok_layers, k <= token_keep, d] fp16
    toks_twin: list[torch.Tensor] = []
    feats_nomon: list[torch.Tensor] = []  # --dump-nomon
    toks_nomon: list[torch.Tensor] = []
    tok_span: list[int] = []  # full answer-span length (the subsample is linspace over it)
    probe_dump: dict = {"refits": {}, "panel": {}, "race_base": {}, "shadows": {}}
    tok_layers = set()
    for ms in panel.values():
        for m in ms.values():
            lu = getattr(m, "layers_used", None) or [m.layer]
            need |= set(lu)
            if getattr(m, "needs_tokens", False):
                tok_layers |= set(lu)
    read_layers = [int(x) for x in a.read_layers.split(",") if x]
    # forks for readability: render this turn's user message for another answer x
    rendered: dict[str, tuple] = {}
    orig_render = tr.render_feedback

    def recording(*args, **kw):
        s = orig_render(*args, **kw)
        rendered[s] = (args, kw)
        return s

    recording.tagged = getattr(orig_render, "tagged", False)
    recording.v5 = getattr(orig_render, "v5", False)
    tr.render_feedback = recording
    texts = {cc.concept: cc.texts for cc in tr.branch_concepts}
    rr = random.Random(a.seed + 11)
    fork_texts = {cc.concept: cc.texts for cc in sampler.concepts}  # --forks: the episode text pools
    neg_cache: dict = {}

    def negatives_for(c):
        if c not in neg_cache:
            if neg_label_settings(acfg) is not None:
                neg_cache[c] = ({k: neg_clean(v, c, acfg) for k, v in texts.items() if k != c},
                                neg_clean(tr.branch_negatives, c, acfg))
            else:
                neg_cache[c] = ({k: v for k, v in texts.items() if k != c}, tr.branch_negatives)
        return neg_cache[c]

    def readability(trj, t, c):
        if t == 0:
            pre = []
            msg = lambda x: tr.opening("real_score", x, c, named=trj.spec.named)  # noqa: E731
        else:
            got = rendered.get(trj.feedback[t - 1])
            if got is None:
                return None
            args, kw = got
            pre = [dict(m) for m in trj.convs[t - 1]]
            msg = lambda x: orig_render(*args[:3], x, *args[4:], **kw)  # noqa: E731
        others, free = negatives_for(c)
        oc = [k for k, v in others.items() if v]
        pos = rr.sample(texts[c], min(a.read_pos, len(texts[c])))
        negs = ([rr.choice(others[rr.choice(oc)]) for _ in range(a.read_neg * 3 // 4)]
                + rr.sample(free, min(a.read_neg // 4, len(free))))
        forks = [[*pre, {"role": "user", "content": msg(x)}, {"role": "assistant", "content": x}]
                 for x in pos + negs]
        y = torch.tensor([1] * len(pos) + [0] * len(negs))
        reads = pol.pooled_shared_prefix(forks, read_layers,
                                         batch=acfg.episode_fork_batch or None)
        res = {}
        for L in read_layers:
            try:
                res[str(L)] = cv_auroc(reads[L].float(), y, C=acfg.refit_C, seed=t)
            except RuntimeError:
                res[str(L)] = None
        return res

    records, all_trajs, ep0 = [], [], 0
    for b in range(0, a.episodes, a.batch):
        specs = (plan_specs_all[b : b + a.batch] if plan_specs_all is not None
                 else sampler.sample(min(a.batch, a.episodes - b)))
        ctx = pol.reference() if a.base else torch.no_grad()
        mctx = (pol.merged() if acfg.merged_reads and not a.base else contextlib.nullcontext())
        with ctx, mctx, torch.no_grad():
            ts = time.time()
            trajs, rmet = tr.rollout(specs)
            timing["rollout"] += time.time() - ts
            free_cache()
            if isinstance(rmet, dict):  # refit read / fit seconds of this rollout (trainer metrics)
                timing.setdefault("rollout_metrics", []).append(
                    {k: v for k, v in rmet.items() if k.startswith("sec") and isinstance(v, (int, float))})
            ts = time.time()
            all_trajs.extend(trajs)
            refit_mon = {e.monitor.name: e.monitor for e in tr.pool.entries
                         if e.monitor.name in set(tr._transient)}
            evs = [refit_events(trj.results) for trj in trajs]
            if a.shadows or a.pin_kind:  # the shown turn-0 probes are scored like refits too
                refit_mon.update({e.monitor.name: e.monitor for e in tr.pool.entries
                                  if e.family == "turn0_panel" and any(e.monitor.name in ev for ev in evs)})
            b_shadow: dict = {}
            if a.shadows:  # fit this batch's shadows (same reads as their parents), score them below
                ts_sh = time.time()
                fit_shadows(set(refit_mon))
                timing["shadow_fits"] += time.time() - ts_sh
                b_shadow = {n: v for n, v in shadow_mon.items()
                            if v[0] is not None and v[1]["shadow_of"] in refit_mon}
                ts = time.time()
            if a.twin_refits:
                fit_twins(set(refit_mon))
            b_fixed = fit_fixed(trajs, ep0) if fixed_turns else {}
            b_need, b_tok = set(need), set(tok_layers)  # + the layers this batch's refits read
            for m in [*refit_mon.values(), *(v[0] for v in b_shadow.values()), *(v[0] for v in b_fixed.values())]:
                lu = getattr(m, "layers_used", None) or [m.layer]
                b_need |= set(lu)
                if getattr(m, "needs_tokens", False):
                    b_tok |= set(lu)
            state: dict = {}
            rec0 = len(records)  # this batch's records: rec0 + t * len(trajs) + i
            for t in range(a.turns):
                tok: dict = {}
                pooled = pol.pooled_incremental([trj.convs[t] for trj in trajs],
                                                sorted(b_need | b_tok), state,
                                                token_out=tok if (b_tok or dtok_layers) else None)
                for L in b_tok:
                    pooled[("tok", L)] = tok[L]
                for i, trj in enumerate(trajs):
                    r = trj.results[t]
                    cond = trj.spec.cond
                    c = r.get("concept", trj.spec.monitored_at(t))
                    tc = trj.spec.text_concept  # mismatch: the written (unmonitored) concept B
                    dodge = cond in DODGE
                    pc = c if dodge else tc
                    rec = {"ep": ep0 + i, "turn": t, "cond": cond, "concept": c,
                           "text_concept": tc, "off": bool(r.get("off")),
                           "detected": bool(r["monitored"]["detected"]),
                           "active": r.get("active_monitor"),
                           **({"force_switch": r.get("force_switch"), "forced": r.get("forced")}
                              if a.force else {}),
                           "panel": ({p: margin(m, pooled, i, pc) for p, m in panel[pc].items()}
                                     if pc in panel else {}),
                           "refits": {}, "refit_meta": {}}
                    if cond == "mismatch" and c in panel:  # A's panel on B's text
                        rec["false_alarm"] = {p: margin(m, pooled, i, c)
                                              for p, m in panel[c].items()}
                    for n, e in evs[i].items():
                        m = refit_mon.get(n)
                        if m is None or not m.covers(c):
                            continue
                        rec["refits"][n] = margin(m, pooled, i, c)
                        rec["refit_meta"][n] = {**e, "kind": tr._kind_of.get(n),
                                                **({"scale": scale_of(m, c)} if (a.twin_refits or fixed_turns) else {})}
                    for fn, (fm, fi, ft_, fk) in b_fixed.items():
                        if fi == i and fm.covers(c):
                            rec.setdefault("fixed_refits", {})[fn] = [*margin(fm, pooled, i, c), scale_of(fm, c), ft_, fk]
                    for sn, (sm, smeta) in b_shadow.items():  # never shown: deployed=False
                        par = smeta["shadow_of"]
                        if par in evs[i] and sm.covers(c):
                            rec["refits"][sn] = margin(sm, pooled, i, c)
                            rec["refit_meta"][sn] = {
                                "deployed": False, "beaten": None, "kind": smeta["kind"],
                                "shadow_of": par, "shadow_type": smeta["shadow_type"],
                                "fit_turn": evs[i][par]["deployed"]}
                    if (a.read_every and t % a.read_every == 0 and not rec["off"] and dodge
                            and CONDITION_SPECS[cond]["brief"]):
                        tr0 = time.time()
                        rec["readability"] = readability(trj, t, c)
                        timing["readability"] += time.time() - tr0
                    if feat_layers:
                        rec["split"] = data.split.get(tc or c)
                        feats.append(torch.stack([pooled[L][i].float().cpu()
                                                  for L in feat_layers]).half())
                    if dtok_layers:  # full-span tokens above (exact monitors); subsample here
                        tok_span.append(len(tok[dtok_layers[0]][i]))
                        toks_dump.append(torch.stack([_keep_tokens(tok[L][i], a.token_keep)
                                                      for L in dtok_layers]).half())
                    records.append(rec)
                del pooled, tok
            state.clear()
            free_cache()
            timing["turn_reads_scoring_incl_readability"] += time.time() - ts
            if a.forks:  # FORK-ROBUST check (post-rollout; the race is unchanged): at every monitored turn of a
                # dodge episode, a.forks forks keep the history through turn t-1 (incl. the real verdict feedback)
                # and swap turn t's answer for another same-concept text not used in the episode (its user
                # message re-rendered for that text); every probe deployed at t scores every fork
                ts_f = time.time()
                mons_all = {e.monitor.name: e.monitor for e in tr.pool.entries}
                for i, trj in enumerate(trajs):
                    s_ = trj.spec
                    if s_.cond not in DODGE:
                        continue
                    unused = [x for x in fork_texts.get(s_.text_concept, []) if x not in set(s_.texts)]
                    for t in range(len(trj.results)):
                        r = trj.results[t]
                        if r.get("off") or not r.get("deployed"):
                            continue
                        c = r.get("concept", s_.monitored_at(t))
                        if t == 0:
                            pre = []
                            msg = lambda x, s_=s_: tr.opening(  # noqa: E731
                                "real_score" if CONDITION_SPECS[s_.cond]["brief"] else "nobrief", x, s_.monitored,
                                named=s_.named)
                        else:
                            got = rendered.get(trj.feedback[t - 1])
                            if got is None:
                                continue
                            args_, kw_ = got
                            pre = [dict(m) for m in trj.convs[t - 1]]
                            msg = lambda x, args_=args_, kw_=kw_: orig_render(*args_[:3], x, *args_[4:], **kw_)  # noqa: E731
                        xs = fork_draw(a.seed, ep0 + i, t, unused, a.forks)
                        if not xs:
                            continue
                        forks = [[*pre, {"role": "user", "content": msg(x)}, {"role": "assistant", "content": x}]
                                 for x in xs]
                        dep = [n for n in r["deployed"] if n in mons_all]
                        lu, tl = set(), set()
                        for n in dep:
                            m = mons_all[n]
                            ll = getattr(m, "layers_used", None) or [m.layer]
                            lu |= set(ll)
                            if getattr(m, "needs_tokens", False):
                                tl |= set(ll)
                        ft: dict = {}
                        fp = pol.pooled_shared_prefix(forks, sorted(lu | tl), token_out=ft if tl else None,
                                                      batch=acfg.episode_fork_batch or None)
                        for L in tl:
                            fp[("tok", L)] = ft[L]
                        per = {n: [margin(mons_all[n], fp, q, c)[1] for q in range(len(forks))] for n in dep}
                        newest = r.get("active_monitor")
                        rec_ = records[rec0 + t * len(trajs) + i]
                        rec_["forks"] = {
                            "n": len(forks), "newest": newest, "deployed": dep,
                            "pass_newest": sum(not d for d in per.get(newest, [])) if newest in per else None,
                            "pass_all": sum(not any(per[n][q] for n in dep) for q in range(len(forks))),
                            "pass_per_probe": {n: sum(not d for d in v) for n, v in per.items()}}
                        del fp, ft
                free_cache()
                timing["fork_reads"] += time.time() - ts_f
            ts = time.time()
            if twin:  # the same texts, adapters off (one unmerge per batch inside merged())
                tstate: dict = {}
                tw_layers = sorted(set(feat_layers) | set(dtok_layers))
                tw_tok = set(dtok_layers)
                if a.twin_panel:  # + every layer / token layer the panel reads (full tokens)
                    tw_layers = sorted(set(tw_layers) | need | tok_layers)
                    tw_tok |= tok_layers
                tw_probe_tok = set()
                if a.twin_refits:  # + the base twin probes of this batch's refits (full tokens)
                    for pn in {n for ev in evs for n in ev}:
                        mm = twin_mon.get(pn)
                        if mm is None:
                            continue
                        ll = getattr(mm, "layers_used", None) or [mm.layer]
                        tw_layers = sorted(set(tw_layers) | set(ll))
                        if getattr(mm, "needs_tokens", False):
                            tw_tok |= set(ll)
                            tw_probe_tok |= set(ll)
                full_tok = a.twin_panel or bool(tw_probe_tok)
                with pol.reference():
                    for t in range(a.turns):
                        ttok: dict = {}
                        tp = pol.pooled_incremental(
                            [trj.convs[t] for trj in trajs], tw_layers, tstate,
                            token_out=ttok if tw_tok else None,
                            token_keep=None if full_tok else a.token_keep)
                        for L in (tok_layers if a.twin_panel else set()) | tw_probe_tok:
                            tp[("tok", L)] = ttok[L]
                        for i in range(len(trajs)):
                            feats_twin.append(torch.stack([tp[L][i].float().cpu()
                                                           for L in feat_layers]).half())
                            if dtok_layers:
                                toks_twin.append(torch.stack([
                                    _keep_tokens(ttok[L][i], a.token_keep)
                                    for L in dtok_layers]).half())
                            if a.twin_refits:
                                rr2 = records[rec0 + t * len(trajs) + i]
                                c2 = rr2["concept"]
                                rr2["twin_refits"] = {pn: [*margin(twin_mon[pn], tp, i, c2), scale_of(twin_mon[pn], c2)]
                                                      for pn in evs[i] if twin_mon.get(pn) is not None
                                                      and twin_mon[pn].covers(c2)}
                            if a.twin_panel:
                                rr_ = records[rec0 + t * len(trajs) + i]
                                pc_ = (rr_["concept"] if rr_["cond"] in DODGE
                                       else rr_["text_concept"])
                                rr_["panel_twin"] = ({p: margin(m, tp, i, pc_)
                                                      for p, m in panel[pc_].items()}
                                                     if pc_ in panel else {})
                        del tp, ttok
                tstate.clear()
            free_cache()
            timing["twin_reads"] += time.time() - ts
            ts = time.time()
            if a.dump_nomon:  # adapter ON, monitoring context removed (no brief, no verdicts), same text
                ts_n = time.time()
                nstate: dict = {}
                n_layers = sorted(set(feat_layers) | set(dtok_layers) | need | tok_layers)
                n_tok = set(dtok_layers) | tok_layers
                for t in range(a.turns):
                    nc = [tr._nobrief_conv(trj.convs[t]) for trj in trajs]  # the kl_ref "nobrief" context
                    ok = [x is not None for x in nc]
                    nc = [x if x is not None else trj.convs[t] for x, trj in zip(nc, trajs, strict=True)]
                    ntok: dict = {}
                    npool = pol.pooled_incremental(nc, n_layers, nstate,
                                                   token_out=ntok if n_tok else None)
                    for L in tok_layers:
                        npool[("tok", L)] = ntok[L]
                    for i in range(len(trajs)):
                        feats_nomon.append(torch.stack([npool[L][i].float().cpu()
                                                        for L in feat_layers]).half())
                        if dtok_layers:
                            toks_nomon.append(torch.stack([_keep_tokens(ntok[L][i], a.token_keep)
                                                           for L in dtok_layers]).half())
                        rr_ = records[rec0 + t * len(trajs) + i]
                        rr_["nomon_ok"] = ok[i]
                        pc_ = rr_["concept"] if rr_["cond"] in DODGE else rr_["text_concept"]
                        rr_["panel_nomon"] = ({p: margin(m, npool, i, pc_) for p, m in panel[pc_].items()}
                                              if pc_ in panel else {})
                    del npool, ntok
                nstate.clear()
                free_cache()
                timing["nomon_reads"] += time.time() - ts_n
            ts = time.time()
            if a.dump_text:  # the exact conversations of this batch (texts only; never committed)
                from next_chameleons.tica.answer_format import text_id as _tid

                for i, trj in enumerate(trajs):
                    s_ = trj.spec
                    turns = {}
                    for t in range(len(trj.convs)):
                        conv = trj.convs[t]
                        ans = conv[-1]["content"]
                        user = conv[-2]["content"] if len(conv) >= 2 else ""
                        req = tr.inputs.get(_tid(ans)) or tr.inputs.get(_tid(ans.strip())) if tr.inputs else None
                        cut = user.rfind(req) if req else -1
                        turns[t] = {
                            "user": user, "request": req,
                            "brief": (user[:cut].rstrip() if cut > 0 else ("" if cut == 0 else None))
                            if t == 0 else None,
                            "feedback": (user[:cut].rstrip() if cut > 0 else None) if t > 0 else None,
                            "feedback_raw": trj.feedback[t - 1] if t > 0 and t - 1 < len(trj.feedback) else None,
                            "answer": ans,
                            "source": {"pool": "data episode split (teacher-forced)",
                                       "concept": s_.text_concept or "concept-free negatives",
                                       "in_spec_texts": ans == s_.text(t)},
                            "off": bool(trj.results[t].get("off")) if t < len(trj.results) else None,
                            "system": [m["content"] for m in conv if m["role"] == "system"] or None}
                    texts_out[ep0 + i] = {"cond": s_.cond, "concept": s_.monitored, "text_concept": s_.text_concept,
                                          "spec_id": s_.id, "turns": turns}
            if a.dump_probes:  # every refit of this batch (dropped from the pool below)
                for n, m in refit_mon.items():
                    eps = [ep0 + i for i, ev in enumerate(evs) if n in ev]
                    ev = evs[eps[0] - ep0][n] if eps else {}
                    key = n if n not in probe_dump["refits"] else f"{n}|ep{eps[0] if eps else b}"
                    probe_dump["refits"][key] = dump_probe(
                        m, tr._kind_of.get(n),
                        {"ep": eps[0] if eps else None, "eps": eps, "batch": b // a.batch,
                         "deployed": ev.get("deployed"), "beaten": ev.get("beaten")}, tensors=not a.slim_probes)
                for sn, (sm, smeta) in b_shadow.items():
                    if sn not in probe_dump["shadows"]:
                        probe_dump["shadows"][sn] = dump_probe(sm, smeta["kind"], {
                            **smeta, "eps": [ep0 + i for i, ev in enumerate(evs)
                                             if smeta["shadow_of"] in ev],
                            "never_shown": "shadow refit: fit with its parent, never shown"}, tensors=not a.slim_probes)
            timing["probe_dump"] += time.time() - ts
        ep0 += len(trajs)
        tr.pool.entries = [e for e in tr.pool.entries if e.monitor.name not in set(tr._transient)]
        tr._transient = []
        rendered.clear()
        print(f"[batch {b // a.batch}] episodes {ep0} {time.time() - t0:.0f}s "
              + json.dumps({k: round(v) for k, v in timing.items() if isinstance(v, float)}),
              flush=True)
    if a.dump_text:
        Path(a.dump_text).parent.mkdir(parents=True, exist_ok=True)
        Path(a.dump_text).write_text(json.dumps({"config": a.config, "adapter": a.adapter, "base": a.base,
                                                 "seed": a.seed, "topics": topics, "episodes": texts_out}))
    agg = aggregate(records, a.turns)
    rep = {"config": a.config, "adapter": a.adapter, "base": a.base, "topics": topics,
           "episodes": a.episodes, "turns": a.turns, "kind": a.kind,
           "panel_names": sorted({p for v in panel.values() for p in v}),
           "panel_failed": failed, "by_probe_index": probe_index_metrics(all_trajs),
           "aggregate": agg, "selectivity": selectivity(records, a.turns),
           "records": records, "sec": round(time.time() - t0),
           "kind_layers": {k: kind_layers(split_kind(k)[0], split_kind(k)[1] or prim)
                           for k in a.panel_kinds.split(",") if k},
           # every flag value (incl. the dodgescope dumps) + the stratified episode plan
           "args": vars(a), "twin": twin,
           "shown_only": (f"model was shown ONLY probe kind {split_kind(a.pin_kind)[0]} at layer "
                          f"{split_kind(a.pin_kind)[1] or ','.join(map(str, kind_layers(a.pin_kind, prim)))}"
                          if a.pin_kind else None),
           "panel_label": ("never-shown panel" + (" + PASSIVE probe grid (never shown)"
                                                  if a.load_panel and "grid" in a.load_panel else "")),
           "episode_plan": [{"ep": k, "concept": s.text_concept, "monitored": s.monitored,
                             "cond": s.cond} for k, s in enumerate(plan_specs_all or [])]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep))
    ts_save = time.time()
    if feat_layers:  # activations + numeric labels only (evasion detector; never committed)
        meta = [{k: r[k] for k in ("ep", "turn", "cond", "off", "concept", "text_concept",
                                   "split", "detected")} for r in records]
        d = {"layers": feat_layers, "x": torch.stack(feats), "meta": meta,
             "adapter": a.adapter, "base": a.base}
        if twin:
            d["x_twin"] = torch.stack(feats_twin)
        if feats_nomon:
            d["x_nomon"] = torch.stack(feats_nomon)
        Path(a.dump_feats).parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, a.dump_feats)
        if dtok_layers:
            fp = Path(a.dump_feats)
            dt = {"layers": dtok_layers, "keep": a.token_keep, "tok": toks_dump,
                  "n_span": tok_span, "n_tok": [len(x[0]) for x in toks_dump], "meta": meta,
                  "adapter": a.adapter, "base": a.base,
                  "note": "tok[i]: [n_layers, k, d] fp16, the answer span's tokens at "
                          "round(linspace(0, n_span - 1, k)) (all of them if n_span <= keep)"}
            if twin:
                dt["tok_twin"] = toks_twin
            if toks_nomon:
                dt["tok_nomon"] = toks_nomon
            torch.save(dt, fp.with_name(fp.stem + "_tokens.pt"))
    if a.dump_probes:  # the real probes: refits (per batch above), base race probes, the panel
        for e in tr.pool.entries:  # after the batches: the race base probes only
            if e.monitor.name not in probe_dump["refits"]:
                probe_dump["race_base"][e.monitor.name] = dump_probe(
                    e.monitor, tr._kind_of.get(e.monitor.name, "linear"),
                    {"family": e.family, "held_out": e.held_out}, tensors=not a.slim_probes)
        if a.slim_probes and loaded is not None:  # the loaded panel lives in its own files
            probe_dump["panel_source"] = a.load_panel
        else:
            for c, ms in panel.items():
                probe_dump["panel"][c] = {p: dump_probe(m, p.split(":", 1)[1]) for p, m in ms.items()}
        probe_dump.update(config=a.config, adapter=a.adapter, base=a.base, topics=topics,
                          panel_failed=failed)
        Path(a.dump_probes).parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(probe_dump, a.dump_probes)
        except Exception as err:  # keep the weights even if a pickle fails
            print(f"[dump-probes] full save failed ({err!r}); saving without obj/state",
                  flush=True)
            for v in [*probe_dump["refits"].values(), *probe_dump["race_base"].values(),
                      *probe_dump["shadows"].values(),
                      *(x for ms in probe_dump["panel"].values() for x in ms.values())]:
                v["obj"] = v["state"] = None
            torch.save(probe_dump, a.dump_probes)
    if a.dump_feats or a.dump_probes:  # the timing breakdown incl. the dump saves (JSON again)
        timing["dumps_save"] = time.time() - ts_save
    timing["total"] = time.time() - t_start
    if torch.cuda.is_available():
        timing["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
        timing["gpu_max_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    rep["timing"] = timing
    out.write_text(json.dumps(rep))
    print("timing " + json.dumps({k: round(v, 1) for k, v in timing.items() if isinstance(v, float)}),
          flush=True)
    short = {k: v for k, v in agg.items() if k not in ("panel", "layer_profile_by_turn",
                                                        "readability_by_turn")}
    short["panel_det"] = {p: v["det"] for p, v in agg["panel"].items()}
    short["selectivity"] = {p: {k: v[k] for k in ("real", "mismatch", "nobrief")}
                            for p, v in rep["selectivity"]["text_probe"].items()}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
