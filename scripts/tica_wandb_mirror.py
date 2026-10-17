"""Mirror a TICA run's logs into a structured wandb dashboard (REPLICATION 2026-10-17k). Runs on the
LOGIN node (internet), never inside the training job, so it can be attached to a run that is already
going and costs no GPU time. Aggregates only: no dataset text is ever read or logged.

Reads <run>/adaptive/train_log.jsonl + resume_*/adaptive chunks (global steps, as tica_plot.load),
<run>/val/step_*.json (validation races) and <run>/adaptive*/refits.jsonl (refit gate outcomes), and
logs them re-keyed into dashboard sections:

  progress/   pass rate, beaten-within-k, turns-to-beat, refits, detection at turn 0 / last turn
  family/     per concept FAMILY: beaten fraction, turns-to-beat, hazard
  concept/    per concept: the same
  probe_kind/ per probe kind (arch@layer): beaten fraction, turns, catch rate, gate pass rate;
              probe_arch/ and probe_layer/ = its architecture / layer marginals + refit acceptance
  heldout/    held-out probes (never trained against) detection
  per_turn/   detection by turn index within an episode
  switch/     topic-switch segments; first_probe/ fork-fit first probes
  monitor_off/ detection on "Monitor: OFF" turns (should stay at the base model's level)
  curriculum/ concept / kind sampling probabilities, ramp lambda
  capability/ GSM8K forks, anchor + KL; loss/ objective terms; timing/; system/ (lr, memory)
  val/        dev-concept validation races (own step axis: they finish out of order)

State (last mirrored step, validation steps sent) lives in <run>/wandb_mirror.json, so it can be
re-run or left in --follow mode. Online when `wandb login` was done, else offline (+ `wandb sync`).

  python scripts/tica_wandb_mirror.py --run RUN_DIR [--project tica] [--follow 300]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica_plot import load

SECTIONS = [  # (regex on the raw key without "rollout/", target template); first match wins
    (r"^task_family/([^/]+)/(.+)$", r"family/\1/\2"),
    (r"^task_concept/([^/]+)/(.+)$", r"concept/\1/\2"),
    (r"^beatk_(frac|n|turns)_(\d+)$", r"by_probe_index/k\2_\1"),
    (r"^beat1_(first|switch)_frac$", r"by_probe_index/k1_\1_frac"),
    (r"^beat_(frac|n|turns)_(.+)$", r"probe_kind/\2/beat_\1"),
    (r"^kind_catch_(.+)$", r"probe_kind/\1/catch"),
    (r"^gate_(ok|att)_(.+)$", r"probe_kind/\2/gate_\1"),
    (r"^gate_fail_(\w+?)_(.+)$", r"probe_kind/\2/gate_fail_\1"),
    (r"^reloc_(.+)$", r"motion/reloc_\1"),
    (r"^turn_(move.*)$", r"motion/\1"),
    (r"^kind_p_(.+)$", r"curriculum/kind_p/\1"),
    (r"^concept_p_(.+)$", r"curriculum/concept_p/\1"),
    (r"^curriculum_lambda$", r"curriculum/lambda"),
    (r"^detturn_real_score_(\d+)$", r"per_turn/monitored_turn_\1"),
    (r"^detturn_(\w+?)_(\d+)$", r"per_turn/\1_turn_\2"),
    (r"^detho_turn_(\d+)$", r"heldout/detected_turn_\1"),
    (r"^detho_(.+)$", r"heldout/\1"),
    (r"^det_(.+)$", r"progress/detected_\1"),
    (r"^(beat_within\d+|beat_frac|beat_n|beat_turns_mean|mean_pass_turn|pass_rate)$",
     r"progress/\1"),
    (r"^(segbeat_.+)$", r"progress/\1"),
    (r"^ep_off_(.+)$", r"monitor_off/\1"),
    (r"^ep_first_probe_(.+)$", r"first_probe/\1"),
    (r"^sw_(.+)$", r"switch/\1"),
    (r"^ep_(.+)$", r"progress/ep_\1"),
    (r"^sec_(.+)$", r"timing/\1"),
    (r"^(cap_gen|anchor)/(.+)$", r"capability/\1_\2"),
    (r"^train/kl$", r"capability/kl"),
    (r"^train/(.+)$", r"loss/\1"),
    (r"^(lr|grad_norm|max_mem_gb)$", r"system/\1"),
    (r"^sec$", r"timing/step"),
]
_RX = [(re.compile(a), b) for a, b in SECTIONS]


def rekey(k: str) -> str:
    raw = k[len("rollout/"):] if k.startswith("rollout/") else k
    for rx, tpl in _RX:
        if rx.match(raw):
            return rx.sub(tpl, raw)
    return f"other/{raw}"


def flat(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k == "step" or isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if v != v:  # NaN
            continue
        out[rekey(k)] = v
    # probe ARCHITECTURE and LAYER marginals of the per-kind beaten fraction (count-weighted)
    agg = defaultdict(lambda: [0.0, 0.0])
    for k, v in out.items():
        m = re.match(r"^probe_kind/(.+)/beat_frac$", k)
        n = out.get(f"probe_kind/{m.group(1)}/beat_n", 0) if m else 0
        if m and n:
            arch, _, L = m.group(1).partition("@")
            for key in (f"probe_arch/{arch}", *([f"probe_layer/L{L}"] if L else [])):
                agg[key][0] += v * n
                agg[key][1] += n
    for key, (s, n) in agg.items():
        out[f"{key}/beat_frac"] = s / n
        out[f"{key}/beat_n"] = n
    return out


def refit_gate(run: Path) -> dict[int, dict]:
    """Per global step: refit attempts / accepted per probe kind (refits.jsonl)."""

    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for sub in [run / "adaptive", *sorted(run.glob("resume_*/adaptive"))]:
        f = sub / "refits.jsonl"
        if not f.exists():
            continue
        off = int(sub.parent.name.split("_")[1]) if sub.parent.name.startswith("resume_") else 0
        for line in f.open():
            r = json.loads(line)
            k = str(r.get("kind")).split("@")[0]
            s = int(r.get("step", 0)) + 1 + off  # refits.jsonl steps are 0-based
            by[s][k][0] += 1
            by[s][k][1] += bool(r.get("kept"))
    return {s: {f"probe_arch/{k}/refit_attempts": a, f"probe_arch/{k}/refit_accept_rate": c / a}
            for s, d in by.items() for k, (a, c) in d.items()}


def load_val(run: Path) -> dict[int, dict]:
    out = {}
    for f in sorted((run / "val").glob("step_*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:  # being written
            continue
        row = {f"val/{k}": v for k, v in d.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
        row.update({f"val/by_probe_index/{k}": v for k, v in (d.get("by_probe_index") or {}).items()
                    if isinstance(v, (int, float)) and v == v})
        row.update({f"val/family/{k}": v for k, v in (d.get("per_family") or {}).items()})
        out[int(d["step"])] = row
    return out


def open_run(run: Path, project: str):
    import wandb

    cfg = {}
    cf = run / "adaptive" / "config.json"
    if cf.exists():
        cfg = json.loads(cf.read_text())
    kw = dict(project=project, name=run.name, id=f"{run.name}-mirror", resume="allow", config=cfg,
              dir=str(run), settings=wandb.Settings(silent=True))
    try:
        wb = wandb.init(**kw)
    except Exception:  # not logged in / no network: offline, `wandb sync` later
        wb = wandb.init(**kw, mode="offline")
    wb.define_metric("train_step")
    wb.define_metric("val_step")
    for sec in ("progress", "by_probe_index", "family", "concept", "probe_kind", "probe_arch",
                "probe_layer",
                "heldout", "per_turn", "motion", "switch", "first_probe", "monitor_off",
                "curriculum", "capability", "loss", "timing", "system", "other"):
        wb.define_metric(f"{sec}/*", step_metric="train_step")
    wb.define_metric("val/*", step_metric="val_step")
    return wb


def mirror(wb, run: Path, state_path: Path) -> int:
    """Log every train step / validation result not yet sent; returns the number of log calls."""

    st = json.loads(state_path.read_text()) if state_path.exists() else {"step": 0, "val": []}
    gate = refit_gate(run)
    n = 0
    for r in load(run):
        g = int(r["step"])
        if g <= st["step"]:
            continue
        wb.log({**flat(r), **gate.get(g, {}), "train_step": g})
        st["step"] = g
        n += 1
    for g, d in sorted(load_val(run).items()):
        if g in st["val"]:
            continue
        wb.log({**d, "val_step": g})
        st["val"].append(g)
        n += 1
    es = run / "adaptive" / "early_stop.json"
    if es.exists():
        e = json.loads(es.read_text())
        wb.summary.update({f"early_stop/{k}": v for k, v in e.items()
                           if isinstance(v, (int, float, bool))})
    state_path.write_text(json.dumps(st))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--project", default="tica")
    ap.add_argument("--follow", type=int, default=0, help="re-mirror every N seconds (0 = once)")
    a = ap.parse_args()
    state = a.run / "wandb_mirror.json"
    wb = open_run(a.run, a.project)  # one wandb run for the whole session (offline-safe)
    try:
        while True:
            n = mirror(wb, a.run, state)
            print(json.dumps({"time": time.strftime("%H:%M:%S"), "logged": n}), flush=True)
            if not a.follow:
                break
            time.sleep(a.follow)
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
