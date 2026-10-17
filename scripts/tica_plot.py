"""TICA plots (aggregates only; never prints or plots any dataset text).

1) Legacy overview grid (one row per run): loss terms, rollout metrics, per-kind catch rate.
     python scripts/tica_plot.py RUN_DIR [RUN_DIR ...] --out plots.png
2) Training curves of one run from <run>/adaptive/train_log.jsonl (+ resume_* chunks),
   <run>/val/step_*.json and <run>/adaptive/early_stop.json -> <outdir>/<run>_train.png and
   <outdir>/<run>_families.png (per-family beaten + turns-to-beat small multiples):
     python scripts/tica_plot.py --run RUN_DIR [--run RUN_DIR2]
3) Race plots from tica_switch_eval JSONs -> <outdir>/races_km.png (Kaplan-Meier "fraction of
   newly deployed probes beaten by judged turn k" from deploy_records; censored records stay at
   risk until their 'turns') and <outdir>/races_concepts.png (per-concept pass rate + mean
   turns-to-beat):
     python scripts/tica_plot.py --races A.json B.json [--outdir runs/plots] [--tag NAME]
Train-log keys are looked up with and without the "rollout/" prefix the trainer adds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROLL = 10  # rolling-mean window (logged steps) for the new training curves


# ---------------------------------------------------------------- loading
def load(run: Path) -> list[dict]:
    rows = []
    for sub in [run / "adaptive", *sorted(run.glob("resume_*/adaptive"))]:
        f = sub / "train_log.jsonl"
        if not f.exists():
            continue
        off = int(sub.parent.name.split("_")[1]) if sub.parent.name.startswith("resume_") else 0
        rows = [r for r in rows if r["step"] <= off]  # a resume chunk replaces later steps
        rows += [{**r, "step": r["step"] + off} for r in map(json.loads, f.open())]
    return rows


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def get(row: dict, key: str):
    """row[key] or row["rollout/" + key] (the trainer prefixes rollout metrics)."""
    v = row.get(key)
    return row.get(f"rollout/{key}") if v is None else v


def series(rows, key):
    pts = [(r["step"], get(r, key)) for r in rows if _num(get(r, key))]
    return [p[0] for p in pts], [p[1] for p in pts]


def keys_like(rows, pattern: str) -> list[str]:
    """Sorted unprefixed keys matching regex `pattern` (applied after stripping 'rollout/')."""
    rx = re.compile(pattern)
    out = {k.removeprefix("rollout/") for r in rows for k in r}
    return sorted(k for k in out if rx.fullmatch(k))


def smooth(xs, ys, w=5):
    """Trailing moving average over the last w finite points (NaN-aware)."""
    out = []
    for i in range(len(ys)):
        win = [y for y in ys[max(0, i - w + 1): i + 1] if _num(y)]
        out.append(sum(win) / len(win) if win else float("nan"))
    return xs, out


# ---------------------------------------------------------------- 1) legacy grid
def legacy(runs: list[Path], out: str, plt):
    fig, ax = plt.subplots(len(runs), 3, figsize=(16, 3.6 * len(runs)), squeeze=False)
    for i, run in enumerate(runs):
        rows = load(run)
        name = run.name
        p = ax[i][0]
        for k, lab in [("train/loss", "total"), ("train/hide_mse", "hide (probe score)"),
                       ("train/match_mse", "match base (no-brief/mismatch)"),
                       ("train/nll", "LM nll"),
                       ("cap_gen/nll", "capability nll (own correct answers)")]:
            xs, ys = series(rows, k)
            if xs:
                p.plot(*smooth(xs, ys), label=lab)
        p.set_title(f"{name}: train loss terms (5-step mean)")
        p.set_xlabel("step")
        p.legend(fontsize=7)
        p = ax[i][1]
        for k, lab in [("rollout/pass_rate", "pass rate (in-context rollouts)"),
                       ("rollout/sw_seg_later", "pass, later turns of a segment"),
                       ("rollout/det_nobrief_tN", "no-brief detection (want 1)"),
                       ("rollout/detho_nobrief_tN", "no-brief held-out-layer det"),
                       ("cap_gen/acc", "GSM8K acc while hiding")]:
            xs, ys = series(rows, k)
            if xs:
                p.plot(xs, ys, marker="o", ms=3, label=lab)
        p.set_ylim(-0.05, 1.05)
        p.set_title(f"{name}: rollout metrics (every 10 steps)")
        p.set_xlabel("step")
        p.legend(fontsize=7)
        p = ax[i][2]
        kinds = sorted({k for r in rows for k in r if k.startswith("rollout/kind_catch_")})
        for k in kinds:
            xs, ys = series(rows, k)
            p.plot(xs, ys, marker="o", ms=3, label=k.split("kind_catch_")[1])
        p.set_ylim(-0.05, 1.05)
        p.set_title(f"{name}: cumulative catch rate of the ACTIVE refit, by probe kind")
        p.set_xlabel("step")
        if kinds:
            p.legend(fontsize=7)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


# ---------------------------------------------------------------- 2) training curves
def _empty(p, msg):
    p.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9, transform=p.transAxes)
    p.set_xticks([])
    p.set_yticks([])


def _lines(p, rows, specs, w=ROLL, ylim=None, marker=False):
    """Plot rolling means of [(key, label)]; returns the number of lines drawn."""
    n = 0
    for k, lab in specs:
        xs, ys = series(rows, k)
        if not xs:
            continue
        if marker:
            p.plot(xs, ys, alpha=0.25, lw=0.8)
            p.plot(*smooth(xs, ys, w), label=lab, color=p.lines[-1].get_color())
        else:
            p.plot(*smooth(xs, ys, w), label=lab)
        n += 1
    if ylim:
        p.set_ylim(*ylim)
    if n:
        p.legend(fontsize=7)
    p.set_xlabel("step")
    return n


def load_val(run: Path) -> list[tuple[int, float]]:
    pts = {}
    for f in [*sorted(run.glob("val/step_*.json")), *sorted(run.glob("resume_*/val/step_*.json"))]:
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _num(d.get("val_score")) and "step" in d:
            pts[int(d["step"])] = float(d["val_score"])
    return sorted(pts.items())


def load_early_stop(run: Path) -> dict:
    for f in [*sorted(run.glob("resume_*/adaptive/early_stop.json"), reverse=True),
              run / "adaptive" / "early_stop.json"]:
        if f.exists():
            try:
                return json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
    return {}


def train_curves(run: Path, outdir: Path, plt) -> list[Path]:
    rows = load(run)
    name = run.name
    if not rows:
        print(f"# {run}: no train_log.jsonl rows, skipped")
        return []
    fig, ax = plt.subplots(3, 3, figsize=(18, 12))
    ax = ax.ravel()
    roll = f"{ROLL}-step rolling mean"
    p = ax[0]
    if not _lines(p, rows, [("pass_rate", "pass rate"), ("beat_frac", "beat_frac"),
                            ("segbeat_frac", "segbeat_frac")], ylim=(-0.05, 1.05), marker=True):
        _empty(p, "no pass_rate / beat_frac keys")
    p.set_title(f"pass rate + fraction of new probes beaten ({roll})")
    p = ax[1]
    if not _lines(p, rows, [(f"beat_within{k}", f"beaten within {k} turn(s)") for k in (1, 3, 5)],
                  ylim=(-0.05, 1.05)):
        _empty(p, "no beat_within* keys (older run)")
    p.set_title(f"new probes beaten within k judged turns ({roll})")
    p = ax[2]
    if not _lines(p, rows, [("beat_turns_mean", "beat_turns_mean"),
                            ("segbeat_turns_mean", "segbeat_turns_mean")]):
        _empty(p, "no beat_turns_mean keys (older run)")
    p.set_title(f"mean judged turns to beat a new probe ({roll})")
    p = ax[3]
    if not _lines(p, rows, [("train/kl", "train/kl"), ("cap_gen/kl", "cap_gen/kl")],
                  marker=True):
        _empty(p, "no KL keys")
    p.set_title(f"KL to base ({roll})")
    p = ax[4]
    if not _lines(p, rows, [("cap_gen/acc", "GSM8K acc while hiding")], ylim=(-0.05, 1.05),
                  marker=True):
        _empty(p, "no cap_gen/acc")
    p.set_title(f"capability while hiding ({roll})")
    p = ax[5]
    if not _lines(p, rows, [("det_real_score_tN", "monitor-layer probe score, last turn"),
                            ("detho_real_score_tN", "held-out-layer probe score, last turn")],
                  marker=True):
        _empty(p, "no detho_real_score_tN")
    p.set_title(f"real-concept probe score at turn N ({roll})")
    p = ax[6]
    ks = keys_like(rows, r"kind_p_.+")
    if not _lines(p, rows, [(k, k.removeprefix("kind_p_")) for k in ks], w=1):
        _empty(p, "no kind_p_* keys (curriculum off)")
    p.set_title("probe-kind sampling weights")
    p = ax[7]
    ks = keys_like(rows, r"concept_p_.+")
    if not _lines(p, rows, [(k, k.removeprefix("concept_p_")[:28]) for k in ks], w=1):
        _empty(p, "no concept_p_* keys (curriculum off)")
    elif len(ks) > 12:
        p.legend(fontsize=5, ncol=2)
    p.set_title("concept sampling weights")
    p = ax[8]
    val = load_val(run)
    es = load_early_stop(run)
    if val:
        p.plot([s for s, _ in val], [v for _, v in val], marker="o", ms=3, label="val_score")
        sm = es.get("smoothed") or {}
        if sm:
            xs = sorted(int(k) for k in sm)
            p.plot(xs, [sm[str(x)] for x in xs], label="smoothed (early stop)")
        p.set_xlabel("step")
    if es.get("best_step") is not None:
        p.axvline(es["best_step"], color="k", ls="--", lw=1,
                  label=f"best step {es['best_step']}" + (" (stopped)" if es.get("stop") else ""))
    if val or es.get("best_step") is not None:
        p.legend(fontsize=7)
    else:
        _empty(p, "no val/step_*.json or early_stop.json")
    p.set_title("validation score")
    fig.suptitle(f"{name}: training curves")
    fig.tight_layout()
    outs = [outdir / f"{name}_train.png"]
    fig.savefig(outs[0], dpi=110)
    plt.close(fig)

    # per-family small multiples: beaten (left) + turns-to-beat (right, twin axis)
    fams = sorted({k.split("/")[1] for k in keys_like(rows, r"task_family/[^/]+/beaten")})
    concepts = sorted({k.split("/")[1] for k in keys_like(rows, r"task_concept/.+/beaten")})
    panels = [("family", f) for f in fams] or [("concept", c) for c in concepts]
    nc = min(4, max(1, len(panels)))
    nr = max(1, math.ceil(len(panels) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.4 * nr), squeeze=False)
    ax = ax.ravel()
    if not panels:
        _empty(ax[0], "no task_family/* or task_concept/* keys in this train log")
    for p, (kind, g) in zip(ax, panels, strict=False):
        xs, ys = series(rows, f"task_{kind}/{g}/beaten")
        p.plot(*smooth(xs, ys, ROLL), color="tab:blue", label="beaten")
        p.set_ylim(-0.05, 1.05)
        p.set_ylabel("fraction beaten", color="tab:blue", fontsize=8)
        q = p.twinx()
        xs, ys = series(rows, f"task_{kind}/{g}/turns_to_beat")
        if xs:
            q.plot(*smooth(xs, ys, ROLL), color="tab:orange", label="turns to beat")
        q.set_ylabel("turns to beat", color="tab:orange", fontsize=8)
        xs, ys = series(rows, f"task_{kind}/{g}/hazard")
        title = f"{kind} {g[:32]}"
        if ys:
            title += f"  (last hazard {smooth(xs, ys, ROLL)[1][-1]:.2f})"
        p.set_title(title, fontsize=9)
        p.set_xlabel("step")
    for p in ax[len(panels):] if panels else ax[1:]:
        p.set_visible(False)
    fig.suptitle(f"{name}: per-{panels[0][0] if panels else 'family'} beaten + turns-to-beat "
                 f"({roll})")
    fig.tight_layout()
    outs.append(outdir / f"{name}_families.png")
    fig.savefig(outs[1], dpi=110)
    plt.close(fig)
    return outs


# ---------------------------------------------------------------- 3) race plots
def km(records: list[dict]) -> tuple[list[int], list[float]]:
    """Kaplan-Meier cumulative beaten fraction 1 - S(k) at k = 0..max turns. A record is at risk
    at turn t iff turns >= t; beaten records are events at their 'turns', others censored."""
    recs = [r for r in records if _num(r.get("turns"))]
    if not recs:
        return [], []
    tmax = int(max(r["turns"] for r in recs))
    s, xs, ys = 1.0, [0], [0.0]
    for t in range(1, tmax + 1):
        n_t = sum(1 for r in recs if r["turns"] >= t)
        d_t = sum(1 for r in recs if r.get("beaten") and r["turns"] == t)
        if n_t:
            s *= 1 - d_t / n_t
        xs.append(t)
        ys.append(1 - s)
    return xs, ys


def race_labels(paths: list[Path]) -> list[str]:
    stems = [p.stem for p in paths]
    return [f"{p.parent.name}/{p.stem}" if stems.count(p.stem) > 1 else p.stem for p in paths]


def race_plots(paths: list[Path], outdir: Path, tag: str, plt) -> list[Path]:
    races = []
    for p, lab in zip(paths, race_labels(paths), strict=True):
        try:
            races.append((lab, json.loads(p.read_text())))
        except (OSError, json.JSONDecodeError) as e:
            print(f"# {p}: unreadable ({type(e).__name__}), skipped")
    if not races:
        return []
    outs = []
    fig, p = plt.subplots(figsize=(9, 5.5))
    n = 0
    for lab, r in races:
        recs = r.get("deploy_records")
        if not isinstance(recs, list) or not recs:
            print(f"# {lab}: no deploy_records (older JSON), not in the KM plot")
            continue
        xs, ys = km(recs)
        if not xs:
            continue
        cens = sum(1 for d in recs if not d.get("beaten"))
        p.step(xs, ys, where="post", label=f"{lab} (n={len(recs)}, cens={cens})")
        n += 1
    if n:
        p.legend(fontsize=7)
    else:
        _empty(p, "no race has deploy_records")
    p.set_ylim(-0.02, 1.02)
    p.set_xlabel("judged turns since the probe was deployed (k)")
    p.set_ylabel("fraction of new probes beaten by turn k (Kaplan-Meier)")
    p.set_title("turns to beat a newly deployed probe")
    p.grid(alpha=0.3)
    fig.tight_layout()
    outs.append(outdir / f"{tag}_km.png")
    fig.savefig(outs[-1], dpi=110)
    plt.close(fig)

    # per-concept bars: pass rate after the first turn + mean turns-to-beat (beaten records)
    per = []
    for lab, r in races:
        pr = r.get("pass_rate_after_first_turn_by_topic") or {}
        tt = defaultdict(list)
        for d in r.get("deploy_records") or []:
            if d.get("beaten") and d.get("concept") and _num(d.get("turns")):
                tt[d["concept"]].append(d["turns"])
        cs = sorted(set(pr) | set(tt))
        per.append((lab, cs, pr, {c: sum(v) / len(v) for c, v in tt.items()}))
    per = [x for x in per if x[1]]
    if not per:
        print("# no per-concept data in any race")
        return outs
    fig, ax = plt.subplots(len(per), 2, figsize=(16, 3.8 * len(per)), squeeze=False)
    for i, (lab, cs, pr, tt) in enumerate(per):
        xs = range(len(cs))
        short = [c[:22] for c in cs]
        p = ax[i][0]
        p.bar(xs, [pr.get(c, float("nan")) for c in cs], color="tab:blue")
        p.set_ylim(0, 1)
        p.set_title(f"{lab}: pass rate after the first turn", fontsize=9)
        p = ax[i][1]
        if tt:
            p.bar(xs, [tt.get(c, float("nan")) for c in cs], color="tab:orange")
        else:
            _empty(p, "no beaten probe with a concept in deploy_records")
        p.set_title(f"{lab}: mean judged turns to beat (beaten probes)", fontsize=9)
        for p in ax[i][: 2 if tt else 1]:
            p.set_xticks(list(xs))
            p.set_xticklabels(short, rotation=55, ha="right", fontsize=7)
    fig.tight_layout()
    outs.append(outdir / f"{tag}_concepts.png")
    fig.savefig(outs[-1], dpi=110)
    plt.close(fig)
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="legacy overview grid (needs --out)")
    ap.add_argument("--out", help="legacy grid PNG path")
    ap.add_argument("--run", action="append", default=[], help="training-curve plots of a run")
    ap.add_argument("--races", nargs="+", default=[], help="tica_switch_eval race JSONs")
    ap.add_argument("--outdir", default="runs/plots")
    ap.add_argument("--tag", default="races", help="file prefix of the race plots")
    a = ap.parse_args()
    if not (a.runs or a.run or a.races):
        ap.error("give RUN_DIRs + --out, --run, or --races")
    if a.runs and not a.out:
        ap.error("positional RUN_DIRs (legacy grid) need --out")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.runs:
        legacy([Path(r) for r in a.runs], a.out, plt)
    for run in a.run:
        for f in train_curves(Path(run), outdir, plt):
            print(f)
    if a.races:
        for f in race_plots([Path(r) for r in a.races], outdir, a.tag, plt):
            print(f)


if __name__ == "__main__":
    main()
