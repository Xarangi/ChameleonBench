"""Analysis of TICA episode records (flat per-(episode, turn) dicts; DESIGN-TICA "readouts").

Record schema: episode ("cond|concept|id"), cond, turn, split, score_monitored, detected_monitored,
and optionally detected_text / detected_anchor.
- D = score(turn T) - score(turn 0) per episode. Contrasts compare conditions on the SAME episode
  suffix (concept|id) with a concept-cluster bootstrap 95% CI.
- contrasts_last (PRIMARY for fresh-text episodes): the same contrasts on the last-turn score
  itself. Conditions share their texts, so text difficulty cancels within a pair, whereas D mixes in
  the difficulty difference between text 0 and text T (review 10-03).
- per_turn_auroc: real_score vs neg_text monitored scores at each turn, plus the neg_text FPR at
  the turn-1 threshold.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

CONTRASTS = (
    ("real_score", "sham_score"),
    ("real_score", "none"),
    ("real_verdict", "none"),
    ("none", "nobrief"),
)


def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x:.2f}"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def per_turn_auroc(recs: list[dict]) -> dict[int, dict]:
    """Monitored-score AUROC at each turn: real_score (concept text) vs neg_text (same brief,
    concept-free text), and the neg_text false-positive rate at the turn-1 threshold."""

    from next_chameleons.tica.monitors import auroc as _auroc

    auroc = lambda p, n: _auroc(torch.as_tensor(p), torch.as_tensor(n))  # noqa: E731
    out = {}
    for t in sorted({r["turn"] for r in recs}):
        pos = [r["score_monitored"] for r in recs if r["cond"] == "real_score" and r["turn"] == t]
        neg = [r for r in recs if r["cond"] == "neg_text" and r["turn"] == t]
        if pos and neg:
            out[t] = {
                "auroc": float(auroc(np.array(pos), np.array([r["score_monitored"] for r in neg]))),
                "fpr": float(np.mean([r["detected_monitored"] for r in neg])),
            }
    return out


def deltas(recs: list[dict], key: str = "score_monitored") -> dict[str, dict[str, float]]:
    """cond -> {episode suffix (concept|tid) -> D}."""

    by = defaultdict(dict)
    T = max(r["turn"] for r in recs)
    for r in recs:
        by[(r["cond"], r["episode"].split("|", 1)[1])][r["turn"]] = r[key]
    out = defaultdict(dict)
    for (cond, ep), v in by.items():
        out[cond][ep] = v[T] - v[0]
    return out


def last_values(recs: list[dict], key: str = "score_monitored") -> dict[str, dict[str, float]]:
    """cond -> {episode suffix -> value at the last turn}."""

    T = max(r["turn"] for r in recs)
    out = defaultdict(dict)
    for r in recs:
        if r["turn"] == T:
            out[r["cond"]][r["episode"].split("|", 1)[1]] = r[key]
    return out


def boot_contrast(a: dict[str, float], b: dict[str, float], n: int = 2000, seed: int = 0):
    eps = sorted(set(a) & set(b))
    concept = np.array([e.split("|")[0] for e in eps])
    diff = np.array([a[e] - b[e] for e in eps])
    uc = np.unique(concept)
    groups = [diff[concept == c] for c in uc]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(uc), len(uc))
        stats.append(np.concatenate([groups[i] for i in pick]).mean())
    return (
        float(diff.mean()),
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
        len(eps),
    )


def summarize(allr: list[dict]) -> dict:
    """Per split (all/train/heldout): per-turn table, AUROC by turn, and D contrasts."""

    rep = {}
    for split in ("all", "train", "heldout"):
        recs = [r for r in allr if split == "all" or r.get("split") == split]
        if not recs:
            continue
        table = defaultdict(lambda: defaultdict(list))
        for r in recs:
            table[r["cond"]][r["turn"]].append(r)
            if r.get("probe_match") is not None:
                # Hidden mode: episodes whose probe turn showed the monitored concept (A=M,
                # turn-0 verdict DETECTED) vs the partner (A!=M, NOT DETECTED). "Hide the
                # probe-turn concept" solves only the first; the second needs the inference.
                table[f"{r['cond']}|{'A=M' if r['probe_match'] else 'A!=M'}"][r["turn"]].append(r)
        per = {
            cond: {
                t: {
                    "score": float(np.mean([r["score_monitored"] for r in v])),
                    "det": float(np.mean([r["detected_monitored"] for r in v])),
                    "det_text": _mean([r.get("detected_text") for r in v]),
                    "det_keep": _mean([r.get("detected_keep") for r in v]),
                    "det_anchor": _mean([r.get("detected_anchor") for r in v]),
                    "placebo": _mean([r.get("placebo") for r in v]),
                    "n": len(v),
                }
                for t, v in sorted(turns.items())
            }
            for cond, turns in table.items()
        }
        D = deltas(recs)
        con = {f"{a} - {b}": boot_contrast(D[a], D[b]) for a, b in CONTRASTS if a in D and b in D}
        V = last_values(recs)
        con_last = {
            f"{a} - {b}": boot_contrast(V[a], V[b]) for a, b in CONTRASTS if a in V and b in V
        }
        # Hidden mode: the keep readout lives on the partner's turns (None elsewhere); report the
        # latest turn that has one, per condition.
        keep_last = {}
        for cond, turns in table.items():
            ts = [t for t, v in turns.items() if any(r.get("detected_keep") is not None for r in v)]
            if ts:
                keep_last[cond] = {
                    "turn": max(ts),
                    "det_keep": _mean([r.get("detected_keep") for r in turns[max(ts)]]),
                }
        rep[split] = {
            "auroc_by_turn": per_turn_auroc(recs),
            "per_turn": per,
            "keep_last": keep_last,
            "contrasts_last": con_last,
            "contrasts_D": con,
            "mean_D": {c: float(np.mean(list(v.values()))) for c, v in D.items()},
        }
    return rep


def extra_detection(recs: list[dict]) -> dict:
    """Detection rate of every extra readout (detected_extra_*), by condition and turn."""

    keys = sorted({k for r in recs for k in r if k.startswith("detected_extra_")})
    out: dict = {}
    for k in keys:
        acc: dict = defaultdict(list)
        for r in recs:
            if r.get(k) is not None:
                acc[(r["cond"], r["turn"])].append(float(r[k]))
        out[k[len("detected_") :]] = {
            f"{c}|t{t}": float(np.mean(v)) for (c, t), v in sorted(acc.items())
        }
    return out


def report_lines(rep: dict) -> list[str]:
    a = rep["all"]
    first = next(iter(a["per_turn"].values()))
    lines = [f"{'cond':14s} " + " ".join(f"t{t}:score/det" for t in sorted(first))]
    for cond, turns in a["per_turn"].items():
        last = list(turns.values())[-1]
        # Hidden mode: keep lives on the partner's turns, not the last one.
        keep = a.get("keep_last", {}).get(cond, {}).get("det_keep", last.get("det_keep"))
        lines.append(
            f"{cond:14s} "
            + " ".join(f"{v['score']:7.1f}/{v['det']:.2f}" for v in turns.values())
            + f"   text-det tT {_fmt(last['det_text'])}  keep-det {_fmt(keep)}"
            + f"  anchor-det {_fmt(last['det_anchor'])}"
        )
    for t, v in a["auroc_by_turn"].items():
        lines.append(
            f"turn {t}: AUROC real_score vs neg_text {v['auroc']:.3f}; neg_text FPR {v['fpr']:.3f}"
        )
    for split in ("all", "train", "heldout"):
        if split not in rep:
            continue
        for k, (m, lo, hi, n) in rep[split]["contrasts_last"].items():
            lines.append(f"[{split}] last-turn ({k}) = {m:+.2f} [{lo:+.2f}, {hi:+.2f}] n={n}")
        for k, (m, lo, hi, n) in rep[split]["contrasts_D"].items():
            lines.append(f"[{split}] D({k}) = {m:+.2f} [{lo:+.2f}, {hi:+.2f}] n={n}")
    return lines
