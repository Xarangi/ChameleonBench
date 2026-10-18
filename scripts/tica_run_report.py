"""TRAINING-RUN REPORT (tica/run_report.py): every organism-level curve of a TICA training run
across its resume chunks, as one JSON (+ --plots: PNGs into <run>/report/, never committed).
Aggregates only.

  python scripts/tica_run_report.py --run RUN_DIR [--base-offset 360] [--families SPLIT.json]
         [--window 20] [--coarse 80] [--out O.json] [--plots]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from next_chameleons.tica.run_report import report

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # dataviz slots 1-6


def _mid(w: str) -> float:
    a, b = map(int, w.split("-"))
    return (a + b) / 2


def plots(rep: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)

    def lines(series: dict, title: str, ylabel: str, name: str):
        fig, ax = plt.subplots(figsize=(7, 4))
        for (lab, pts), col in zip(series.items(), SERIES, strict=False):
            pts = [(x, y) for x, y in pts if y is not None]
            if pts:
                ax.plot(*zip(*pts, strict=True), color=col, lw=2, marker="o", ms=4, label=lab)
        ax.set_title(title, loc="left")
        ax.set_xlabel("global training step")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / name, dpi=120)
        plt.close(fig)

    pi = rep["probe_index"]["by_index"]
    keys = ["k1_first", "k1_switch", "k2", "k3", "k4"]
    ok = lambda v: v.get("beaten") if v.get("n", 0) >= 10 else None  # noqa: E731 (small cells)
    lines({k: [(_mid(w), ok(d.get(k, {}))) for w, d in pi.items()] for k in keys},
          "Pass rate by probe index", "segments that beat bank k", "probe_index.png")
    lines({"retry pass": [(_mid(w), d["pass"]) for w, d in rep["retry_pass"].items()]},
          "Pass on the turn after a catch (same bank)", "pass rate", "retry_pass.png")
    ho = rep["heldout_static"]
    conds = sorted({c for d in ho.values() for c in d})
    lines({c: [(_mid(w), d.get(c, {}).get("det")) for w, d in ho.items()] for c in conds[:6]},
          "Held-out static probes (L20/24), detection", "detected", "heldout_static.png")
    rr = rep["refit_readability"]["by_window"]
    lines({"refit AUROC < 0.99": [(_mid(w), d["frac_lt_099"]) for w, d in rr.items()],
           "refit kept": [(_mid(w), d["kept"]) for w, d in rr.items()]},
          "Refit readability on the policy's states", "fraction", "refit_readability.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--base-offset", type=int, default=0,
                    help="global step of <run>/adaptive's step 0 (branches: TICA_STEP_OFFSET)")
    ap.add_argument("--families", help="split json with a 'families' map (concept -> family)")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--coarse", type=int, default=80)
    ap.add_argument("--out")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    fam = json.loads(Path(a.families).read_text())["families"] if a.families else None
    rep = report(a.run, a.base_offset, a.window, a.coarse, fam)
    out = Path(a.out) if a.out else a.run / "report" / "run_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    if a.plots:
        plots(rep, out.parent)
    pi = rep["probe_index"]["by_index"]
    for w in list(pi)[-4:]:
        print(w, " | ".join(f"{k}: {v['beaten']} (n={v['n']})" for k, v in pi[w].items()),
              "| retry", rep["retry_pass"].get(w))
    print("wrote", out)


if __name__ == "__main__":
    main()
