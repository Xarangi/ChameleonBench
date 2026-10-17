"""Health check over RUNNING (or finished) TICA window runs: read the logs, flag problems early.

  python scripts/tica_watch.py [run_glob ...] [--runs DIR] [--tail N]

For every train_log.jsonl under the run directories it prints one row per seed (step, sec/step,
losses, the pass rate at the last turn, the no-brief |f|) and then an ISSUES list. The checks are
the pre-registered bars plus the failure modes we have actually hit before, so a run that is going
wrong is visible while it is still running rather than at the end:

- win_mse not falling          -> the oracle target is not being learned (first vs last window)
- pass_real at the last turn   -> below the verdict-blind ceiling = no search; 1.00 within a few
                                  steps = a degenerate solution, look at the per-window breakdown
- no-brief |f| drifting up     -> the model is suppressing when it was not asked (A3c validity bar)
- nll rising                   -> the forced text is being lost (exact-repeat bar will fail)
- grad_norm non-finite / spiky -> instability
- step time / wall-clock       -> ETA against the SLURM time limit
It also greps the matching SLURM logs for tracebacks and CUDA OOM, and reports any eval.json found
(G and its CI), so a finished seed is summarized in the same table.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import subprocess
from pathlib import Path

RUNS = Path("/scratch/sneheel/next_chameleons_artifacts/tica/runs")
LOGS = Path("/home/sneheel/chameleons/logs")
def blind_ceiling(seed_dir: Path, turns: int) -> float:
    """1 / (number of windows): the best a verdict-blind policy can do per turn. Read from the
    run's own config.json -- deriving it from the turn count was wrong for any arm whose window
    count differs from the original S1/S2 (caught on the 3-window wide arm, 10-10)."""

    try:
        cfg = json.loads((seed_dir / "train" / "config.json").read_text())
        return 1.0 / max(1, len(cfg["env"]["windows"]))
    except Exception:
        return {3: 0.5}.get(turns, 0.25)


def read_jsonl(p: Path) -> list[dict]:
    out = []
    for line in p.read_text().splitlines():
        # a half-written last line on a running job is skipped
        with contextlib.suppress(json.JSONDecodeError):
            out.append(json.loads(line))
    return out


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def slurm_state() -> dict[str, tuple[str, str]]:
    """job name -> (state, elapsed) for this user's current jobs."""

    try:
        out = subprocess.run(
            ["squeue", "-u", "sneheel", "-h", "-o", "%j|%T|%M"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return {}
    d = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 3:
            d[parts[0]] = (parts[1], parts[2])
    return d


def scan_slurm_log(name: str, tail: int = 4000) -> list[str]:
    hits = []
    for f in sorted(LOGS.glob(f"{name}-*.out"), key=lambda p: p.stat().st_mtime)[-1:]:
        txt = f.read_text(errors="ignore")[-tail * 200 :]
        for pat, msg in (
            (r"Traceback \(most recent call last\)", "traceback"),
            (r"CUDA out of memory", "CUDA OOM"),
            (r"DUE TO TIME LIMIT", "hit the SLURM time limit"),
            (r"did not converge", "probe fit did not converge"),
        ):
            if re.search(pat, txt):
                hits.append(f"{f.name}: {msg}")
    return hits


def summarize_eval(p: Path) -> str:
    rep = json.loads(p.read_text())
    probes = rep.get("probes", {})
    bits = []
    for tag, r in probes.items():
        g = r.get("G_yoked")
        if g:
            bits.append(f"{tag} G {g['mean']:+.2f} [{g['lo']:+.2f},{g['hi']:+.2f}]")
        gc = r.get("gen_check", {}).get("min_exact")
        if gc is not None:
            bits.append(f"exact {gc:.2f}")
    return "  ".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=["l3*"])
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--window", type=int, default=20, help="steps averaged at each end")
    args = ap.parse_args()

    root = Path(args.runs)
    sq = slurm_state()
    rows, issues = [], []
    seen_runs = set()
    for g in args.globs or ["l3*"]:
        for run in sorted(root.glob(g)):
            if not run.is_dir():
                continue
            for seed_dir in sorted(run.glob("seed*")) or [run]:
                log = seed_dir / "train" / "train_log.jsonl"
                ev = list(seed_dir.glob("eval*/eval.json"))
                if not log.exists() and not ev:
                    continue
                tag = f"{run.name}/{seed_dir.name}"
                seen_runs.add(run.name)
                job = f"tica-{run.name.split('_')[0]}-{seed_dir.name.replace('seed', 's')}"
                state, elapsed = sq.get(job, ("-", "-"))
                if not log.exists():
                    rows.append(f"{tag:28s} {state:>8s} {elapsed:>8s}  (eval only)")
                    for e in ev:
                        rows.append(f"{'':28s}   {e.parent.name}: {summarize_eval(e)}")
                    continue
                r = read_jsonl(log)
                if not r:
                    continue
                w = min(args.window, max(1, len(r) // 3))
                first, last = r[:w], r[-w:]
                T = max(
                    (int(k.split("_t")[1]) for k in r[-1] if k.startswith("pass_real_t")),
                    default=0,
                ) + 1
                ceil = blind_ceiling(seed_dir, T)
                pk, nk = f"pass_real_t{T - 1}", f"absf_nobrief_t{T - 1}"
                p_last, p_first = mean([x.get(pk) for x in last]), mean([x.get(pk) for x in first])
                nb = mean([x.get(nk) for x in last])
                wm_f, wm_l = mean([x["win_mse"] for x in first]), mean([x["win_mse"] for x in last])
                sec = mean([x["sec"] for x in last])
                rows.append(
                    f"{tag:28s} {state:>8s} {elapsed:>8s}  step {r[-1]['step']:>3d}  "
                    f"{sec:5.0f}s/st  win_mse {wm_f:.3f}->{wm_l:.3f}  "
                    f"match {mean([x['match_mse'] for x in last]):.3f}  "
                    f"nll {mean([x['nll'] for x in last]):.3f}  "
                    f"pass_t{T - 1} {p_first:.2f}->{p_last:.2f} (blind {ceil:.2f})  "
                    f"nobrief|f| {nb:.3f}"
                )
                # Per-turn detection curve for the branch-refit loop: a falling curve is
                # in-context iteration, a flat low one is a static dodge.
                tk = sorted(
                    (k for k in r[-1] if k.startswith("rollout/detturn_real_score_")),
                    key=lambda k: int(k.rsplit("_", 1)[1]),
                )
                if tk:
                    rows.append(
                        f"{'':28s}   det by turn (real): "
                        + " ".join(f"t{k.rsplit('_', 1)[1]} {mean([x.get(k) for x in last]):.2f}"
                                   for k in tk)
                    )
                pw = sorted(k for k in r[-1] if k.startswith("pass_real_w"))
                if pw:
                    rows.append(
                        f"{'':28s}   per-window pass/f: "
                        + "  ".join(
                            f"w{k[-1]} {mean([x.get(k) for x in last]):.2f}/"
                            f"{mean([x.get('f_real_w' + k[-1]) for x in last]):+.2f}"
                            for k in pw
                        )
                    )
                for e in ev:
                    rows.append(f"{'':28s}   {e.parent.name}: {summarize_eval(e)}")

                n = len(r)
                if n >= 3 * w:
                    if wm_l > 0.9 * wm_f:
                        issues.append(f"{tag}: win_mse flat ({wm_f:.3f} -> {wm_l:.3f}) at step {n}")
                    if p_last <= ceil + 0.05:
                        issues.append(
                            f"{tag}: pass_t{T - 1} {p_last:.2f} is at/below the blind ceiling "
                            f"{ceil:.2f} at step {n} -- no search yet"
                        )
                    if mean([x["nll"] for x in last]) > 1.5 * mean([x["nll"] for x in first]):
                        issues.append(f"{tag}: nll rising -- the forced text is being lost")
                if nb > 0.10:
                    issues.append(f"{tag}: no-brief |f| {nb:.3f} > 0.10 (A3c validity bar)")
                if any(not math.isfinite(x.get("grad_norm", 0.0)) for x in r):
                    issues.append(f"{tag}: non-finite grad_norm")
                if p_last >= 0.99 and n <= 60:
                    issues.append(
                        f"{tag}: pass_t{T - 1} = {p_last:.2f} by step {n} -- suspiciously fast, "
                        f"check the per-window breakdown for a degenerate solution"
                    )

    print("\n".join(rows) if rows else "no runs found")
    for run in sorted(seen_runs):
        for seed in ("s0", "s1", "s2", "base"):
            issues += [
                f"{run}/{seed}: {h}" for h in scan_slurm_log(f"tica-{run.split('_')[0]}-{seed}")
            ]
    print("\nISSUES" if issues else "\nISSUES: none")
    for i in dict.fromkeys(issues):
        print(" -", i)


if __name__ == "__main__":
    main()
