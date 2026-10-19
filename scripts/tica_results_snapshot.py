"""Aggregate-only TICA results snapshot (docs/tica/RESULTS.md), so the numbers travel with git to
another cluster without the run directories. Never copies text: only numbers, booleans, short
labels (concept / kind / layer names) and nested dicts / numeric lists survive; records, configs,
paths and anything text-like are dropped, and the output is checked for long strings.

  python scripts/tica_results_snapshot.py RUN[:OFFSET] ... [--out results/tica/snapshot_<date>.json]

Per run: a fresh run report (tica/run_report.report; OFFSET = TICA_STEP_OFFSET of a branch) plus the
aggregate fields of the eval JSONs under val/, sanity_full/, panel/, evdet/, eval_*/ and report/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GLOBS = ("val/step_*.json", "val/race_step_*.json", "sanity_full/*.json", "panel/*.json",
         "evdet/detector.json", "eval_*/*.json", "eval_*/long/*.json")
# keys whose values are records / text / paths / per-example detail, never aggregates
DROP = {"data", "text", "texts", "records", "deploy_records", "refit_details", "config", "adapter",
        "race", "files", "inputs", "run", "chunks", "request", "prompt", "samples", "examples",
        "recovery_turns", "judge_fpr", "refit_secs", "refit_kinds", "verdicts", "panel_failed",
        "manifest", "cmd", "command", "error", "errors", "per_item"}
MAX_STR = 80
MAX_LIST = 64  # longer numeric lists are per-example detail (per-item correctness), not aggregates


def clean(x):
    """Aggregates only: drop DROP keys, long strings, per-record and long lists."""

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if str(k) in DROP:
                continue
            c = clean(v)
            if c is not None:
                out[str(k)] = c
        return out or None
    if isinstance(x, list):
        if len(x) > MAX_LIST or any(isinstance(v, dict) for v in x):
            return None  # per-record / per-example lists
        items = [clean(v) for v in x]
        return items if all(v is not None for v in items) else None
    if isinstance(x, str):
        return x if len(x) <= MAX_STR and "/" not in x else None
    if isinstance(x, (bool, int, float)) or x is None:
        return x
    return None


def longest_str(x) -> int:
    if isinstance(x, dict):
        return max([len(str(k)) for k in x] + [longest_str(v) for v in x.values()], default=0)
    if isinstance(x, list):
        return max((longest_str(v) for v in x), default=0)
    return len(x) if isinstance(x, str) else 0


def snapshot_run(run: Path, offset: int) -> dict:
    from next_chameleons.tica.run_report import report

    out: dict = {}
    try:
        out["run_report"] = clean(report(run, offset, window=20, coarse=60))
    except Exception as e:  # eval-only dirs have no training logs
        out["run_report_error"] = type(e).__name__
    files = {}
    for g in GLOBS:
        for p in sorted(run.glob(g)):
            try:
                c = clean(json.loads(p.read_text()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if c:
                files[str(p.relative_to(run))] = c
    out["evals"] = files
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="RUN_DIR[:TICA_STEP_OFFSET]")
    ap.add_argument("--out", default=f"results/tica/snapshot_{datetime.date.today()}.json")
    a = ap.parse_args()
    doc = "aggregate-only TICA results (scripts/tica_results_snapshot.py; docs/tica/RESULTS.md)"
    snap = {"_doc": doc,
            "date": str(datetime.date.today()), "runs": {}}
    for spec in a.runs:
        path, _, off = spec.partition(":")
        run = Path(path)
        snap["runs"][run.name] = {"base_offset": int(off or 0), **snapshot_run(run, int(off or 0))}
        print(run.name, len(snap["runs"][run.name]["evals"]), "eval files", flush=True)
    n = longest_str(snap)
    assert n <= max(MAX_STR, len(snap["_doc"])), f"string of length {n} in the snapshot"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(snap, indent=1, sort_keys=True))
    print("wrote", a.out, Path(a.out).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
