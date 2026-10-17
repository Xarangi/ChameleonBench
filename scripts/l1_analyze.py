"""TICA Level 1: summarize a base test or trained-model eval (records.jsonl).

A thin wrapper around tica/analysis.py, so base tests (scripts/l1_base_test.py) and trained-model
evals (scripts/tica.py eval) are analyzed by the same code:
- per condition x turn: monitored score and detection, text-probe and anchor detection, placebo;
- per-turn AUROC real_score vs neg_text and neg_text FPR;
- contrasts (concept-cluster bootstrap 95% CI): last-turn score (primary) and D = score(T) -
  score(0) (secondary), for real - sham, real - none, real_verdict - none, none - nobrief.
Splits (all / train / heldout) are reported apart.

Usage: python scripts/l1_analyze.py --records FILE [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from next_chameleons.tica import analysis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    recs = [json.loads(line) for line in args.records.open()]
    rep = analysis.summarize(recs)
    print("\n".join(analysis.report_lines(rep)))
    if args.out:
        args.out.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
