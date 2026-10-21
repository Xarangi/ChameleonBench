"""Verify every input of the TamIA v4b runs (TamIA 10-21): files parse, splits match the git
split, every text has a request and a 27B label, the capability pool / GSM8K / Dolly / probe dir
look like Narval's. Aggregates only.

  python scripts/tamia_verify_data.py --config final_v4b_4b [--config final_v4b_score_4b]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from next_chameleons.tica.config import load_run_config

RAW = Path("/scratch/s/sneheel/next_chameleons_raw_cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    a = ap.parse_args()
    bad = []

    def ok(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            bad.append(msg)

    for name in a.config:
        cfg = load_run_config(name)
        acfg = cfg.rl.adaptive
        print(f"== {name}  (out {cfg.out})")
        from next_chameleons.tica.answer_format import text_id
        from next_chameleons.tica.data import Level1Data

        d = Level1Data(cfg.data)
        split_git = json.loads(Path("configs/tica/final_split_v5.json").read_text())
        n_tr = sorted(c for c, s in d.split.items() if s == "train")
        n_ho = sorted(c for c, s in d.split.items() if s == "heldout")
        ok(n_tr == sorted(split_git["train"]), f"train concepts {len(n_tr)} == git split v5 (20)")
        ok(n_ho == sorted(split_git["heldout"]),
           f"held-out concepts {len(n_ho)} == git split v5 (38)")
        def _txt(x):
            return x if isinstance(x, str) else (x.get("text") or x.get("answer") or "")

        all_texts = set()
        for c in d.raw["concepts"]:
            for part in ("fit", "calib", "episode"):
                all_texts.update(t for t in map(_txt, c.get(part, [])) if t)
        for xs in d.raw.get("negatives", {}).values():
            all_texts.update(t for t in map(_txt, xs) if t)
        print(f"  texts in data (positives + negatives): {len(all_texts)}")
        inputs = json.loads(Path(cfg.episodes.inputs).read_text())
        print(f"  elicit inputs: {len(inputs)} entries")
        labels = json.loads(Path(acfg.neg_labels).read_text())
        lab = labels.get("labels", labels)
        print(f"  27B labels: {len(lab)} texts labelled; meta {str(labels.get('meta', ''))[:120]}")
        ids = {text_id(x) for x in all_texts}
        cov = len(ids & set(lab)) / max(1, len(ids))
        # Narval's label file covers 0.932 of the texts (unlabelled = clean, as on Narval): the
        # missing ones are mostly shared negatives and DolusChat (10-21 check)
        ok(cov >= 0.93, f"27B labels cover {cov:.3f} of the data's texts (Narval file: 0.932)")
        miss_req = sum(1 for x in all_texts if x not in inputs and text_id(x) not in inputs)
        print(f"  texts without an elicit request (by text or id): {miss_req}")
        pool = json.loads(Path(acfg.cap_gen_pool).read_text())
        cnt = {k: (len(v), sum(bool(r.get("ok")) for r in v))
               for k, v in pool.items() if isinstance(v, list)}
        print(f"  cap_gen_pool rows (n, ok): {cnt}")
        ok(all(n > 0 for n, _ in cnt.values()) and {"probe", "np", "fork"} <= set(cnt),
           "cap_gen_pool probe/np/fork")
        gsm = json.loads((RAW / "capability/gsm8k_train.json").read_text())
        ok(len(gsm) == 7473, f"gsm8k_train {len(gsm)} items (7473)")
        dl = RAW / "tica/dolly15k/databricks-dolly-15k.jsonl"
        nd = sum(1 for line in dl.open() if line.strip())
        ok(nd >= 15000, f"dolly {nd} rows (15011)")
        q = json.loads((Path(cfg.probes.dir) / "quality.json").read_text())
        ok(q.get("primary_layer") == 12, f"probes primary_layer {q.get('primary_layer')} (12)")
        for L in (12, 20, 24):
            ok((Path(cfg.probes.dir) / f"bank_L{L}.npz").stat().st_size > 1e6, f"bank_L{L}.npz")
        fam = json.loads(Path(acfg.concept_families).read_text())
        ok(set(fam["families"]) >= set(split_git["train"]) | set(split_git["heldout"]),
           "families cover all concepts")
        print(f"  family counts (train): {Counter(fam['families'][c] for c in split_git['train'])}")
    print("ALL OK" if not bad else f"{len(bad)} FAILED: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
