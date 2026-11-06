"""11-06 GRADED race tasks (user: graded tests wherever possible): dl/*.rows.json (fetched on the
login node from the HF datasets-server rows API; provenance dl/fetch_1106_provenance*.json) ->
capability/<bench>.json + heldout_manifest.json entries. Offline; small JSON only.

  python scripts/materialize_graded_tasks.py [--cap /scratch/.../capability]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from next_chameleons.benchmarks import capability_heldout as ch

CAP = Path("/scratch/s/sneheel/next_chameleons_raw_cache/capability")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default=str(CAP))
    cap = Path(ap.parse_args().cap)
    dl = cap / "dl"
    out = {}
    out["mbpp_java"] = ch.materialize_humaneval_java(dl / "mbpp_java/test.rows.json", cap,
                                                     "28441b6", bench="mbpp_java",
                                                     config="mbpp-java")
    for lang in ("de", "ja"):
        out[f"mgsm_{lang}"] = ch.materialize_mgsm(dl / f"mgsm/{lang}_test.rows.json", cap, lang,
                                                  "b2f13d4")
    for lang, (cfg, _) in ch.BELEBELE_LANGS.items():
        out[f"belebele_{lang}"] = ch.materialize_belebele(dl / f"belebele/{cfg}_test.rows.json",
                                                          cap, lang, "7899cdf")
    rows = json.loads((cap / "mmlu_stem.json").read_text())  # physics + cs (cached, full subjects)
    for f in sorted((dl / "mmlu_extra").glob("*_test.rows.json")):
        rows += [{**r, "subject": r.get("subject") or f.name.split("_test")[0]}
                 for r in json.loads(f.read_text())]
    for topic in ch.MMLU_TOPICS:
        out[f"mmlu_{topic}"] = ch.materialize_mmlu_topic(rows, cap, topic, "c30699e")
    print(json.dumps({k: v["n"] for k, v in out.items()}))


if __name__ == "__main__":
    main()
