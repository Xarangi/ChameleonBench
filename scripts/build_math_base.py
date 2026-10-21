"""Teacher-forced MATH dodger data (REPLICATION 2026-10-14r; user: LLM-generated responses, loss on
correctness, same setting as the dodger). Answers = the BASE model's own GSM8K-TRAIN solutions,
kept only when finished, correct (exact final answer) and showing working (mathrl.build_base_gens,
cached in runs/mathrl_4b/base_gens.json). Each is the answer to its own problem (request = the
mathrl solve prompt). Teacher-forcing these with the LM loss trains "keep producing correct
solutions" while the hide loss trains "do not represent math" -- the combined objective.
Output: a Level1Data JSON = the topic base (other concepts for fork negatives, unchanged splits)
+ one TRAIN concept "mathematical reasoning" (fit / calib / episode disjoint), and an inputs JSON
(topic-base inputs + one request per math solution). Aggregates only.

  python scripts/build_math_base.py --out-data D.json --out-inputs I.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

RAW = Path("/scratch/s/sneheel/next_chameleons_raw_cache/tica")
GENS = Path("/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mathrl_4b/base_gens.json")
PROMPT = ("Solve the following problem. Show your working briefly, then finish with "
          "\"The answer is <number>.\"\n\nProblem: {q}")  # == tica.trainers.mathrl.PROMPT
CONCEPT = "mathematical reasoning"


def text_id(text: str) -> str:  # == tica.answer_format.text_id
    return "ax-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics-data", default=str(RAW / "topic_base_v1.json"))
    ap.add_argument("--topics-inputs", default=str(RAW / "topic_base_v1_inputs.json"))
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--episode", type=int, default=200)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    a = ap.parse_args()
    g = json.loads(GENS.read_text())
    rows = [r for part in ("probe", "np", "fork") for r in g[part] if r["ok"]]
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            uniq.append(r)
    random.Random(0).shuffle(uniq)
    ep, cal = uniq[: a.episode], uniq[a.episode : a.episode + a.calib]
    fit = uniq[a.episode + a.calib :]
    data = json.loads(Path(a.topics_data).read_text())
    inputs = json.loads(Path(a.topics_inputs).read_text())
    clash = 0
    for r in uniq:
        k = text_id(r["text"])
        clash += k in inputs and inputs[k] != PROMPT.format(q=r["q"])
        inputs[k] = PROMPT.format(q=r["q"])
    part = lambda rs: [{"id": text_id(r["text"]), "text": r["text"]} for r in rs]  # noqa: E731
    data["concepts"].append({"concept_id": "math-0", "concept": CONCEPT, "genre": "math",
                             "split": "train", "fit": part(fit), "calib": part(cal),
                             "episode": part(ep)})
    data["source"] = {**data.get("source", {}), "math": str(GENS),
                      "math_request": PROMPT}
    Path(a.out_data).write_text(json.dumps(data))
    Path(a.out_inputs).write_text(json.dumps(inputs))
    print(json.dumps({"math_solutions": len(uniq), "fit": len(fit), "calib": len(cal),
                      "episode": len(ep), "input_clashes": clash,
                      "base_acc_train": g["stats"]["base_acc_train"],
                      "concepts": len(data["concepts"])}))


if __name__ == "__main__":
    main()
