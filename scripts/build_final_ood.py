"""EVAL-ONLY out-of-distribution DOMAIN concepts for the FINAL model (final_v2_4b). Published data
only.

final_base_v2 (concepts + negatives, unchanged) + four concepts in a NEW concept split "ood" (genre
"domain"), never trained and never used as negatives by training:
- "mathematical reasoning": math_base_v1's texts and requests unchanged (the base model's correct
  GSM8K-train solutions; fit / calib / episode as in that file);
- "physics", "chemistry": camel-ai/physics, camel-ai/chemistry (CC-BY-NC-4.0; request = message_1,
  text = message_2);
- "computer code": sahil2801/CodeAlpaca-20k (CC-BY-4.0; request = instruction [+ input], text =
  output), code-like multi-line outputs only.
The three new sources: answers of 150-1500 chars (the Dolly negatives' band), exact-dedupe, --n
sampled with a fixed seed, split 60 / 20 / 20 (fit / calib / episode).

NEGATIVES FILTER: final_base_v2's broad negatives hold NC rated "mathematical" and "chemistry-based"
texts, labelled concept-free; they are recomputed with build_final_base.nc_negatives (same files,
filters and caps) and dropped. Negatives that exactly equal a new concept text are dropped too.
Aggregates only (no text is printed).

  python scripts/build_final_ood.py --out-data D.json --out-inputs I.json --out-split S.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_final_base import RAW, nc_negatives, split3, text_id

TICA = RAW / "tica"
OOD = TICA / "ood_domains"  # downloaded from the HF hub (resolve/main), one directory per source
SOURCES = {
    "physics": ("camel", OOD / "camel_physics" / "physics.zip",
                "camel-ai/physics (CC-BY-NC-4.0)"),
    "chemistry": ("camel", OOD / "camel_chemistry" / "chemistry.zip",
                  "camel-ai/chemistry (CC-BY-NC-4.0)"),
    "computer code": ("codealpaca", OOD / "codealpaca20k" / "code_alpaca_20k.json",
                      "sahil2801/CodeAlpaca-20k (CC-BY-4.0)"),
}
DROP_NC = ("mathematical", "chemistry-based")  # NC adjectives that ARE the new concepts
CODE_CHARS = set("=(){}[];:<>")


GSM_TEST = OOD / "gsm8k_test" / "test.json"  # openai/gsm8k main test (1319), downloaded 10-16
GSM_EVAL = RAW / "capability" / "gsm8k.json"  # the 200 capability-eval questions (excluded)


def _gsm_text(ans: str) -> str:
    body, _, final = ans.partition("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nThe answer is {final.strip()}."


def code_like(x: str) -> bool:
    """>= 3 non-empty lines, at least half of them carrying code punctuation."""

    lines = [s for s in x.splitlines() if s.strip()]
    return len(lines) >= 3 and sum(bool(CODE_CHARS & set(s)) for s in lines) >= len(lines) / 2


def load_rows(kind: str, path: Path) -> list[tuple[str, str]]:
    """(request, text) pairs of a source, file order."""

    if kind == "camel":
        z = zipfile.ZipFile(path)
        rows = [json.loads(z.read(n)) for n in sorted(z.namelist()) if n.endswith(".json")]
        return [(r["message_1"].strip(), r["message_2"].strip()) for r in rows]
    rows = json.loads(path.read_text())
    return [(r["instruction"].strip() + (f"\n\n{r['input'].strip()}" if r.get("input", "").strip()
                                         else ""), r["output"].strip()) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", default=str(TICA / "final_base_v2.json"))
    ap.add_argument("--base-inputs", default=str(TICA / "final_base_v2_inputs.json"))
    ap.add_argument("--base-split", default="configs/tica/final_split_v2.json")
    ap.add_argument("--math-data", default=str(TICA / "math_base_v1.json"))
    ap.add_argument("--math-inputs", default=str(TICA / "math_base_v1_inputs.json"))
    ap.add_argument("--math-concept", default="mathematical reasoning")
    ap.add_argument("--math-source", default="base_gens", choices=["base_gens", "gsm8k_test"],
                    help="base_gens = math_base_v1 (base-model GSM8K-TRAIN solutions; final_v2 "
                         "trained on these problems via cap_gen); gsm8k_test = the published "
                         "human-written GSM8K-TEST solutions (calculator annotations removed, "
                         "'#### N' -> 'The answer is N.'), minus the capability-eval questions")
    ap.add_argument("--n", type=int, default=600, help="texts per new published concept")
    ap.add_argument("--min-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    ap.add_argument("--out-split", required=True)
    a = ap.parse_args()
    data = json.loads(Path(a.base_data).read_text())
    inputs = json.loads(Path(a.base_inputs).read_text())
    split = json.loads(Path(a.base_split).read_text())
    if "ood" in split or any(c["split"] == "ood" for c in data["concepts"]):
        raise SystemExit("base already has an 'ood' split")
    ent = lambda xs: [{"id": text_id(x), "text": x} for x in xs]  # noqa: E731
    report: dict = {}

    # 1. negatives: the NC rows of the new domains, recomputed exactly as build_final_base did
    nc = list(nc_negatives())
    neg_ids = {p: {t["id"] for t in data["negatives"][p]} for p in data["negatives"]}
    all_neg = set().union(*neg_ids.values())
    nc_ids = {text_id(x) for _, x, _ in nc}
    drop = {text_id(x) for _, x, c in nc if c in DROP_NC}
    report["nc_negatives_recomputed"] = {"rows": len(nc), "unique": len(nc_ids),
                                         "found_in_base": len(nc_ids & all_neg),
                                         "drop_unique": len(drop),
                                         "per_adjective": {c: sum(1 for *_, d in nc if d == c)
                                                           for c in DROP_NC}}
    if nc_ids - all_neg:
        raise SystemExit(f"{len(nc_ids - all_neg)} recomputed NC negatives not in the base: "
                         "the selection is not reproduced")

    # 2. new concepts
    existing = {t["text"] for c in data["concepts"] for p in ("fit", "calib", "episode")
                for t in c[p]}
    new, clash = [], 0
    if a.math_source == "gsm8k_test":  # clean control: problems never seen in training
        evalq = {r["question"] for r in json.loads(GSM_EVAL.read_text())}
        rows = [(r["question"], _gsm_text(r["answer"])) for r in json.loads(GSM_TEST.read_text())
                if r["question"] not in evalq]
        rows = list(dict((x, q) for q, x in rows).items())
        rng_m = random.Random(f"{a.seed}-gsm8k_test")
        rng_m.shuffle(rows)
        mi = {text_id(x): q for x, q in rows}
        fit, cal, ep = split3([x for x, _ in rows], rng_m)
        mparts = {"fit": fit, "calib": cal, "episode": ep}
        report["math_gsm8k_test"] = {"rows": len(rows), "eval_questions_excluded": len(evalq)}
    else:
        md = json.loads(Path(a.math_data).read_text())
        mi = json.loads(Path(a.math_inputs).read_text())
        mc = next(c for c in md["concepts"] if c["concept"] == a.math_concept)
        mparts = {p: [t["text"] for t in mc[p]] for p in ("fit", "calib", "episode")}
    for x in (x for xs in mparts.values() for x in xs):
        k = text_id(x)
        clash += k in inputs and inputs[k] != mi[k]
        inputs[k] = mi[k]
    report["math_overlap_existing_concepts"] = sum(x in existing for xs in mparts.values()
                                                   for x in xs)
    new.append(("ood-math", a.math_concept, mparts,
                "GSM8K test, human-written solutions (MIT)" if a.math_source == "gsm8k_test"
                else f"math_base_v1 ({Path(a.math_data).name}: base-model GSM8K-train "
                     "solutions; GSM8K MIT)"))
    taken = set(existing) | {x for xs in mparts.values() for x in xs}
    for name, (kind, path, lic) in SOURCES.items():
        rows = load_rows(kind, path)
        st = {"rows": len(rows)}
        rows = [(q, x) for q, x in rows if q and a.min_len <= len(x) <= a.max_len]
        st["len_ok"] = len(rows)
        if kind == "codealpaca":
            rows = [(q, x) for q, x in rows if code_like(x)]
            st["code_like"] = len(rows)
        rows = list(dict((x, q) for q, x in rows).items())  # dedupe texts (last request wins)
        st["unique"] = len(rows)
        st["overlap_existing_or_new_dropped"] = sum(x in taken for x, _ in rows)
        rows = [(x, q) for x, q in rows if x not in taken]
        if len(rows) < a.n:
            raise SystemExit(f"{name}: only {len(rows)} texts (< {a.n})")
        rng = random.Random(f"{a.seed}-{name}")
        rows = rng.sample(rows, a.n)
        for x, q in rows:
            k = text_id(x)
            clash += k in inputs and inputs[k] != q
            inputs[k] = q
            taken.add(x)
        fit, cal, ep = split3([x for x, _ in rows], rng)
        new.append((f"ood-{name.replace(' ', '-')}", name, {"fit": fit, "calib": cal,
                                                             "episode": ep}, lic))
        report.setdefault("filter", {})[name] = st
    for cid, name, parts, _ in new:
        data["concepts"].append({"concept_id": cid, "concept": name, "genre": "domain",
                                 "split": "ood", **{p: ent(xs) for p, xs in parts.items()}})
    report["input_clashes_overwritten"] = clash

    # 3. negatives: drop the NC math / chemistry rows and any exact copy of a new concept text
    new_ids = {text_id(x) for _, _, parts, _ in new for xs in parts.values() for x in xs}
    dropped = {}
    for p, ts in data["negatives"].items():
        keep = [t for t in ts if t["id"] not in drop and t["id"] not in new_ids]
        dropped[p] = {"nc_math_chem": sum(t["id"] in drop for t in ts),
                      "equal_to_new_concept_text": sum(t["id"] in new_ids and t["id"] not in drop
                                                       for t in ts),
                      "kept": len(keep)}
        data["negatives"][p] = keep
    report["negatives_dropped"] = dropped

    meta = {name: {"family": "domain", "source": lic} for _, name, _, lic in new}
    data["source"] = {**data.get("source", {}), "ood_concepts": meta,
                      "ood_negatives_dropped": dropped,
                      "ood_builder": "scripts/build_final_ood.py"}
    Path(a.out_data).write_text(json.dumps(data))
    Path(a.out_inputs).write_text(json.dumps(inputs))
    out_split = {**split, "_doc": "FINAL OOD split v1 (scripts/build_final_ood.py): final_split_v2 "
                 "+ 'ood' = EVAL-ONLY domain concepts never trained (math = math_base_v1 GSM8K "
                 "base solutions, physics / chemistry = camel-ai, code = CodeAlpaca-20k). NC "
                 "'mathematical' / 'chemistry-based' broad negatives removed.",
                 "ood": [name for _, name, _, _ in new],
                 "families": {**split.get("families", {}),
                              **{name: "domain" for _, name, _, _ in new}}}
    Path(a.out_split).write_text(json.dumps(out_split, indent=1) + "\n")
    lens = {}
    for _, name, parts, _ in new:
        L = sorted(len(x) for xs in parts.values() for x in xs)
        lens[name] = {**{p: len(xs) for p, xs in parts.items()},
                      "mean_chars": round(sum(L) / len(L)), "median_chars": L[len(L) // 2]}
    report["ood_concepts"] = lens
    report["sources"] = {n: m["source"] for n, m in meta.items()}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
