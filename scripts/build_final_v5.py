"""FINAL concept split v5 (REPLICATION 2026-10-18f; user 10-18: "standardize to 4 per set", "keep
emotion fully held out", biology OK). Published data only; aggregates printed, never texts. Two
splits (train / heldout), fixed-step training, no dev. Built on build_final_v4.py (its helpers and
its build(), with the v5 concept tables patched in); v4 stays reproducible.

TRAIN: 4 per area (20).
- topic: sports, politics and elections, astronomy, war and the military (law and courts DROPPED,
  not held out: the topic split only guarantees conflict-freeness for the v1 split)
- language: German (NC), Japanese, Arabic, Hindi (Aya); Spanish DROPPED
- format: all-caps (NC), SQL, JSON, JavaScript
- tone: jokes, comforting, casual, formal (NC style "formal")
- domain: maths, chemistry, literature, finance
HELD-OUT:
- topic: the 10 v4 held-out topics; language: the 8 v4 held-out languages
- format: HTML (NC), Python (CodeAlpaca), Java (Nan-Do/code-search-net-java shard 0: code with the
  Javadoc stripped; request from func_name + the docstring's first paragraph)
- tone: confusion, supportive (NC), cautious (withmartian/TONEBANK response_cautious; 200 rows, NO
  licence stated -> eval only)
- domain: physics (camel-ai), biology (NC benign "biology-focused": EXEMPT from the BIO_MAX filter,
  held-out only, user 10-18; purity 0.75 since most rows are co-rated chemistry-based > 0.5),
  computer science (arXiv cs.* by PRIMARY category, no math / stat cross-list,
  librarian-bots/arxiv-metadata-snapshot shard 5; hard negatives = non-CS abstracts of the same
  shard, so the probe cannot learn "academic abstract")
- emotion (held-out FAMILY): build_emotion_eval.emotion_concepts (7 emotions; other emotions'
  stories + neutral stories as hard negatives; one emotion-free request)
- safety (held-out FAMILY): DolusChat, Apollo roleplaying, Apollo insider trading (as v4) +
  "deception: sycophancy" (AlignmentResearch/diverse-deception gsm8k_sycophancy, CC-BY-4.0: the
  model agrees with the user's wrong answer; honest reply to the same conversation = hard negative;
  request = the conversation before the reply, last --max-request characters).
  NOT added: diverse-deception alignment_faking_claude (labels = refusal vs harmful COMPLIANCE by a
  keyword heuristic: a harmfulness concept with harmful text, not deception; user: no harm data).

  python scripts/build_final_v5.py --out-data D.json --out-inputs I.json --out-split S.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_final_v4 as v4
from build_emotion_eval import emotion_concepts
from build_final_base import BENIGN, DOLLY, STYLE, split3, text_id
from build_final_ood import SOURCES, load_rows
from build_final_v3 import MATH, select_nc

TICA = v4.TICA
V5 = TICA / "v5_sources"
TRAIN_TOPICS = ("sports", "politics and elections", "astronomy", "war and the military")
NC = {**v4.NC, "formal": ("formal tone", "tone", "train", "style")}
LANG = {k: v for k, v in v4.LANG.items() if k != "Spanish"}
BIO_ADJ, BIO_NAME = "biology-focused", "biology"
SYNTAX = {
    **v4.SYNTAX,
    "Java code": ("format", "heldout", "Nan-Do/code-search-net-java shard 0 (Apache-2.0 card; "
                                       "per-repo licences)"),
    "cautious tone": ("tone", "heldout", "withmartian/TONEBANK response_cautious (no licence "
                                         "stated; eval only)"),
}
CS_SRC = "librarian-bots/arxiv-metadata-snapshot shard 5 (CC0 metadata), cs.* primary category"
SYCO_SRC = "AlignmentResearch/diverse-deception gsm8k_sycophancy (CC-BY-4.0)"


def java_rows() -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    t = pq.read_table(V5 / "java" / "train0.parquet",
                      columns=["code", "docstring", "func_name"]).to_pydict()
    out = []
    for c, d, f in zip(t["code"], t["docstring"], t["func_name"], strict=True):
        doc = re.sub(r"\s+", " ", (d or "").split("\n\n")[0]).strip()
        if len(doc) < 20:
            continue
        name = (f or "").split(".")[-1]
        out.append((f"Write a Java method{f' `{name}`' if name else ''}: {doc[:300]}", c.strip()))
    return out


def cautious_rows() -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    t = pq.read_table(V5 / "tonebank" / "train.parquet",
                      columns=["prompt", "response_cautious"]).to_pydict()
    return [(p.strip(), x.strip()) for p, x in zip(t["prompt"], t["response_cautious"],
                                                    strict=True) if p and x]


def arxiv_rows(lo: int, hi: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(cs.* primary-category abstracts, non-CS abstracts); math / stat cross-lists dropped."""

    import pyarrow.parquet as pq

    t = pq.read_table(V5 / "arxiv" / "train5.parquet",
                      columns=["title", "abstract", "categories"]).to_pydict()
    cs, other = [], []
    for ti, ab, cat in zip(t["title"], t["abstract"], t["categories"], strict=True):
        cats = (cat or "").split()
        ab, ti = re.sub(r"\s+", " ", ab or "").strip(), re.sub(r"\s+", " ", ti or "").strip()
        if not cats or not ti or not lo <= len(ab) <= hi:
            continue
        if any(c.startswith(("math.", "stat.")) for c in cats):
            continue
        row = (f'Write an abstract for a paper titled "{ti}".', ab)
        if cats[0].startswith("cs."):
            cs.append(row)
        elif not any(c.startswith("cs.") for c in cats):
            other.append(row)
    return cs, other


def syco_pairs(lo: int, hi: int, max_request: int) -> list[tuple[str, str, str]]:
    """(request, deceptive reply, honest reply) per gsm8k_sycophancy scenario."""

    import pyarrow.parquet as pq

    t = pq.read_table(V5 / "diverse_deception" / "train.parquet",
                      columns=["messages", "label", "source_dataset", "pair_id"]).to_pylist()
    by: dict[str, dict] = {}
    for r in t:
        if r["source_dataset"] != "gsm8k_sycophancy":
            continue
        m = r["messages"]
        if not m or m[-1]["role"] != "assistant":
            continue
        ctx = "\n\n".join(f"{x['role'].capitalize()}: {x['content'].strip()}" for x in m[:-1])
        by.setdefault(r["pair_id"], {"ctx": ctx})[int(r["label"])] = m[-1]["content"].strip()
    out = []
    for _, v in sorted(by.items()):
        if 0 in v and 1 in v and all(lo <= len(v[k]) <= hi for k in (0, 1)):
            out.append((v["ctx"][-max_request:], v[1], v[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", default=str(TICA / "topic_base_v1.json"))
    ap.add_argument("--base-inputs", default=str(TICA / "topic_base_v1_inputs.json"))
    ap.add_argument("--math-data", default=str(TICA / "math_base_v1.json"))
    ap.add_argument("--math-inputs", default=str(TICA / "math_base_v1_inputs.json"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--n-cs-hard", type=int, default=400)
    ap.add_argument("--min-texts", type=int, default=140)
    ap.add_argument("--max-request", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    ap.add_argument("--out-split", required=True)
    ap.add_argument("--out-report", default="")
    a = ap.parse_args()
    lo, hi = 150, 1500
    # v5 concept tables into the v4 builder (module globals read at call time)
    v4.TRAIN_TOPICS, v4.NC, v4.LANG, v4.SYNTAX = TRAIN_TOPICS, NC, LANG, SYNTAX
    # the NC style negatives (academic / educational / informative) are mostly formal-rated: with
    # "formal" in the purity set they drop 715 -> 26. Keep v4's pool; the 27B neg_labels keep
    # formal-central texts out of the formal tone's negatives at fit time.
    nc_neg = v4.nc_negatives_v3
    v4.nc_negatives_v3 = lambda files, nc=NC, **kw: nc_neg(
        files, nc={k: v for k, v in nc.items() if k != "formal"}, **kw)
    md = json.loads(Path(a.math_data).read_text())
    mi = json.loads(Path(a.math_inputs).read_text())
    mc = next(c for c in md["concepts"] if c["concept"] == MATH)
    math_rows = [(mi[t["id"]], t["text"]) for p in ("fit", "calib", "episode") for t in mc[p]]
    camel = {k: load_rows(*SOURCES[k][:2]) for k in ("chemistry", "physics")}
    dolly = [json.loads(x) for x in DOLLY.read_text().splitlines() if x.strip()]
    nc_files = {"benign": json.loads(BENIGN.read_text()), "style": json.loads(STYLE.read_text())}
    syntax = {**v4.syntax_rows(), "Java code": java_rows(), "cautious tone": cautious_rows()}
    data, inputs, sp, rep = v4.build(
        json.loads(Path(a.base_data).read_text()), json.loads(Path(a.base_inputs).read_text()),
        nc_files, math_rows, camel, syntax, dolly, n=a.n, min_n=a.min_texts, lo=lo, hi=hi,
        max_request=a.max_request, seed=a.seed)

    # held-out extras that need their own selection / hard negatives
    ent = lambda xs: [{"id": text_id(x), "text": x} for x in xs]  # noqa: E731
    taken = {t["text"] for c in data["concepts"] for p in ("fit", "calib", "episode")
             for t in c[p]}
    fams = dict(sp["families"])
    meta = data["source"]["v4_concepts"]
    extra = []

    def add(cid, name, fam, rows, src, hard=None):
        st = rep["filter"].setdefault(name, {"rows": len(rows)})
        rows = v4.take(rows, a.n, a.min_texts, taken, str(a.seed), name, st)
        r = random.Random(f"{a.seed}-split-{name}")
        fit, cal, ep = split3([x for _, x in rows], r)
        c = {"concept_id": cid, "concept": name, "genre": fam, "split": "heldout",
             "fit": ent(fit), "calib": ent(cal), "episode": ent(ep)}
        for q, x in rows:
            inputs[text_id(x)] = q
        if hard:
            for q, x in hard:
                inputs.setdefault(text_id(x), q)
            hf, hc, he = split3([x for _, x in hard], r)
            c["hard_neg"] = {"fit": hf, "calib": hc, "episode": he}
        extra.append(c)
        fams[name] = fam
        meta[name] = {"family": fam, "source": src}

    # biology: NC benign "biology-focused", the BIO_MAX filter off for this concept only
    bio_nc = {**NC, BIO_ADJ: (BIO_NAME, "domain", "heldout", "benign")}
    # purity 0.75 (as v4.PURITY's NC concepts): at the default 0.5, 372 / 511 rows co-rated
    # chemistry-based > 0.5 (biochemistry) would leave 105 texts
    sel, bio_st = select_nc(nc_files, nc=bio_nc, purity={**v4.PURITY, BIO_ADJ: 0.75},
                            bio_max=2.0)
    rep["nc_filter"][BIO_ADJ] = bio_st[BIO_ADJ]
    add("nc-biology", BIO_NAME, "domain", [(p, x) for p, x, _ in sel[BIO_ADJ]],
        "NC rated benign file (biology-focused; bio filter off, held-out only)")
    cs, other = arxiv_rows(lo, hi)
    hard = random.Random(f"{a.seed}-cs-hard").sample(other, min(a.n_cs_hard, len(other)))
    add("arxiv-cs", "computer science", "domain", cs, CS_SRC, hard=hard)
    rep["filter"]["computer science"]["non_cs_pool"] = len(other)

    # sycophancy: paired deceptive / honest replies (safety family, as v4's safety concepts)
    prs = syco_pairs(lo, hi * 2, a.max_request)
    rng = random.Random(f"{a.seed}-deception: sycophancy")
    prs = rng.sample(prs, min(a.n, len(prs)))
    for q, dec, hon in prs:
        inputs[text_id(dec)] = q
        inputs[text_id(hon)] = q
    fit, cal, ep = split3(prs, rng)
    extra.append({"concept_id": "safety-sycophancy", "concept": "deception: sycophancy",
                  "genre": "safety", "split": "heldout",
                  **{k: ent([p[1] for p in xs]) for k, xs in
                     (("fit", fit), ("calib", cal), ("episode", ep))},
                  "hard_neg": {k: [p[2] for p in xs] for k, xs in
                               (("fit", fit), ("calib", cal), ("episode", ep))}})
    fams["deception: sycophancy"] = "safety"
    meta["deception: sycophancy"] = {"family": "safety", "source": SYCO_SRC + " (deceptive reply; "
                                     "matched honest reply = hard negative)"}
    rep["filter"]["deception: sycophancy"] = {"pairs": len(prs)}

    # emotion family (held out)
    emo, emo_inputs, emo_st = emotion_concepts(a.seed)
    for c in emo:
        extra.append(c)
        fams[c["concept"]] = "emotion"
        meta[c["concept"]] = {"family": "emotion",
                              "source": "abotresol/emotion-story-corpora (MIT), DeepSeek parts; "
                                        "other emotions + neutral = hard negatives"}
    inputs.update(emo_inputs)
    rep["filter"]["emotion"] = emo_st

    # no text is both a concept text and a negative; no id clash across concepts
    data["concepts"] += extra
    seen: dict[str, str] = {}
    for c in data["concepts"]:
        for p in ("fit", "calib", "episode"):
            for t in c[p]:
                if seen.setdefault(t["id"], c["concept"]) != c["concept"]:
                    raise SystemExit(f"text shared by {seen[t['id']]} and {c['concept']}")
    for part, ts in data["negatives"].items():
        data["negatives"][part] = [t for t in ts if t["id"] not in seen]
    rep["negatives"].update({p: len(v) for p, v in data["negatives"].items()})
    missing = [i for i in seen if i not in inputs]
    missing += [text_id(x) for c in data["concepts"] for xs in (c.get("hard_neg") or {}).values()
                for x in xs if text_id(x) not in inputs]
    if missing:
        raise SystemExit(f"{len(missing)} texts without a request")

    split = {"train": [], "heldout": []}
    for c in data["concepts"]:
        split[c["split"]].append(c["concept"])
    data["source"]["v5_builder"] = "scripts/build_final_v5.py"
    out_split = {
        "_doc": "FINAL split v5 (scripts/build_final_v5.py): train / heldout only (fixed-step "
                "training, no dev); 4 train members per area (topic, language, format, tone, "
                "domain); emotion and safety = held-out families.",
        **split, "families": fams}
    rep["concepts"] = {}
    for c in data["concepts"]:
        L = sorted(len(t["text"]) for p in ("fit", "calib", "episode") for t in c[p])
        rep["concepts"][c["concept"]] = {
            "family": fams[c["concept"]], "split": c["split"],
            **{p: len(c[p]) for p in ("fit", "calib", "episode")},
            "hard_neg": sum(len(v) for v in (c.get("hard_neg") or {}).values()),
            "median_chars": L[len(L) // 2],
            "source": meta.get(c["concept"], {}).get("source", "")}
    Path(a.out_data).write_text(json.dumps(data))
    Path(a.out_inputs).write_text(json.dumps(inputs))
    Path(a.out_split).write_text(json.dumps(out_split, indent=1) + "\n")
    if a.out_report:
        Path(a.out_report).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"train": len(split["train"]), "heldout": len(split["heldout"]),
                      "negatives": rep["negatives"],
                      "concepts": {k: [v["family"], v["split"], v["fit"] + v["calib"]
                                       + v["episode"], v["hard_neg"], v["median_chars"]]
                                   for k, v in rep["concepts"].items()}}, indent=1))


if __name__ == "__main__":
    main()
