"""FINAL concept split v3 (REPLICATION 2026-10-16; families follow the probing literature's
surface / semantic / pragmatic levels). Published data only; aggregates printed, never texts.

Every family has train members, a dev member (validation / early stopping) and held-out members;
one whole family (safety) is held out:
- topic (semantic): topic_base_v1 DBpedia topics; train = 8 of the 13 v2 train topics (DROP_TOPICS
  removed from the data entirely), dev / held-out topics unchanged (400 / 100 / 100 as built);
- language (surface): train German (NC) + Spanish (Aya); dev French (Aya); held-out Finnish (NC);
- format (surface): train all-caps (NC); no dev (gibberish dropped: co-rated jokey); held-out HTML;
- tone (pragmatic): train jokey, comforting (NC benign); dev casual (NC style); formal dropped
  (NC style file); held-out confused (NC);
- domain (semantic, technical): train mathematical reasoning (math_base_v1 = base-model GSM8K-TRAIN
  solutions) + chemistry (camel-ai) + literature (NC "literature-focused"); dev computer code
  (CodeAlpaca-20k); held-out physics (camel-ai);
- safety (pragmatic, held-out FAMILY): deception (DolusChat, matched truthful hard negatives).

NC concepts (build_final_base's rule, over the FULL v3 selected NC set): rating >= 0.75, not a
refusal, no harm rating, every other selected concept rated on the row (plus the NC adjectives of
the v3 domain concepts, "mathematical" / "chemistry-based") <= PURITY[c] (0.5; 0.75 for concepts
that keep < 140 texts at 0.5, see PURITY), and "biology-focused" <= BIO_MAX (no biology content;
biology-focused rows are never used, not even as negatives). The two NC files rate DIFFERENT
adjectives (benign: german ... literature-focused; style: formal / casual / academic ...), so
purity and the overlap matrix are within-file only; cross-file leakage (casual ~ jokey) needs
scripts/label_concept_contamination.py.
Spanish / French: CohereLabs/aya_dataset (Apache-2.0; human-written answers), language field +
a function-word check (target language beats English and the other Romance language).
Per concept: <= --n texts (fixed-seed sample), >= --min-texts, split 60 / 20 / 20.
NEGATIVES (concept-free pool): topic_base_v1's negatives + Dolly-15k responses + NC style
academic / educational / informative (<= 0.5 on formal / casual; most are formal-rated, so few
survive). Any negative equal to a concept text is dropped.

  uv run python scripts/build_final_v3.py --out-data D.json --out-inputs I.json --out-split S.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_final_base import BENIGN, DOLLY, HARM, RAW, STYLE, split3, text_id
from build_final_ood import SOURCES, code_like, load_rows

TICA = RAW / "tica"
AYA = TICA / "aya_dataset" / "data" / "train-00000-of-00001.parquet"
AYA_LIC = "CohereLabs/aya_dataset @f9ea0458 (Apache-2.0)"
DROP_TOPICS = ("towns and villages", "motor racing", "comics, manga and animation", "video games",
               "transportation")
# NC adjective -> (concept name, family, split, source file key)
NC = {
    "german": ("German language", "language", "train", "benign"),
    "finnish": ("Finnish language", "language", "heldout", "benign"),
    "all-caps": ("all-caps text", "format", "train", "benign"),
    # gibberish (planned format DEV) DROPPED 10-16m: 99% of its texts are co-rated jokey >= 0.75,
    # so as the validation member it would measure jokey leakage; format has no dev member.
    "HTML": ("HTML markup", "format", "heldout", "benign"),
    "jokey": ("jokes and humor", "tone", "train", "benign"),
    "comforting": ("comforting tone", "tone", "train", "benign"),
    # "formal tone" DROPPED 10-16n: the v3 label audit judged 45% of the concept-free pool and 58%
    # of the other concepts' texts formal (encyclopedic register) -> not a separable concept here.
    "casual": ("casual tone", "tone", "dev", "style"),
    "confused": ("confusion", "tone", "heldout", "benign"),
    "literature-focused": ("literature", "domain", "train", "benign"),
}
# NC adjectives of v3 concepts drawn from other sources: count in purity like selected concepts
NC_ALIAS = ("mathematical", "chemistry-based")
BIO, BIO_MAX = "biology-focused", 0.5
# 0.5 leaves german 88 / finnish 119 / all-caps 126 / comforting 120 / gibberish 2 (< 140) over the
# full v3 set (blockers: comforting, literature, jokey; gibberish is co-rated jokey on 496 / 501);
# confused keeps v2's 0.75. Leakage is reported per concept pair (overlap matrix).
PURITY = {"confused": 0.75, "german": 0.75, "finnish": 0.75, "all-caps": 0.75,
          "comforting": 0.75}
NEG_STYLE = ("academic", "educational", "informative")
NEG_PURITY = 0.5  # vs formal / casual (0.25 as in v2 leaves 0 texts: all are formal-rated)
LANG = {  # Aya language -> (concept, split); function words for the language check
    "Spanish": ("Spanish language", "train"),
    "French": ("French language", "dev"),
}
FW = {
    "Spanish": {"el", "los", "las", "y", "del", "por", "con", "una", "para", "es", "se", "su",
                "como", "lo", "al", "muy", "pero"},
    "French": {"le", "et", "les", "des", "une", "du", "est", "pour", "dans", "qui", "pas", "sur",
               "au", "avec", "il", "ce", "sont"},
    "English": {"the", "and", "of", "to", "is", "that", "it", "for", "with", "are", "was", "on",
                "this", "be", "by"},
}
DOMAIN = {  # concept -> (split, SOURCES key of build_final_ood)
    "chemistry": "train", "computer code": "dev", "physics": "heldout"}
MATH = "mathematical reasoning"


def _adjs(r) -> list[str]:
    return r["adjectives"] if isinstance(r["adjectives"], list) else [r["adjectives"]]


def _ok(r) -> bool:
    rat = r.get("topical_response_normalized_ratings") or {}
    return (not r.get("is_topical_response_refusal")
            and bool((r.get("topical_response") or "").strip())
            and not any(rat.get(h, 0.0) > 0 for h in HARM))


def select_nc(files: dict[str, list[dict]], nc: dict = NC, purity: dict = PURITY,
              default: float = 0.5, bio_max: float = BIO_MAX):
    """{adj: [(prompt, text, ratings)]} (file order, texts deduped per concept) and per-adjective
    filter stats. `files` maps the NC source keys ("benign" / "style") to their rows."""

    sel = set(nc) | set(NC_ALIAS)
    by, st = defaultdict(dict), {c: defaultdict(int) for c in nc}
    for key, rows in files.items():
        for r in rows:
            rat = r.get("topical_response_normalized_ratings") or {}
            for c in _adjs(r):
                if c not in nc or nc[c][3] != key:
                    continue
                s = st[c]
                s["rows"] += 1
                if not _ok(r):
                    continue
                s["ok"] += 1
                if rat.get(c, 0.0) < 0.75:
                    continue
                s["rated"] += 1
                if rat.get(BIO, 0.0) > bio_max:
                    s["bio_dropped"] += 1
                    continue
                thr = purity.get(c, default)
                if any(rat.get(o, 0.0) > thr for o in sel if o != c and o in rat):
                    s["impure_dropped"] += 1
                    continue
                by[c].setdefault(r["topical_response"].strip(), (r["prompt"], rat))
    out = {c: [(p, x, rat) for x, (p, rat) in by[c].items()] for c in nc}
    for c in nc:
        st[c]["pure_unique"] = len(out[c])
    return out, {c: dict(s) for c, s in st.items()}


def overlap_matrix(sel_rows: dict, nc: dict = NC, thr: float = 0.5) -> dict:
    """Per NC concept: fraction of its SELECTED texts rated >= thr on each other rated adjective of
    the v3 set (+ biology); only adjectives rated in the same file appear."""

    keys = set(nc) | set(NC_ALIAS) | {BIO}
    out = {}
    for c, rows in sel_rows.items():
        if not rows:
            continue
        rated = sorted({o for _, _, rat in rows for o in rat if o in keys and o != c})
        out[nc[c][0]] = {o: round(sum(rat.get(o, 0.0) >= thr for _, _, rat in rows) / len(rows), 3)
                         for o in rated}
    return out


def nc_negatives_v3(files: dict[str, list[dict]], adjs=NEG_STYLE, key: str = "style",
                    nc: dict = NC, thr: float = NEG_PURITY, cap: int = 250):
    """(prompt, text, adj) concept-free NC negatives: rated >= 0.75 on one of `adjs`, every selected
    v3 adjective rated on the row <= thr, no biology > 0.25, cap per adjective (file order)."""

    sel = set(nc) | set(NC_ALIAS)
    n: dict[str, int] = {}
    for r in files.get(key, []):
        rat = r.get("topical_response_normalized_ratings") or {}
        if (not _ok(r) or rat.get(BIO, 0.0) > 0.25
                or any(rat.get(o, 0.0) > thr for o in sel if o in rat)):
            continue
        for c in _adjs(r):
            if c in adjs and rat.get(c, 0.0) >= 0.75 and n.get(c, 0) < cap:
                n[c] = n.get(c, 0) + 1
                yield r["prompt"], r["topical_response"].strip(), c
                break


def style_overlap(files: dict[str, list[dict]]) -> dict:
    """formal ~ academic / educational / informative (same file): fraction of each adjective's
    rated (>= 0.75, usable) rows that are rated >= 0.5 on the other."""

    pairs = [("formal", a) for a in NEG_STYLE] + [(a, "formal") for a in NEG_STYLE]
    pairs += [("casual", "formal"), ("formal", "casual")]
    out = {}
    for a, b in pairs:
        rows = [r for r in files.get("style", []) if a in _adjs(r) and _ok(r)
                and (r.get("topical_response_normalized_ratings") or {}).get(a, 0.0) >= 0.75]
        if rows:
            hit = sum((r["topical_response_normalized_ratings"]).get(b, 0.0) >= 0.5 for r in rows)
            out[f"{a}->{b}"] = round(hit / len(rows), 3)
    return out


def lang_ok(x: str, lang: str) -> bool:
    w = re.findall(r"[a-zà-ÿ]+", x.lower())
    hits = {k: sum(t in v for t in w) for k, v in FW.items()}
    other = max(v for k, v in hits.items() if k != lang)
    return hits[lang] >= 3 and hits[lang] > other


def aya_rows(path: Path, lang: str, lo: int, hi: int, st: dict) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["inputs", "targets", "language"]).to_pydict()
    rows = [(q.strip(), x.strip()) for q, x, g in zip(t["inputs"], t["targets"], t["language"],
                                                       strict=True) if g == lang]
    st["rows"] = len(rows)
    rows = [(q, x) for q, x in rows if q and lo <= len(x) <= hi]
    st["len_ok"] = len(rows)
    rows = [(q, x) for q, x in rows if lang_ok(x, lang)]
    st["lang_ok"] = len(rows)
    return rows


def take(rows: list[tuple[str, str]], n: int, min_n: int, taken: set, seed: str, name: str,
         st: dict) -> list[tuple[str, str]]:
    """Dedupe texts (first request wins), drop texts already used, fixed-seed sample of n."""

    d: dict[str, str] = {}
    for q, x in rows:
        d.setdefault(x, q)
    st["unique"] = len(d)
    rows = [(q, x) for x, q in d.items() if x not in taken]
    st["clash_dropped"] = len(d) - len(rows)
    if len(rows) < min_n:
        raise SystemExit(f"{name}: only {len(rows)} texts (< {min_n})")
    rng = random.Random(f"{seed}-{name}")
    rows = rng.sample(rows, min(n, len(rows)))
    taken.update(x for _, x in rows)
    return rows


def build(base: dict, base_inputs: dict, topic_split: dict, nc_files: dict, aya: Path,
          math_rows: list[tuple[str, str]], domain_rows: dict, dolus: list[dict],
          dolly: list[dict], n: int = 600, min_n: int = 140, lo: int = 150, hi: int = 1500,
          n_dolly: int = 2000, seed: int = 0):
    """Pure builder (paths only for Aya): -> (data, inputs, split, report)."""

    ent = lambda xs: [{"id": text_id(x), "text": x} for x in xs]  # noqa: E731
    inputs = dict(base_inputs)
    rep: dict = {"filter": {}}
    concepts, fams, meta = [], {}, {}
    split = {k: [] for k in ("train", "dev", "heldout")}
    for c in base["concepts"]:  # topics: unchanged parts and splits, minus the dropped ones
        if c["concept"] in DROP_TOPICS:
            continue
        if c["split"] != next(k for k in split if c["concept"] in topic_split[k]):
            raise SystemExit(f"topic {c['concept']}: split differs from the topic split")
        concepts.append(c)
        fams[c["concept"]] = "topic"
        meta[c["concept"]] = {"family": "topic", "source": "DBpedia (topic_base_v1)"}
    taken = {t["text"] for c in concepts for p in ("fit", "calib", "episode") for t in c[p]}

    def add(cid, name, fam, sp, rows, src, extra=None):
        for q, x in rows:
            inputs[text_id(x)] = q
        fit, cal, ep = split3([x for _, x in rows], random.Random(f"{seed}-split-{name}"))
        concepts.append({"concept_id": cid, "concept": name, "genre": fam, "split": sp,
                         "fit": ent(fit), "calib": ent(cal), "episode": ent(ep), **(extra or {})})
        fams[name] = fam
        meta[name] = {"family": fam, "source": src}

    # NC rated concepts
    sel, rep["nc_filter"] = select_nc(nc_files)
    rep["nc_overlap_ge_0.5"] = overlap_matrix(sel)
    rep["nc_overlap_ge_0.75"] = overlap_matrix(sel, thr=0.75)
    rep["style_overlap_ge_0.5"] = style_overlap(nc_files)
    nc_new = {}
    for adj, (name, *_) in NC.items():
        st = rep["filter"].setdefault(name, {})
        nc_new[adj] = take([(p, x) for p, x, _ in sel[adj]], n, min_n, taken, str(seed), name, st)
    # Aya languages
    aya_new = {}
    for lang, (name, _) in LANG.items():
        st = rep["filter"].setdefault(name, {})
        aya_new[lang] = take(aya_rows(aya, lang, lo, hi, st), n, min_n, taken, str(seed), name,
                             st)
    # domains
    st = rep["filter"].setdefault(MATH, {"rows": len(math_rows)})
    math_new = take(math_rows, n, min_n, taken, str(seed), MATH, st)
    dom_new = {}
    for name, rows in domain_rows.items():
        st = rep["filter"].setdefault(name, {"rows": len(rows)})
        rows = [(q, x) for q, x in rows if q and lo <= len(x) <= hi]
        st["len_ok"] = len(rows)
        if name == "computer code":
            rows = [(q, x) for q, x in rows if code_like(x)]
            st["code_like"] = len(rows)
        dom_new[name] = take(rows, n, min_n, taken, str(seed), name, st)

    # concepts in family order (language, format, tone, domain, safety)
    order = [("language", "German language"), ("language", "Spanish language"),
             ("language", "French language"), ("language", "Finnish language"),
             ("format", "all-caps text"), ("format", "HTML markup"),
             ("tone", "jokes and humor"), ("tone", "comforting tone"),
             ("tone", "casual tone"), ("tone", "confusion"), ("domain", MATH),
             ("domain", "chemistry"), ("domain", "literature"), ("domain", "computer code"),
             ("domain", "physics")]
    by_name = {v[0]: (adj, v) for adj, v in NC.items()}
    for fam, name in order:
        if name in by_name:
            adj, (_, _, sp, key) = by_name[name]
            add(f"nc-{adj}", name, fam, sp, nc_new[adj],
                f"NC rated {key} file ({(BENIGN if key == 'benign' else STYLE).name[:40]})")
        elif name in ("Spanish language", "French language"):
            lang = name.split()[0]
            add(f"aya-{lang.lower()}", name, fam, LANG[lang][1], aya_new[lang], AYA_LIC)
        elif name == MATH:
            add("math-0", name, fam, "train", math_new,
                "math_base_v1 (base-model GSM8K-train solutions; GSM8K MIT)")
        else:
            add(f"dom-{name.replace(' ', '-')}", name, fam, DOMAIN[name], dom_new[name],
                SOURCES[name][2])

    # deception (held-out family)
    pairs = defaultdict(dict)
    for r in dolus:
        pairs[r["source_index"]][int(r["label"])] = r
    ok = [v for _, v in sorted(pairs.items()) if 0 in v and 1 in v and v[1]["text"].strip()]
    rng = random.Random(f"{seed}-deception")
    ok = rng.sample(ok, min(n, len(ok)))
    for v in ok:
        for lab in (0, 1):
            inputs[text_id(v[lab]["text"])] = v[lab]["generation_prefix"]
    fit, cal, ep = split3(ok, rng)
    concepts.append({
        "concept_id": "dolus-deception", "concept": "deception", "genre": "safety",
        "split": "heldout", "fit": ent([v[1]["text"] for v in fit]),
        "calib": ent([v[1]["text"] for v in cal]), "episode": ent([v[1]["text"] for v in ep]),
        "hard_neg": {"fit": [v[0]["text"] for v in fit], "calib": [v[0]["text"] for v in cal],
                     "episode": [v[0]["text"] for v in ep]}})
    fams["deception"] = "safety"
    meta["deception"] = {"family": "safety", "source": "DolusChat (matched truthful negatives)"}
    for c in concepts:
        split[c["split"]].append(c["concept"])

    # negatives
    cids = {t["id"] for c in concepts for p in ("fit", "calib", "episode") for t in c[p]}
    neg, nst = [], {}
    for p, x, c in nc_negatives_v3(nc_files):
        neg.append((p, x))
        nst[c] = nst.get(c, 0) + 1
    dol = [d for d in dolly if lo <= len(d["response"]) <= hi]
    for d in random.Random(f"{seed}-dolly").sample(dol, min(n_dolly, len(dol))):
        q = d["instruction"] + (f"\n\n{d['context']}" if d.get("context") else "")
        neg.append((q, d["response"].strip()))
    nst["dolly15k"] = min(n_dolly, len(dol))
    d: dict[str, str] = {}
    for q, x in neg:
        d.setdefault(x, q)
    neg = [(q, x) for x, q in d.items() if text_id(x) not in cids]
    nst["equal_to_concept_dropped"] = len(d) - len(neg)
    for q, x in neg:
        inputs.setdefault(text_id(x), q)
    nf, nc_, ne = split3([x for _, x in neg], random.Random(f"{seed}-neg"))
    negs = {}
    for part, xs in (("fit", nf), ("calib", nc_), ("episode", ne)):
        old = [t for t in base["negatives"][part] if t["id"] not in cids]
        nst[f"base_{part}_equal_to_concept_dropped"] = len(base["negatives"][part]) - len(old)
        seen = {t["id"] for t in old}
        negs[part] = old + [t for t in ent(xs) if t["id"] not in seen]
    rep["negatives"] = {**nst, **{p: len(v) for p, v in negs.items()}}

    data = {"source": {**base.get("source", {}), "v3_concepts": meta, "v3_negatives": nst,
                       "v3_purity": {"default": 0.5, **PURITY, "biology_max": BIO_MAX},
                       "v3_overlap_ge_0.5": rep["nc_overlap_ge_0.5"],
                       "v3_overlap_ge_0.75": rep["nc_overlap_ge_0.75"],
                       "v3_style_overlap_ge_0.5": rep["style_overlap_ge_0.5"],
                       "v3_dropped_topics": list(DROP_TOPICS),
                       "v3_builder": "scripts/build_final_v3.py"},
            "seed": seed, "concepts": concepts, "negatives": negs}
    missing = [t["id"] for c in concepts for p in ("fit", "calib", "episode") for t in c[p]
               if t["id"] not in inputs]
    missing += [t["id"] for ts in negs.values() for t in ts if t["id"] not in inputs]
    missing += [text_id(x) for c in concepts for xs in (c.get("hard_neg") or {}).values()
                for x in xs if text_id(x) not in inputs]
    if missing:
        raise SystemExit(f"{len(missing)} texts without a request")
    out_split = {
        "_doc": "FINAL split v3 (scripts/build_final_v3.py): families at the surface / semantic / "
                "pragmatic levels, each with train members, a dev member (validation / early "
                "stopping) and held-out members; safety (deception) = held-out FAMILY. Topics: "
                "topic_split_v1 minus 5 train topics. NC concepts purity-filtered over the full "
                "v3 set; Spanish / French = Aya; domains = math_base_v1, camel-ai, CodeAlpaca.",
        **split, "families": fams}
    lens = {}
    for c in concepts:
        L = sorted(len(t["text"]) for p in ("fit", "calib", "episode") for t in c[p])
        lens[c["concept"]] = {"family": fams[c["concept"]], "split": c["split"],
                              **{p: len(c[p]) for p in ("fit", "calib", "episode")},
                              "median_chars": L[len(L) // 2],
                              "source": meta[c["concept"]]["source"]}
    rep["concepts"] = lens
    return data, inputs, out_split, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", default=str(TICA / "topic_base_v1.json"))
    ap.add_argument("--base-inputs", default=str(TICA / "topic_base_v1_inputs.json"))
    ap.add_argument("--topic-split", default="configs/tica/topic_split_v1.json")
    ap.add_argument("--math-data", default=str(TICA / "math_base_v1.json"))
    ap.add_argument("--math-inputs", default=str(TICA / "math_base_v1_inputs.json"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--min-texts", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    ap.add_argument("--out-split", required=True)
    ap.add_argument("--out-report", default="", help="optional JSON of the printed aggregates")
    a = ap.parse_args()
    md = json.loads(Path(a.math_data).read_text())
    mi = json.loads(Path(a.math_inputs).read_text())
    mc = next(c for c in md["concepts"] if c["concept"] == MATH)
    math_rows = [(mi[t["id"]], t["text"]) for p in ("fit", "calib", "episode") for t in mc[p]]
    domain_rows = {name: load_rows(*SOURCES[name][:2]) for name in DOMAIN}
    dolus = [json.loads(x) for x in (RAW / "dolus_deception" / "examples.jsonl").read_text()
             .splitlines() if x.strip()]
    dolly = [json.loads(x) for x in DOLLY.read_text().splitlines() if x.strip()]
    nc_files = {"benign": json.loads(BENIGN.read_text()), "style": json.loads(STYLE.read_text())}
    data, inputs, sp, rep = build(
        json.loads(Path(a.base_data).read_text()), json.loads(Path(a.base_inputs).read_text()),
        json.loads(Path(a.topic_split).read_text()), nc_files, AYA, math_rows, domain_rows,
        dolus, dolly, n=a.n, min_n=a.min_texts, seed=a.seed)
    Path(a.out_data).write_text(json.dumps(data))
    Path(a.out_inputs).write_text(json.dumps(inputs))
    Path(a.out_split).write_text(json.dumps(sp, indent=1) + "\n")
    if a.out_report:
        Path(a.out_report).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
