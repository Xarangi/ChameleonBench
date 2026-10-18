"""FINAL concept split v4 (REPLICATION 2026-10-17aa; user 10-17: "combine dev and held out", "at
most 3-5 of each", more languages / code-like concepts). Published data only; aggregates printed,
never texts. Two splits only (train / heldout): training runs a fixed number of steps, no dev.

Per family, 4-5 train members; everything else held out:
- topic: train sports, politics and elections, astronomy, law and courts, war and the military;
  held-out the 8 v3 held-out + 2 v3 dev topics. The other v3 train topics are DROPPED (not held out:
  the topic split only guarantees train-vs-held-out conflict-freeness for the v1 split, and e.g.
  spaceflight ~ astronomy).
- language: train German (NC), Spanish, Japanese, Standard Arabic, Hindi (Aya); held-out Finnish
  (NC), French, Portuguese, Turkish, Russian, Greek, Thai, Tamil (Aya). Latin-script languages pass
  a function-word check (target beats English and the other Latin languages); the others a script
  check (>= 60% of letters in the language's script; Japanese counts kana + CJK and needs kana).
- format (syntax): train all-caps (NC), SQL (gretelai/synthetic_text_to_sql: context + query),
  JSON (glaiveai/glaive-function-calling-v2 function schemas), JavaScript (CodeSearchNet JS);
  held-out HTML (NC), Python code (CodeAlpaca-20k, Python-like code only).
- tone: train jokes, comforting, casual (NC rated); held-out confusion, supportive (NC rated);
  angry dropped (56 rated rows).
- domain: train mathematical reasoning (math_base_v1), chemistry (camel-ai), literature (NC),
  finance (gbharti/finance-alpaca); held-out physics (camel-ai).
- safety (held-out FAMILY): DolusChat, Apollo roleplaying, Apollo insider trading; ANSWER = the
  response only (v3's deception answer was prompt + response); honest response = matched hard
  negative; insider-trading requests cut to their last --max-request characters.
Positives are trusted from the sources (+ the NC rating / purity rule and the language / code
checks above); the 27B judge labels are used to keep NEGATIVES clean (tica/neg_labels.py).
Negatives: v3's pool (topic_base_v1 negatives + Dolly-15k + NC style academic / educational /
informative), minus any text equal to a concept text.

  python scripts/build_final_v4.py --out-data D.json --out-inputs I.json --out-split S.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_final_base import BENIGN, DOLLY, RAW, STYLE, split3, text_id
from build_final_ood import SOURCES, code_like, load_rows
from build_final_v3 import (
    AYA,
    AYA_LIC,
    FW,
    MATH,
    nc_negatives_v3,
    select_nc,
    take,
)

TICA = RAW / "tica"
V4 = TICA / "v4_sources"
TRAIN_TOPICS = ("sports", "politics and elections", "astronomy", "law and courts",
                "war and the military")
NC = {  # NC adjective -> (concept, family, split, NC file key)
    "german": ("German language", "language", "train", "benign"),
    "finnish": ("Finnish language", "language", "heldout", "benign"),
    "all-caps": ("all-caps text", "format", "train", "benign"),
    "HTML": ("HTML markup", "format", "heldout", "benign"),
    "jokey": ("jokes and humor", "tone", "train", "benign"),
    "comforting": ("comforting tone", "tone", "train", "benign"),
    "confused": ("confusion", "tone", "heldout", "benign"),
    "literature-focused": ("literature", "domain", "train", "benign"),
    "casual": ("casual tone", "tone", "train", "style"),
    # "angry" DROPPED: only 56 NC style rows rated >= 0.75 (17 pure at 0.5) < --min-texts
    "supportive": ("supportive tone", "tone", "heldout", "style"),
}
PURITY = {"confused": 0.75, "german": 0.75, "finnish": 0.75, "all-caps": 0.75,
          "comforting": 0.75, "supportive": 0.75}
LANG = {  # Aya language -> (concept, split, check)
    "Spanish": ("Spanish language", "train", "fw"),
    "Japanese": ("Japanese language", "train", "ja"),
    "Standard Arabic": ("Arabic language", "train", "Arabic"),
    "Hindi": ("Hindi language", "train", "Devanagari"),
    "French": ("French language", "heldout", "fw"),
    "Portuguese": ("Portuguese language", "heldout", "fw"),
    "Turkish": ("Turkish language", "heldout", "fw"),
    "Russian": ("Russian language", "heldout", "Cyrillic"),
    "Greek": ("Greek language", "heldout", "Greek"),
    "Thai": ("Thai language", "heldout", "Thai"),
    "Tamil": ("Tamil language", "heldout", "Tamil"),
}
FW4 = {
    **FW,
    "Portuguese": {"o", "os", "as", "e", "do", "da", "dos", "das", "em", "um", "uma", "para",
                   "com", "não", "que", "se", "mais", "pelo", "pela", "são"},
    "Turkish": {"ve", "bir", "bu", "da", "de", "için", "ile", "çok", "olan", "olarak", "gibi",
                "daha", "ama", "veya", "en", "ise", "kadar"},
}


def fw_ok(x: str, lang: str) -> bool:
    w = re.findall(r"[a-zà-ÿçğışöü]+", x.lower())
    hits = {k: sum(t in v for t in w) for k, v in FW4.items()}
    other = max(v for k, v in hits.items() if k != lang)
    return hits[lang] >= 3 and hits[lang] > other


def script_ok(x: str, script: str) -> bool:
    letters = [ch for ch in x if ch.isalpha()]
    if not letters:
        return False
    names = [unicodedata.name(ch, "") for ch in letters]
    if script == "ja":
        kana = sum(n.startswith(("HIRAGANA", "KATAKANA")) for n in names)
        cjk = sum(n.startswith("CJK UNIFIED") for n in names)
        return kana > 0 and (kana + cjk) / len(letters) >= 0.6
    return sum(n.startswith(script.upper()) for n in names) / len(letters) >= 0.6


def aya_rows(lang: str, check: str, lo: int, hi: int, st: dict) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    t = pq.read_table(AYA, columns=["inputs", "targets", "language"]).to_pydict()
    rows = [(q.strip(), x.strip()) for q, x, g in zip(t["inputs"], t["targets"], t["language"],
                                                       strict=True) if g == lang]
    st["rows"] = len(rows)
    rows = [(q, x) for q, x in rows if q and lo <= len(x) <= hi]
    st["len_ok"] = len(rows)
    ok = (lambda x: fw_ok(x, lang)) if check == "fw" else (lambda x: script_ok(x, check))
    rows = [(q, x) for q, x in rows if ok(x)]
    st["lang_ok"] = len(rows)
    return rows


def python_like(x: str) -> bool:
    """CodeAlpaca output that reads as Python: code-like, Python keywords, no C-family braces."""

    kw = r"^\s*(def |import |from \w+ import |print\(|for \w+ in )"
    return (code_like(x) and bool(re.search(kw, x, re.M))
            and x.count("{") + x.count(";") <= 2)


def syntax_rows() -> dict[str, list[tuple[str, str]]]:
    import pyarrow.parquet as pq

    out = {}
    t = pq.read_table(V4 / "sql" / "train.parquet",
                      columns=["sql_prompt", "sql_context", "sql"]).to_pydict()
    out["SQL code"] = [(p.strip(), f"{c.strip()}\n\n{s.strip()}") for p, c, s in
                       zip(t["sql_prompt"], t["sql_context"], t["sql"], strict=True)]
    rows = []
    for r in json.loads((V4 / "json" / "glaive.json").read_text()):
        s = r.get("system") or ""
        i = s.find("{")
        if i < 0:
            continue
        body = s[i:].strip()
        try:  # a single function schema that parses as a JSON object
            obj = json.loads(body)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            continue
        rows.append((f"Write the JSON schema for the function `{obj['name']}`.", body))
    out["JSON data"] = rows
    t = pq.read_table(V4 / "javascript" / "train0.parquet",
                      columns=["code", "docstring", "func_name"]).to_pydict()
    out["JavaScript code"] = [
        (f"Write a JavaScript function{f' `{f}`' if f else ''}: {d.strip()}", c.strip())
        for c, d, f in zip(t["code"], t["docstring"], t["func_name"], strict=True)
        if d and d.strip()]
    out["Python code"] = [(q, x) for q, x in load_rows(*SOURCES["computer code"][:2])
                          if python_like(x)]
    fin = json.loads((V4 / "finance" / "finance.json").read_text())
    out["finance"] = [(r["instruction"].strip() + (f"\n\n{r['input'].strip()}"
                                                    if (r.get("input") or "").strip() else ""),
                       r["output"].strip()) for r in fin]
    return out


SYNTAX = {  # concept -> (family, split, source label)
    "SQL code": ("format", "train", "gretelai/synthetic_text_to_sql (Apache-2.0): context + query"),
    "JSON data": ("format", "train", "glaiveai/glaive-function-calling-v2 (Apache-2.0): schemas"),
    "JavaScript code": ("format", "train", "Nan-Do/code-search-net-javascript (Apache-2.0)"),
    "Python code": ("format", "heldout", "sahil2801/CodeAlpaca-20k (CC-BY-4.0), Python-like"),
    "finance": ("domain", "train", "gbharti/finance-alpaca (MIT)"),
}
SAFETY = {  # concept -> (source file, source name or None)
    "deception: DolusChat": (RAW / "dolus_deception" / "examples.jsonl", None),
    "deception: roleplaying": (RAW / "apollo_repe_deception" / "examples.rollouts_backup.jsonl",
                               "scale-safety-research/roleplaying"),
    "deception: insider trading": (RAW / "apollo_repe_deception" /
                                   "examples.rollouts_backup.jsonl",
                                   "scale-safety-research/insider_trading"),
}


def safety_pairs(path: Path, source: str | None) -> list[dict]:
    by = defaultdict(dict)
    for x in path.read_text().splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        if source and r["source"] != source:
            continue
        resp = r["text"][len(r["generation_prefix"]):].strip()
        if resp:
            by[r["source_index"]][int(r["label"])] = (r["generation_prefix"].strip(), resp)
    return [v for _, v in sorted(by.items()) if 0 in v and 1 in v]


def build(base: dict, base_inputs: dict, nc_files: dict, math_rows, camel: dict, syntax: dict,
          dolly: list[dict], n: int = 600, min_n: int = 140, lo: int = 150, hi: int = 1500,
          n_dolly: int = 2000, max_request: int = 2000, seed: int = 0):
    ent = lambda xs: [{"id": text_id(x), "text": x} for x in xs]  # noqa: E731
    inputs = dict(base_inputs)
    rep: dict = {"filter": {}}
    concepts, fams, meta = [], {}, {}
    for c in base["concepts"]:
        if c["split"] == "train" and c["concept"] not in TRAIN_TOPICS:
            continue  # dropped v3-train topics
        concepts.append({**c, "split": "train" if c["concept"] in TRAIN_TOPICS else "heldout"})
        fams[c["concept"]] = "topic"
        meta[c["concept"]] = {"family": "topic", "source": "DBpedia (topic_base_v1)"}
    taken = {t["text"] for c in concepts for p in ("fit", "calib", "episode") for t in c[p]}

    def add(cid, name, fam, sp, rows, src):
        for q, x in rows:
            inputs[text_id(x)] = q
        fit, cal, ep = split3([x for _, x in rows], random.Random(f"{seed}-split-{name}"))
        concepts.append({"concept_id": cid, "concept": name, "genre": fam, "split": sp,
                         "fit": ent(fit), "calib": ent(cal), "episode": ent(ep)})
        fams[name] = fam
        meta[name] = {"family": fam, "source": src}

    sel, rep["nc_filter"] = select_nc(nc_files, nc=NC, purity=PURITY)
    for adj, (name, fam, sp, key) in NC.items():
        st = rep["filter"].setdefault(name, {})
        rows = take([(p, x) for p, x, _ in sel[adj]], n, min_n, taken, str(seed), name, st)
        add(f"nc-{adj}", name, fam, sp, rows,
            f"NC rated {key} file ({(BENIGN if key == 'benign' else STYLE).name[:40]})")
    for lang, (name, sp, check) in LANG.items():
        st = rep["filter"].setdefault(name, {})
        rows = take(aya_rows(lang, check, lo, hi, st), n, min_n, taken, str(seed), name, st)
        add(f"aya-{lang.lower().replace(' ', '-')}", name, "language", sp, rows, AYA_LIC)
    st = rep["filter"].setdefault(MATH, {"rows": len(math_rows)})
    add("math-0", MATH, "domain", "train", take(math_rows, n, min_n, taken, str(seed), MATH, st),
        "math_base_v1 (base-model GSM8K-train solutions; GSM8K MIT)")
    for name, sp in (("chemistry", "train"), ("physics", "heldout")):
        st = rep["filter"].setdefault(name, {"rows": len(camel[name])})
        rows = [(q, x) for q, x in camel[name] if q and lo <= len(x) <= hi]
        st["len_ok"] = len(rows)
        add(f"dom-{name}", name, "domain", sp, take(rows, n, min_n, taken, str(seed), name, st),
            SOURCES[name][2])
    for name, (fam, sp, src) in SYNTAX.items():
        st = rep["filter"].setdefault(name, {"rows": len(syntax[name])})
        rows = [(q, x) for q, x in syntax[name] if q and lo <= len(x) <= hi]
        st["len_ok"] = len(rows)
        add(f"v4-{name.split()[0].lower()}", name, fam, sp,
            take(rows, n, min_n, taken, str(seed), name, st), src)
    for name, (path, source) in SAFETY.items():
        prs = safety_pairs(path, source)
        rng = random.Random(f"{seed}-{name}")
        prs = [v for v in prs if lo <= len(v[1][1]) <= hi * 2]
        prs = rng.sample(prs, min(n, len(prs)))
        for v in prs:
            for lab in (0, 1):
                req, resp = v[lab]
                inputs[text_id(resp)] = req[-max_request:]
        fit, cal, ep = split3(prs, rng)
        part = {"fit": fit, "calib": cal, "episode": ep}
        concepts.append({
            "concept_id": "safety-" + name.split(": ")[1].replace(" ", "-").lower(),
            "concept": name, "genre": "safety", "split": "heldout",
            **{k: ent([v[1][1] for v in xs]) for k, xs in part.items()},
            "hard_neg": {k: [v[0][1] for v in xs] for k, xs in part.items()}})
        fams[name] = "safety"
        meta[name] = {"family": "safety", "source": f"{source or 'AlignmentResearch/DolusChat'} "
                                                    "(response only; matched honest negatives)"}
        rep["filter"][name] = {"pairs": len(prs)}

    cids = {t["id"] for c in concepts for p in ("fit", "calib", "episode") for t in c[p]}
    neg, nst = [], {}
    for p, x, c in nc_negatives_v3(nc_files, nc=NC):
        neg.append((p, x))
        nst[c] = nst.get(c, 0) + 1
    dol = [d for d in dolly if lo <= len(d["response"]) <= hi]
    for d in random.Random(f"{seed}-dolly").sample(dol, min(n_dolly, len(dol))):
        neg.append((d["instruction"] + (f"\n\n{d['context']}" if d.get("context") else ""),
                    d["response"].strip()))
    nst["dolly15k"] = min(n_dolly, len(dol))
    d: dict[str, str] = {}
    for q, x in neg:
        d.setdefault(x, q)
    neg = [(q, x) for x, q in d.items() if text_id(x) not in cids]
    for q, x in neg:
        inputs.setdefault(text_id(x), q)
    nf, nc_, ne = split3([x for _, x in neg], random.Random(f"{seed}-neg"))
    negs = {}
    for part, xs in (("fit", nf), ("calib", nc_), ("episode", ne)):
        old = [t for t in base["negatives"][part] if t["id"] not in cids]
        seen = {t["id"] for t in old}
        negs[part] = old + [t for t in ent(xs) if t["id"] not in seen]
    rep["negatives"] = {**nst, **{p: len(v) for p, v in negs.items()}}

    missing = [t["id"] for c in concepts for p in ("fit", "calib", "episode") for t in c[p]
               if t["id"] not in inputs]
    missing += [t["id"] for ts in negs.values() for t in ts if t["id"] not in inputs]
    missing += [text_id(x) for c in concepts for xs in (c.get("hard_neg") or {}).values()
                for x in xs if text_id(x) not in inputs]
    if missing:
        raise SystemExit(f"{len(missing)} texts without a request")
    split = {"train": [], "heldout": []}
    for c in concepts:
        split[c["split"]].append(c["concept"])
    data = {"source": {**base.get("source", {}), "v4_concepts": meta, "v4_negatives": nst,
                       "v4_builder": "scripts/build_final_v4.py"},
            "seed": seed, "concepts": concepts, "negatives": negs}
    out_split = {
        "_doc": "FINAL split v4 (scripts/build_final_v4.py): train / heldout only (fixed-step "
                "training, no dev); 4-5 train members per family; safety = held-out family.",
        **split, "families": fams}
    rep["concepts"] = {}
    for c in concepts:
        L = sorted(len(t["text"]) for p in ("fit", "calib", "episode") for t in c[p])
        rep["concepts"][c["concept"]] = {
            "family": fams[c["concept"]], "split": c["split"],
            **{p: len(c[p]) for p in ("fit", "calib", "episode")},
            "median_chars": L[len(L) // 2], "source": meta[c["concept"]]["source"]}
    return data, inputs, out_split, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", default=str(TICA / "topic_base_v1.json"))
    ap.add_argument("--base-inputs", default=str(TICA / "topic_base_v1_inputs.json"))
    ap.add_argument("--math-data", default=str(TICA / "math_base_v1.json"))
    ap.add_argument("--math-inputs", default=str(TICA / "math_base_v1_inputs.json"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--min-texts", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-inputs", required=True)
    ap.add_argument("--out-split", required=True)
    ap.add_argument("--out-report", default="")
    a = ap.parse_args()
    md = json.loads(Path(a.math_data).read_text())
    mi = json.loads(Path(a.math_inputs).read_text())
    mc = next(c for c in md["concepts"] if c["concept"] == MATH)
    math_rows = [(mi[t["id"]], t["text"]) for p in ("fit", "calib", "episode") for t in mc[p]]
    camel = {k: load_rows(*SOURCES[k][:2]) for k in ("chemistry", "physics")}
    dolly = [json.loads(x) for x in DOLLY.read_text().splitlines() if x.strip()]
    nc_files = {"benign": json.loads(BENIGN.read_text()), "style": json.loads(STYLE.read_text())}
    data, inputs, sp, rep = build(
        json.loads(Path(a.base_data).read_text()), json.loads(Path(a.base_inputs).read_text()),
        nc_files, math_rows, camel, syntax_rows(), dolly, n=a.n, min_n=a.min_texts, seed=a.seed)
    Path(a.out_data).write_text(json.dumps(data))
    Path(a.out_inputs).write_text(json.dumps(inputs))
    Path(a.out_split).write_text(json.dumps(sp, indent=1) + "\n")
    if a.out_report:
        Path(a.out_report).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"filter": rep["filter"], "negatives": rep["negatives"],
                      "concepts": {k: {kk: v[kk] for kk in ("family", "split", "fit",
                                                            "median_chars")}
                                   for k, v in rep["concepts"].items()}}, indent=1))


if __name__ == "__main__":
    main()
