"""FINAL concept split v3 (scripts/build_final_v3.py): the builder on tiny synthetic inputs (CPU, no
model, no real data) and the shipped split json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DEV = {"topic": {"television and radio", "buildings and architecture"},
       "language": {"French language"}, "format": set(), "tone": {"casual tone"},
       "domain": {"computer code"}, "safety": set()}
HELDOUT_FAMILY = "safety"


def _rows(adj, n, rating, extra=None, file_key="benign"):
    words = "uno dos el los y del por con una para" if adj == "german" else "lorem ipsum"
    return [{"adjectives": [adj], "prompt": f"p-{adj}-{i}", "is_topical_response_refusal": False,
             "topical_response": f"{adj} text {i} {words}",
             "topical_response_normalized_ratings": {adj: rating, **(extra or {})}}
            for i in range(n)]


def _inputs(tmp_path, n=12):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from build_final_v3 import DROP_TOPICS, NC

    topics = {"train": ["t_a", "t_b", *DROP_TOPICS], "dev": ["t_dev"], "heldout": ["t_ho"]}
    concepts, inputs = [], {}
    for sp, names in topics.items():
        for name in names:
            part = {p: [{"id": f"{name}-{p}-{i}", "text": f"{name} {p} {i}"} for i in range(3)]
                    for p in ("fit", "calib", "episode")}
            concepts.append({"concept_id": name, "concept": name, "genre": "g", "split": sp,
                             **part})
            inputs.update({t["id"]: "write" for ts in part.values() for t in ts})
    negs = {p: [{"id": f"neg-{p}", "text": f"free {p}"}] for p in ("fit", "calib", "episode")}
    inputs.update({f"neg-{p}": "q" for p in negs})
    base = {"source": {}, "concepts": concepts, "negatives": negs}
    nc = {"benign": [], "style": []}
    for adj, (*_, key) in NC.items():
        nc[key] += _rows(adj, n, 1.0)
    nc["benign"] += _rows("biology-focused", n, 1.0)  # never a concept, never a negative
    nc["benign"] += _rows("jokey", 5, 1.0, {"biology-focused": 0.75})  # bio-rated: excluded
    nc["benign"] += _rows("german", 5, 1.0, {"comforting": 1.0})  # impure even at 0.75
    nc["style"] += _rows("academic", 4, 1.0, {"casual": 0.25})  # concept-free negatives
    nc["style"] += _rows("informative", 4, 1.0, {"casual": 1.0})  # casual-rated: not a negative
    nc["style"] += _rows("casual", 3, 1.0, {"illegal": 0.25})  # harm-rated: never used
    es = "el perro y los gatos del barrio por la casa con una para es se su como lo al " * 4
    fr = "le chat et les chiens des rues pour une maison dans qui est sur au avec il ce " * 4
    en = "the dog and the cats of the street is that it for with are was on this be by " * 4
    langs, tgts = [], []
    for lang, body in (("Spanish", es), ("French", fr), ("Spanish", en)):  # English: dropped
        for i in range(n):
            langs.append(lang)
            tgts.append(f"{i} {body}")
    aya = tmp_path / "aya.parquet"
    pq.write_table(pa.table({"inputs": [f"q{i}" for i in range(len(tgts))], "targets": tgts,
                             "language": langs}), aya)
    pad = "x" * 160
    math_rows = [(f"solve {i}", f"math {i} {pad}") for i in range(n)]
    code = "\n".join(f"a{j} = f({j});" for j in range(30))
    domain = {"chemistry": [(f"c{i}", f"chem {i} {pad}") for i in range(n)],
              "computer code": [(f"k{i}", f"# {i}\n{code}") for i in range(n)]
              + [(f"k-prose{i}", f"prose {i} {pad}") for i in range(n)],  # not code-like
              "physics": [(f"f{i}", f"phys {i} {pad}") for i in range(n)]}
    dolus = [{"source_index": i, "label": lab, "text": f"dolus {i} {lab}",
              "generation_prefix": f"g{i}"} for i in range(n) for lab in (0, 1)]
    dolly = [{"instruction": f"i{i}", "context": "", "response": f"dolly {i} {pad}"}
             for i in range(n)] + [{"instruction": "dup", "context": "",
                                    "response": f"chem 0 {pad}"}]  # equals a concept text
    return base, inputs, topics, nc, aya, math_rows, domain, dolus, dolly


def test_build_v3_synthetic(tmp_path):
    from build_final_base import text_id
    from build_final_v3 import DROP_TOPICS, build

    from next_chameleons.tica.data import Level1Data

    data, inputs, sp, rep = build(*_inputs(tmp_path), n=10, min_n=5, n_dolly=50)
    names = {c["concept"]: c for c in data["concepts"]}
    assert not set(DROP_TOPICS) & set(names)
    assert not set(DROP_TOPICS) & {x for k in ("train", "dev", "heldout") for x in sp[k]}
    assert set(sp["families"]) == set(names)
    for c in data["concepts"]:
        assert c["split"] in ("train", "dev", "heldout") and c["concept"] in sp[c["split"]]
        n = sum(len(c[p]) for p in ("fit", "calib", "episode"))
        assert n <= 10 or c["genre"] == "g"
        assert all(t["id"] == text_id(t["text"]) or c["genre"] == "g"
                   for p in ("fit", "calib", "episode") for t in c[p])
    assert names["German language"]["split"] == "train"
    assert names["French language"]["split"] == "dev"
    assert names["physics"]["split"] == "heldout"
    ids = lambda c: {t["id"] for p in ("fit", "calib", "episode") for t in c[p]}  # noqa: E731
    assert len(ids(names["German language"])) == 10  # 5 impure rows excluded, 12 pure sampled
    assert rep["nc_filter"]["german"]["impure_dropped"] == 5
    assert rep["nc_filter"]["jokey"]["bio_dropped"] == 5
    assert rep["nc_filter"]["casual"]["ok"] == 12  # harm-rated rows dropped
    assert rep["filter"]["Spanish language"]["lang_ok"] == 12  # English-language rows dropped
    assert rep["filter"]["computer code"]["code_like"] == 12
    allc = set().union(*(ids(c) for c in data["concepts"]))
    negs = [t for ts in data["negatives"].values() for t in ts]
    assert not {t["id"] for t in negs} & allc
    assert rep["negatives"]["equal_to_concept_dropped"] == 1
    assert rep["negatives"].get("academic") == 4 and "informative" not in rep["negatives"]
    assert not any("biology" in t["text"] for t in negs)
    every = allc | {t["id"] for t in negs}
    every |= {text_id(x) for xs in names["deception"]["hard_neg"].values() for x in xs}
    assert every <= set(inputs)
    p = tmp_path / "d.json"
    p.write_text(json.dumps(data))
    d = Level1Data(p)
    assert {c.concept for c in d.concepts("fit", ("dev",))} == set(sp["dev"])
    assert d.hard_negatives("deception", "fit")


def _split():
    from next_chameleons.tica.multi import load_split

    return load_split(str(ROOT / "configs" / "tica" / "final_split_v3.json"))


def test_final_split_v3_families():
    sp = _split()
    tr, dv, ho = (set(sp[k]) for k in ("train", "dev", "heldout"))
    assert len(tr) == len(sp["train"]) and len(dv) == len(sp["dev"])
    assert not (tr & dv or tr & ho or dv & ho)
    fam = sp["families"]
    assert set(fam) == tr | dv | ho
    assert set(fam.values()) == set(DEV)
    for f, dev in DEV.items():
        members = {c for c, g in fam.items() if g == f}
        assert members & dv == dev
        if f == HELDOUT_FAMILY:
            assert members and members <= ho
        else:
            assert members & tr and members & ho
    assert {"towns and villages", "video games"}.isdisjoint(fam)
    assert len([c for c in tr if fam[c] == "topic"]) == 8


def test_final_split_v3_builtin_mirror():
    a = ROOT / "configs" / "tica" / "final_split_v3.json"
    b = ROOT / "src" / "next_chameleons" / "builtin_configs" / "tica" / "final_split_v3.json"
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("name", ["Spanish language", "French language",
                                  "casual tone", "literature", "chemistry"])
def test_label_audit_describes_v3_concepts(name):
    from label_concept_contamination import DESC

    assert name in DESC
    assert set(_split()["families"]) - {"deception"} - set(DESC) <= {
        c for c, f in _split()["families"].items() if f == "topic"}
