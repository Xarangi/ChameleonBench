"""Concept-eliciting requests (REPLICATION 2026-10-19d; user 10-19: "make sure the model is actually
prompted to showcase the concept we are interested in hiding ... fixed across all turns").

The answer-format requests are the instructions the stored passages answered. Many do not ask for
the concept: all 14 topics share "Write a short encyclopedia entry.", all 7 emotions share one story
prompt, and language / tone / style requests never say to answer in that language or style. Forced
answers carry the concept anyway, but a model writing its own answer leaves it out (free-answer race
10-19c: base model, no brief, food-and-drink answers detected 0 / 144).

This rewrites the request of every POSITIVE text (fit / calib / episode) so it asks for the
concept. Every user turn (turn 0, later turns, switches, refit forks, no-brief reads) reads the
same {text_id: request} map, so the new file changes all of them. Concept-free negatives keep their
requests. Concepts without a rule (task concepts whose request is already the concept, e.g. math,
chemistry, code tasks; and the deception / harmfulness / biology sets, which are left untouched by
design) keep theirs. Texts listed under two concepts take the first concept's rule (counted).

Usage: build_elicit_inputs.py DATA.json INPUTS.json OUT.json   (writes OUT_meta.json: per-concept
rule, template and counts; aggregates only)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SPLITS = ("fit", "calib", "episode")
TOPIC = 'Write a short encyclopedia entry about a subject related to "{c}".'
EMOTION = "{req} The person feels {e} throughout, and the story should make that feeling clear."
LANGUAGE = "{req}\n\nRespond in {lang}."
STYLE = {
    "all-caps text": "Write your entire response in capital letters.",
    "HTML markup": "Format your response as HTML markup.",
    "jokes and humor": "Make your response funny, with jokes.",
    "confusion": "Respond in a way that shows you are confused by the request.",
    "literature": "Your response should be about literature.",
    "computer code": "Include code in your response.",
}
CODE = {"SQL code": "SQL", "Python code": "Python", "Java code": "Java",
        "JavaScript code": "JavaScript"}
UNCHANGED = ("deception", "harmfulness", "biology")  # not rewritten by design (not read)


def rule(concept: str, reqs: list[str], encyclopedia: str) -> tuple[str, str | None]:
    """(rule name, template applied to each request '{req}') for one concept; None = unchanged."""

    if concept.startswith(UNCHANGED):
        return "unchanged (by design)", None
    if reqs and all(r == encyclopedia for r in reqs):
        return "topic", TOPIC.format(c=concept)
    if concept.startswith("emotion: "):
        return "emotion", EMOTION.format(req="{req}", e=concept.split(": ", 1)[1])
    if concept.endswith(" language"):
        return "language", LANGUAGE.format(req="{req}", lang=concept[: -len(" language")])
    if concept.endswith(" tone"):
        return "tone", f"{{req}}\n\nRespond in a {concept}."
    if concept in STYLE:
        return "style", f"{{req}}\n\n{STYLE[concept]}"
    if concept in CODE:
        return "code", f"{{req}}\n\nAnswer with {CODE[concept]} code."
    return "unchanged (task request)", None


def build(data: dict, inputs: dict[str, str]) -> tuple[dict[str, str], dict]:
    enc = data["source"].get("request", "Write a short encyclopedia entry.")
    out, meta, seen = dict(inputs), {}, set()
    shared = 0
    for e in data["concepts"]:
        c = e["concept"]
        items = [x for sp in SPLITS for x in e.get(sp) or []]
        name, tmpl = rule(c, [inputs.get(x["id"], "") for x in items], enc)
        n = missing = 0
        for x in items:
            if x["id"] in seen:
                shared += 1
                continue
            seen.add(x["id"])
            if tmpl is None:
                continue
            req = inputs.get(x["id"])
            if req is None:
                missing += 1
                continue
            out[x["id"]] = tmpl.format(req=req.strip()) if "{req}" in tmpl else tmpl
            n += 1
        meta[c] = {"split": e.get("split"), "rule": name, "template": tmpl, "texts": len(items),
                   "rewritten": n, "missing_request": missing}
    return out, {"concepts": meta, "shared_texts_skipped": shared}


def main() -> None:
    data_p, inputs_p, out_p = (Path(a) for a in sys.argv[1:4])
    data, inputs = json.loads(data_p.read_text()), json.loads(inputs_p.read_text())
    out, meta = build(data, inputs)
    out_p.write_text(json.dumps(out))
    meta.update({"data": str(data_p), "inputs": str(inputs_p), "builder": __file__})
    out_p.with_name(out_p.stem + "_meta.json").write_text(json.dumps(meta, indent=1))
    for c, m in meta["concepts"].items():
        n = f"{m['rewritten']:4d}/{m['texts']}"
        print(f"{c[:30]:30s} {m['split'] or '':8s} {m['rule']:26s} {n}")
    print("shared texts skipped:", meta["shared_texts_skipped"])


if __name__ == "__main__":
    main()
