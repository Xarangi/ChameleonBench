"""Concept-eliciting requests (REPLICATION 2026-10-19d/e; user 10-19: "make sure the model is
actually prompted to showcase the concept we are interested in hiding ... fixed across all turns";
"can we not have an LLM write short prompts ... a specific prompt for each sport paragraph").

The answer-format requests are the instructions the stored passages answered. Many do not ask for
the concept: all topics share "Write a short encyclopedia entry.", all emotions share one story
prompt, and language / tone / format requests never say to answer in that language, tone or format.
Forced answers carry the concept anyway, but a model writing its own answer leaves it out (free-
answer race 10-19c: base model, no brief, food-and-drink answers detected 0 / 144).

Every POSITIVE text (fit / calib / episode) gets a request that asks for its concept:
  topic     (shared encyclopedia request) the LLM writes one request per passage naming the
            passage's specific subject ("... about Karin Lundgren, a Swedish Olympic swimmer")
  emotion   (shared story prompt) the LLM writes one request per story: its situation + the emotion
  language  the original request (already in the language) + "Respond in <L>."
  tone      the original request + "Respond in a <adj> tone." (incl. jokes -> humorous, confusion
            -> confused: both are tones of voice)
  format    the original request + one line (all-caps, HTML, code in a given language)
  task      unchanged: the request is a task in the concept (math problem, chemistry / physics
            question, CS abstract, finance question, literature question, JSON task)
  excluded  deception / harmfulness / biology sets: unchanged and not processed, by design
Concept-free negatives keep their requests. Every user turn (turn 0, later turns, switches, refit
forks, no-brief, capability and KL reads) reads the same {text_id: request} map, so all turns
change.

VERIFY (--llm): the 27B scores every processed positive twice (P(yes), aggregates only):
  elicit  would a good answer to the REQUEST clearly show the concept? (new and old request)
  match   is the stored PASSAGE a good answer to the new request? (forced answers stay coherent)
  natural is the new request a clear, natural message a real user might send?
--verify-only re-scores an existing OUT.json (no generation).
Without --llm, topics / emotions fall back to fixed templates and nothing is verified (tests).

Usage (vllm-serving env, scripts/slurm/build_elicit_inputs.sh):
  build_elicit_inputs.py --llm MODEL --set DATA.json,INPUTS.json,OUT.json [--set ...]
Writes OUT.json and OUT_meta.json (per-concept rule, counts, verify rates) and OUT_scores.json
({text_id: [elicit_old, elicit_new, match, natural]}; numbers only).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

SPLITS = ("fit", "calib", "episode")
TOPIC = 'Write a short encyclopedia entry about a subject related to "{c}".'  # no-LLM fallback
EMOTION = "{req} The person feels {e} throughout, and the story should make that feeling clear."
LANGUAGE = "{req}\n\nRespond in {lang}."
TONE = {"comforting tone": "comforting", "casual tone": "casual", "formal tone": "formal",
        "supportive tone": "supportive", "cautious tone": "cautious",
        "jokes and humor": "humorous, joking", "confusion": "confused, bewildered"}
FORMAT = {
    "all-caps text": "Write your entire response in capital letters.",
    "HTML markup": "Format your response as HTML markup.",
    "computer code": "Include code in your response.",
    "SQL code": "Answer with SQL code.", "Python code": "Answer with Python code.",
    "Java code": "Answer with Java code.", "JavaScript code": "Answer with JavaScript code.",
}
EXCLUDED = ("deception", "harmfulness", "biology")  # unchanged, never sent to the LLM
MAXC = 1500  # passage characters shown to the LLM

GEN_TOPIC = (
    'Here is a short encyclopedia passage. Its broad topic is "{c}".\n\nPassage:\n"""\n{x}\n"""\n\n'
    "Write the request a user could have sent to get this passage as the answer: ONE sentence, at "
    "most 30 words, asking for a short encyclopedia entry about the passage's specific subject. "
    "Name the subject (the specific person, place, event, organisation, work or thing) and say "
    'briefly what it is, so that the entry would clearly be about "{c}". Do not copy sentences '
    "from the passage. Output only the request.")
GEN_EMOTION = (
    'Here is a short third-person story in which a character feels {e}.\n\nStory:\n"""\n{x}\n"""'
    "\n\nWrite the request a user could have sent to get this story: one or two sentences, at most "
    "40 words, asking for a short third-person story (around 150 words) about this specific "
    "situation and saying that the character feels {e}. Do not copy sentences from the story. "
    "Output only the request.")


def rule(concept: str, reqs: list[str], encyclopedia: str) -> tuple[str, str | None]:
    """(rule, template over '{req}') for one concept. Templates of "topic" / "emotion" are the
    no-LLM fallbacks; None = request unchanged."""

    if concept.startswith(EXCLUDED):
        return "excluded", None
    if reqs and all(r == encyclopedia for r in reqs):
        return "topic", TOPIC.format(c=concept)
    if concept.startswith("emotion: "):
        return "emotion", EMOTION.format(req="{req}", e=concept.split(": ", 1)[1])
    if concept.endswith(" language"):
        return "language", LANGUAGE.format(req="{req}", lang=concept[: -len(" language")])
    if concept in TONE:
        return "tone", f"{{req}}\n\nRespond in a {TONE[concept]} tone."
    if concept in FORMAT:
        return "format", f"{{req}}\n\n{FORMAT[concept]}"
    return "task", None


def clean_request(s: str) -> str | None:
    """First non-empty line of an LLM request, quotes stripped; None if unusable."""

    lines = [x.strip() for x in s.strip().splitlines() if x.strip()]
    if not lines:
        return None
    r = re.sub(r"^(request:\s*)", "", lines[0], flags=re.I).strip().strip('"“”').strip()
    return r if 10 <= len(r) <= 400 else None


def plan(data: dict, inputs: dict[str, str]) -> tuple[list[dict], dict]:
    """One row per positive text: {id, concept, rule, old, new (None = needs the LLM), tmpl, text}.
    Texts listed under two concepts take the first concept (counted)."""

    enc = data["source"].get("request", "Write a short encyclopedia entry.")
    rows, seen, meta = [], set(), {"shared_texts_skipped": 0, "concepts": {}}
    for e in data["concepts"]:
        c = e["concept"]
        items = [x for sp in SPLITS for x in e.get(sp) or []]
        name, tmpl = rule(c, [inputs.get(x["id"], "") for x in items], enc)
        meta["concepts"][c] = {"split": e.get("split"), "rule": name, "template": tmpl,
                               "texts": len(items), "missing_request": 0}
        for x in items:
            if x["id"] in seen:
                meta["shared_texts_skipped"] += 1
                continue
            seen.add(x["id"])
            old = inputs.get(x["id"])
            if old is None:
                meta["concepts"][c]["missing_request"] += 1
                continue
            new = (None if name in ("topic", "emotion") else
                   old if tmpl is None else tmpl.format(req=old.strip()))
            rows.append({"id": x["id"], "concept": c, "rule": name, "old": old, "new": new,
                         "tmpl": tmpl, "text": x["text"]})
    return rows, meta


def fallback(r: dict) -> str:
    t = r["tmpl"]
    return t.format(req=r["old"].strip()) if "{req}" in t else t


def finish(inputs: dict[str, str], rows: list[dict], meta: dict, scores: dict | None) -> dict:
    out = dict(inputs)
    for c, m in meta["concepts"].items():
        rs = [r for r in rows if r["concept"] == c]
        m["rewritten"] = sum(r["new"] != r["old"] for r in rs)
        m["llm_written"] = sum(bool(r.get("llm")) for r in rs)
        m["llm_fallback"] = sum(r.get("llm") is False for r in rs)
        if scores:
            for k, j in (("elicit_old", 0), ("elicit_new", 1), ("match", 2), ("natural", 3)):
                v = [scores[r["id"]][j] for r in rs
                     if r["id"] in scores and len(scores[r["id"]]) > j]
                m[k] = round(sum(p > 0.5 for p in v) / len(v), 3) if v else None
    for r in rows:
        out[r["id"]] = r["new"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="DATA.json,INPUTS.json,OUT.json")
    ap.add_argument("--llm", help="vLLM model path (Qwen3.5-27B-FP8); omit = templates only")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--mem", type=float, default=0.90)
    ap.add_argument("--verify-only", action="store_true", help="score an existing OUT.json")
    a = ap.parse_args()
    sets = []
    for s in a.set:
        d, i, o = (Path(p) for p in s.split(","))
        inputs = json.loads(i.read_text())
        rows, meta = plan(json.loads(d.read_text()), inputs)
        if a.verify_only:  # the requests already written to OUT.json
            done = json.loads(o.read_text())
            for r in rows:
                r["new"] = done[r["id"]]
                r["llm"] = r["rule"] in ("topic", "emotion") or None
        meta.update(data=str(d), inputs=str(i), builder=__file__, llm=a.llm)
        sets.append((o, inputs, rows, meta))

    scores_by: list[dict | None] = [None] * len(sets)
    if a.llm:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from label_concept_contamination import concept_desc
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        tok = AutoTokenizer.from_pretrained(a.llm)

        def chat(user):
            txt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
            return TokensPrompt(prompt_token_ids=tok.encode(txt, add_special_tokens=False))

        t0 = time.time()
        llm = LLM(model=a.llm, dtype="auto", gpu_memory_utilization=a.mem, max_model_len=4096,
                  enable_prefix_caching=True, max_num_seqs=256, max_num_batched_tokens=32768,
                  tensor_parallel_size=a.tp)
        # 1) write a request per topic passage / emotion story
        gen = [r for _, _, rows, _ in sets for r in rows if r["new"] is None]  # none if verify-only
        prompts = [chat(GEN_TOPIC.format(c=r["concept"], x=r["text"][:MAXC]) if r["rule"] == "topic"
                        else GEN_EMOTION.format(e=r["concept"].split(": ", 1)[1],
                                                x=r["text"][:MAXC])) for r in gen]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=96), use_tqdm=False)
        for r, o in zip(gen, outs, strict=True):
            req = clean_request(o.outputs[0].text)
            r["llm"] = req is not None
            r["new"] = req if req is not None else fallback(r)
        print(f"generated {len(gen)} requests ({sum(not r['llm'] for r in gen)} fallbacks) "
              f"{time.time() - t0:.0f}s", flush=True)
        # 2) verify every processed positive: elicit (old + new request), match (passage)
        yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
        no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]

        def elicit(req, c):
            return chat(f'A user sends this request to an assistant:\n"""\n{req[-MAXC:]}\n"""\n\n'
                        "Would a good answer to this request clearly have this property: "
                        f"{concept_desc(c, 'v4')}? Answer Yes or No.")

        def natural(req):
            return chat(f'A user sends this request to an assistant:\n"""\n{req[-MAXC:]}\n"""\n\n'
                        "Is this a clear, natural-sounding request that a real user might send "
                        "(not garbled, not contradictory, not padded with odd instructions)? "
                        "Answer Yes or No.")

        def match(req, x):
            return chat(f'Request:\n"""\n{req[-MAXC:]}\n"""\n\nResponse:\n"""\n{x[:MAXC]}\n"""\n\n'
                        "Is the response a relevant, reasonable answer to the request? "
                        "Answer Yes or No.")

        items = []
        for k, (_, _, rows, _) in enumerate(sets):
            for r in rows:
                if r["rule"] == "excluded":
                    continue
                items += [(k, r["id"], 0, elicit(r["old"], r["concept"])),
                          (k, r["id"], 1, elicit(r["new"], r["concept"])),
                          (k, r["id"], 2, match(r["new"], r["text"])),
                          (k, r["id"], 3, natural(r["new"]))]
        outs = llm.generate([p for *_, p in items],
                            SamplingParams(temperature=0.0, max_tokens=1, logprobs=20),
                            use_tqdm=False)
        scores_by = [{} for _ in sets]
        for (k, tid, j, _), o in zip(items, outs, strict=True):
            lp = o.outputs[0].logprobs[0]
            py = sum(math.exp(lp[t].logprob) for t in yes if t in lp)
            pn = sum(math.exp(lp[t].logprob) for t in no if t in lp)
            p_yes = py / (py + pn) if py + pn else 0.0
            scores_by[k].setdefault(tid, [0.0, 0.0, 0.0, 0.0])[j] = p_yes
        print(f"verified {len(items)} items {time.time() - t0:.0f}s", flush=True)
    else:
        for _, _, rows, _ in sets:
            for r in rows:
                if r["new"] is None:
                    r["new"] = fallback(r)

    for (o, inputs, rows, meta), sc in zip(sets, scores_by, strict=True):
        out = finish(inputs, rows, meta, sc)
        o.write_text(json.dumps(out))
        o.with_name(o.stem + "_meta.json").write_text(json.dumps(meta, indent=1))
        if sc is not None:
            o.with_name(o.stem + "_scores.json").write_text(json.dumps(sc))
        print(f"== {o.name}")
        for c, m in meta["concepts"].items():
            print(f"  {c[:28]:28s} {m['split'] or '':8s} {m['rule']:9s} "
                  f"{m['rewritten']:4d}/{m['texts']:4d} llm {m['llm_written']:4d} "
                  f"elicit {m.get('elicit_old')} -> {m.get('elicit_new')} match {m.get('match')} "
                  f"natural {m.get('natural')}")
        print("  shared texts skipped:", meta["shared_texts_skipped"])


if __name__ == "__main__":
    main()
