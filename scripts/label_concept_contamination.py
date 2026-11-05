"""NEGATIVE-LABEL AUDIT (REPLICATION 2026-10-16f; user 10-16: "make sure these are surely
negatives"). Every text a probe can see as a NEGATIVE for concept c -- other concepts' texts and the
shared concept-free pool -- is labelled by an LLM judge (Qwen3.5-9B, thinking off, greedy) for every
study concept:
  pass 1 (recall): one multi-label prompt per text, "which of these apply" over all concepts;
  pass 2 (precision): every flagged (text, concept) pair re-asked as a yes / no question, P(yes)
  from the first-token log-probs.
Output (no text): {"labels": {text_id: {concept: p_yes}}, "flagged_pass1": {...}, "meta": ...}.
Deception is not judged (needs ground truth; DolusChat has matched truthful negatives).
scripts/negative_contamination_report.py aggregates it per concept. Run in the vllm-serving env.

  python scripts/label_concept_contamination.py --model PATH --data A.json B.json --out L.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

# what the judge is asked about each concept (deception excluded)
DESC = {
    "German language": "written in German",
    "Finnish language": "written in Finnish",
    "Spanish language": "written in Spanish",
    "French language": "written in French",
    "all-caps text": "written entirely or mostly in CAPITAL LETTERS",
    "HTML markup": "contains HTML markup or tags",
    "gibberish": "gibberish: nonsensical or meaningless text",
    "jokes and humor": "jokes or a humorous, jokey style",
    "comforting tone": "a comforting, reassuring or consoling tone",
    "formal tone": "a formal tone or register",
    "casual tone": "a casual, informal or conversational tone",
    "confusion": "an expressed state of confusion or bewilderment",
    "mathematical reasoning": "mathematical reasoning or worked calculations",
    "physics": "physics",
    "chemistry": "chemistry",
    "literature": "literature (books, authors or literary works)",
    "computer code": "computer code or programming",
    "biology": "biology or living organisms",
    "harmfulness (circuit breakers)": "harmful, dangerous or illegal assistance",
    "harmfulness (synthetic)": "harmful, dangerous or illegal assistance",
}
# PROMPT v4 (REPLICATION 2026-10-17n; user 10-17: re-label with the 27B judge, "based on what probes
# or topics it covers broadly and generally enough"). v3e (9B) over-labelled broad concepts (69% of
# sports texts "living organisms": people are organisms; 97% of jokes "casual tone"). v4 asks for
# CENTRAL properties only, the way a probe trained on that concept would see the text, and tightens
# the broadest descriptions. --prompt v3 keeps the old wording (reproducible).
DESC_V4 = {
    **DESC,
    "living organisms": "living organisms as a subject: animals, plants, microbes or biological "
                        "processes (merely involving people does not count)",
    "biology": "biology as a subject: the life sciences (organisms, cells, genetics, ecology, "
               "physiology); merely involving people does not count",
    "casual tone": "a casual, informal, conversational register throughout the text",
    "jokes and humor": "jokes or a humorous style as a central feature of the text",
    "comforting tone": "a comforting, reassuring or consoling tone as a central feature",
    "confusion": "the writer expressing confusion or bewilderment as a central feature",
    "mathematical reasoning": "mathematical reasoning or worked calculations as a central element "
                              "(numbers, units or quantities alone do not count)",
    "physics": "physics as a central subject",
    "chemistry": "chemistry as a central subject",
    "literature": "literature as a subject: books, authors, literary works or literary analysis",
    "computer code": "computer source code or programming as a central element",
    "HTML markup": "HTML markup or tags as a substantial part of the text",
    # v4 concepts (REPLICATION 2026-10-17aa)
    **{f"{lang} language": f"written in {lang}" for lang in (
        "Japanese", "Arabic", "Hindi", "Portuguese", "Turkish", "Russian", "Greek", "Thai",
        "Tamil")},
    "SQL code": "SQL code (table definitions or queries) as a substantial part of the text",
    "JSON data": "JSON data or a JSON schema as a substantial part of the text",
    "JavaScript code": "JavaScript source code as a substantial part of the text",
    "Python code": "Python source code as a substantial part of the text",
    "finance": "finance as a central subject: money, investing, banking, markets or accounting",
    "supportive tone": "a supportive, encouraging tone as a central feature",
    # v5 concepts (REPLICATION 2026-10-18f)
    "formal tone": "a formal register (impersonal, precise, no colloquialisms) throughout the text",
    "cautious tone": "a cautious, hedging tone (qualifications, caveats, uncertainty) as a central "
                     "feature",
    "Java code": "Java source code as a substantial part of the text",
    "computer science": "computer science as a central subject: algorithms, computing systems, "
                        "machine learning or software research (code alone does not count)",
    # emotion: the judge keeps emotion-central texts out of the emotion probes' NEGATIVES (their
    # hard negatives, the other emotions' stories, are negatives by construction)
    **{f"emotion: {e}": f"a person in the text feeling {d}, as a central feature" for e, d in (
        ("happy", "happiness or joy"), ("sad", "sadness or grief"), ("angry", "anger"),
        ("afraid", "fear"), ("guilty", "guilt"), ("surprised", "surprise"),
        ("calm", "calm or serenity"))},
}
FRAME_V4 = ("You label texts for training concept detectors. A property applies only if it is "
            "CENTRAL to the text: a main topic, or a pervasive property of how it is written - "
            "something you would name when summarising the text. Incidental mentions, single "
            "words, or properties shared by almost all texts do not count.\n\n")
# deception needs ground truth (DolusChat / Apollo RepE have matched honest negatives)
SKIP = {"deception", "deception (Apollo RepE)"}
MAXC = 2000  # characters of each text shown to the judge


def concept_desc(c: str, prompt: str = "v3") -> str:
    if prompt == "v4":
        return DESC_V4.get(c, f"the topic '{c}' as a central subject")
    return DESC.get(c, f"the topic '{c}' (substantially about it)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="first N texts only (smoke)")
    ap.add_argument("--shard", default="0/1", help="i/n: label every n-th text from i (one GPU per "
                                                   "shard; the report merges --labels files)")
    ap.add_argument("--prompt", default="v3", choices=["v3", "v4"],
                    help="v4 = central properties only (10-17n)")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs (27B: 2 x A100)")
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--skip-labelled", default="",
                    help="an earlier label file: skip every text of the data files it labelled "
                         "(meta.data); build_final_v5_mw.py merge-labels joins the two (10-21f)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    texts, concepts = {}, []
    for f in a.data:
        d = json.loads(Path(f).read_text())
        for c in d["concepts"]:
            if (c["concept"] not in concepts and c["concept"] not in SKIP
                    and not c["concept"].startswith("deception:")):
                concepts.append(c["concept"])
            for p in ("fit", "calib", "episode"):
                for t in c[p]:
                    texts.setdefault(t["id"], t["text"])
        for ts in d["negatives"].values():
            for t in ts:
                texts.setdefault(t["id"], t["text"])
    if a.skip_labelled:
        old = json.loads(Path(a.skip_labelled).read_text())
        done = set(old.get("labels", {}))  # ids it labelled (data paths: maybe Narval)
        meta = old["meta"]
        files = {f for m in (meta if isinstance(meta, list) else [meta]) for f in m.get("data", [])}
        for f in files:
            if not Path(f).exists():
                continue
            d = json.loads(Path(f).read_text())
            done |= {t["id"] for c in d["concepts"] for p in ("fit", "calib", "episode")
                     for t in c[p]}
            done |= {t["id"] for ts in d["negatives"].values() for t in ts}
        texts = {i: x for i, x in texts.items() if i not in done}
    si, sn = (int(x) for x in a.shard.split("/"))
    ids = sorted(texts)[: a.limit or None][si::sn]
    tok = AutoTokenizer.from_pretrained(a.model)
    menu = "\n".join(f"{k + 1}. {concept_desc(c, a.prompt)}" for k, c in enumerate(concepts))
    frame = FRAME_V4 if a.prompt == "v4" else ""

    def chat(user):  # rendered text -> ids (tokenize=True returns a BatchEncoding here)
        txt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        return tok.encode(txt, add_special_tokens=False)

    def p1(x):
        return chat(f"{frame}Text:\n\"\"\"\n{x[:MAXC]}\n\"\"\"\n\nWhich of the following "
                    f"properties clearly apply to this text?\n{menu}\n\nAnswer with the numbers of "
                    "ALL that clearly apply, comma-separated, or 'none'. No explanation.")

    def p2(x, c):
        return chat(f"{frame}Text:\n\"\"\"\n{x[:MAXC]}\n\"\"\"\n\nDoes this text clearly have "
                    f"this property: {concept_desc(c, a.prompt)}? Answer Yes or No.")

    t0 = time.time()
    llm = LLM(model=a.model, dtype="auto", gpu_memory_utilization=a.mem, max_model_len=4096,
              enable_prefix_caching=True, max_num_seqs=a.max_num_seqs,
              max_num_batched_tokens=32768, tensor_parallel_size=a.tp)
    load = round(time.time() - t0)
    out1 = llm.generate([TokensPrompt(prompt_token_ids=p1(texts[i])) for i in ids],
                        SamplingParams(temperature=0.0, max_tokens=40), use_tqdm=False)
    flagged = {}
    for i, o in zip(ids, out1, strict=True):
        nums = {int(n) for n in re.findall(r"\d+", o.outputs[0].text)}
        hit = [concepts[n - 1] for n in sorted(nums) if 1 <= n <= len(concepts)]
        if hit:
            flagged[i] = hit
    t1 = round(time.time() - t0)
    pairs = [(i, c) for i, cs in flagged.items() for c in cs]
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
    out2 = llm.generate([TokensPrompt(prompt_token_ids=p2(texts[i], c)) for i, c in pairs],
                        SamplingParams(temperature=0.0, max_tokens=1, logprobs=20),
                        use_tqdm=False)
    import math

    labels: dict[str, dict[str, float]] = {}
    for (i, c), o in zip(pairs, out2, strict=True):
        lp = o.outputs[0].logprobs[0]
        py = sum(math.exp(lp[t].logprob) for t in yes if t in lp)
        pn = sum(math.exp(lp[t].logprob) for t in no if t in lp)
        labels.setdefault(i, {})[c] = round(py / (py + pn), 4) if py + pn > 0 else 0.0
    rep = {"labels": labels, "flagged_pass1": flagged,
           "meta": {"model": a.model, "data": a.data, "n_texts": len(ids),
                    "concepts": concepts, "n_flagged_texts": len(flagged),
                    "n_pairs_verified": len(pairs), "load_sec": load, "pass1_sec": t1,
                    "total_sec": round(time.time() - t0), "maxc": MAXC, "prompt": a.prompt,
                    "skip_labelled": a.skip_labelled,
                    "desc": DESC_V4 if a.prompt == "v4" else DESC}}
    Path(a.out).write_text(json.dumps(rep))
    print(json.dumps(rep["meta"], indent=1), flush=True)


if __name__ == "__main__":
    main()
