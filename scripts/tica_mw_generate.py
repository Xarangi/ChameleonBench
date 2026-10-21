"""MODEL-WRITTEN concept texts, step 1: the BASE model answers the published concept-eliciting
requests (REPLICATION 2026-10-21f; docs/tica/V4_MODEL_WRITTEN_DATA.md). Neural Chameleons trained
on model-written texts (gemma-2-27b-it generations filtered by an LLM judge); the v5 split pairs
published requests with published responses. Here the response side becomes the base model's own
answer to the same request, so base probes are fit on the states the model has when it writes the
concept itself.

Prompting = the 10-19i probe-transfer test (job 6392489, scripts/tica_probe_transfer.py): one user
turn = the text's request from the elicit request file (the tagged layout's "nobrief" opening is
exactly the request), no system prompt, thinking off. The prompt string is the Policy's
render_keep_reasoning output for that conversation (tests/test_tica_mw_data.py checks it).

Offline vLLM in the vllm-serving env (scripts/slurm/tica_mw_build.sh). Output: JSONL in the
tica_free_judge.py record shape plus gid / src_id / src_split / sample. SCRATCH ONLY, never commit;
this script prints aggregates only.

  python scripts/tica_mw_generate.py --model 4B --data D.json --inputs ELICIT.json \
      --concepts train --exclude "mathematical reasoning" --samples 3 --shard 0/2 --out G.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

SPLITS = ("fit", "calib", "episode")


def text_id(text: str) -> str:  # = next_chameleons.tica.answer_format.text_id (no package import)
    return "ax-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def render_user(request: str) -> str:
    """Policy.render (keep_think_history, thinking off) of [user: request] + generation prompt."""

    return (f"<|im_start|>user\n{request.strip()}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def select_concepts(data: dict, spec: str, exclude: list[str]) -> list[dict]:
    """spec: a concept split name ('train', 'heldout') or a '|'-separated list of concept names."""

    if spec in ("train", "heldout", "ood"):
        cs = [c for c in data["concepts"] if c["split"] == spec]
    else:
        want = spec.split("|")
        cs = [c for c in data["concepts"] if c["concept"] in want]
        missing = set(want) - {c["concept"] for c in cs}
        if missing:
            raise SystemExit(f"unknown concepts: {sorted(missing)}")
    return [c for c in cs if c["concept"] not in set(exclude)]


def sources(data: dict, inputs: dict, spec: str, exclude: list[str], max_src: int = 0
            ) -> list[dict]:
    """One row per source text: its id, split, concept, request and length (the build matches
    lengths). max_src > 0 keeps the first max_src texts of each concept (smoke)."""

    rows = []
    for c in select_concepts(data, spec, exclude):
        k = 0
        for sp in SPLITS:
            for t in c[sp]:
                if max_src and k >= max_src:
                    break
                req = inputs.get(t["id"])
                if not req:
                    raise SystemExit(f"{c['concept']}: text {t['id']} has no request")
                rows.append({"src_id": t["id"], "src_split": sp, "concept": c["concept"],
                             "user": req, "src_len": len(t["text"])})
                k += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--inputs", required=True, help="elicit request file {text_id: request}")
    ap.add_argument("--concepts", default="train")
    ap.add_argument("--exclude", default="mathematical reasoning",
                    help="'|'-separated concepts that keep their original texts")
    ap.add_argument("--samples", type=int, default=3, help="answers per source text")
    ap.add_argument("--max-src", type=int, default=0, help="smoke: first N texts per concept")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n over source texts")
    ap.add_argument("--mem", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    data = json.loads(Path(a.data).read_text())
    inputs = json.loads(Path(a.inputs).read_text())
    rows = sources(data, inputs, a.concepts, [x for x in a.exclude.split("|") if x], a.max_src)
    si, sn = (int(x) for x in a.shard.split("/"))
    rows = rows[si::sn]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.mem,
              max_model_len=3072 + a.max_new, enable_prefix_caching=True, max_num_seqs=512,
              max_num_batched_tokens=32768)
    load = round(time.time() - t0)
    prompts, params = [], []
    for r in rows:
        prompts.append(TokensPrompt(prompt_token_ids=tok.encode(render_user(r["user"]),
                                                                add_special_tokens=False)))
        seed = (int(hashlib.sha256(r["src_id"].encode()).hexdigest()[:8], 16) + a.seed) % 2**31
        params.append(SamplingParams(n=a.samples, temperature=a.temp, top_p=a.top_p,
                                     top_k=a.top_k, max_tokens=a.max_new, seed=seed))
    outs = llm.generate(prompts, params, use_tqdm=False)
    gen_sec = round(time.time() - t0) - load
    n_tok, fin = 0, Counter()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs, strict=True):
            for k, s in enumerate(o.outputs):
                done = s.finish_reason == "stop"
                n_tok += len(s.token_ids)
                fin[(r["concept"], done)] += 1
                f.write(json.dumps({"gid": f"{r['src_id']}:{k}", "src_id": r["src_id"],
                                    "src_split": r["src_split"], "src_len": r["src_len"],
                                    "sample": k, "concept": r["concept"], "cond": "nobrief",
                                    "turn": 0, "off": False, "detected": None, "passed": None,
                                    "finished": done, "ntok": len(s.token_ids),
                                    "answer": s.text.strip(), "user": r["user"]}) + "\n")
    meta = {"model": a.model, "data": a.data, "inputs": a.inputs, "concepts": a.concepts,
            "exclude": a.exclude, "samples": a.samples, "max_new": a.max_new, "temp": a.temp,
            "top_p": a.top_p, "top_k": a.top_k, "seed": a.seed, "shard": a.shard,
            "n_src": len(rows), "n_gen": sum(fin.values()), "gen_tokens": n_tok,
            "load_sec": load, "gen_sec": gen_sec,
            "finished": {c: round(fin[(c, True)] / max(1, fin[(c, True)] + fin[(c, False)]), 3)
                         for c in sorted({c for c, _ in fin})}}
    Path(a.out + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
