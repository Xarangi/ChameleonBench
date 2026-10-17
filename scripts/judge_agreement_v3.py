"""Second-judge reliability check of the concept labels (REPLICATION 2026-10-16o). A stratified
sample of (text, concept) pairs from the Qwen3.5-9B label audit is re-asked to a stronger judge
(Qwen3.5-27B-FP8, same yes / no prompt, P(yes) from first-token log-probs): strata = judged YES,
flagged-then-NO, unflagged random pairs, and a concept's OWN texts the 9B judge did not confirm
(the positives the positive filter drops). Reports agreement per stratum + Cohen's kappa. No text
is printed. Run in the vllm-serving env.

  python scripts/judge_agreement_v3.py --model PATH --data D.json --labels L.json --out O.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_concept_contamination import MAXC, SKIP, concept_desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=400, help="pairs per stratum")
    ap.add_argument("--mem", type=float, default=0.9)
    ap.add_argument("--eager", action="store_true", help="no CUDA graphs (fits 27B-FP8 on 1 GPU)")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs (27B OOMs on 1 A100)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    d = json.loads(Path(a.data).read_text())
    L = json.loads(Path(a.labels).read_text())
    lab, fl = L["labels"], L["flagged_pass1"]
    texts, own = {}, {}
    for c in d["concepts"]:
        for p in ("fit", "calib", "episode"):
            for t in c[p]:
                texts[t["id"]] = t["text"]
                own.setdefault(t["id"], c["concept"])
    for ts in d["negatives"].values():
        for t in ts:
            texts.setdefault(t["id"], t["text"])
    concepts = [c["concept"] for c in d["concepts"] if c["concept"] not in SKIP]
    rng = random.Random(0)
    yes = [(i, c) for i, cs in lab.items() for c, p in cs.items() if p >= 0.5 and i in texts]
    no = [(i, c) for i, cs in lab.items() for c, p in cs.items() if p < 0.5 and i in texts]
    own_unconf = [(i, c) for i, c in own.items()
                  if c in concepts and lab.get(i, {}).get(c, 0) < 0.5]
    ids = list(texts)
    rand = []
    while len(rand) < a.n:
        i, c = rng.choice(ids), rng.choice(concepts)
        if c not in fl.get(i, []) and own.get(i) != c:
            rand.append((i, c))
    strata = {"judged_yes": rng.sample(yes, min(a.n, len(yes))),
              "flagged_then_no": rng.sample(no, min(a.n, len(no))),
              "unflagged_random": rand,
              "own_unconfirmed": rng.sample(own_unconf, min(a.n, len(own_unconf)))}
    tok = AutoTokenizer.from_pretrained(a.model)

    def prompt(i, c):
        txt = tok.apply_chat_template(
            [{"role": "user",
              "content": f"Text:\n\"\"\"\n{texts[i][:MAXC]}\n\"\"\"\n\nDoes this text clearly "
                         f"have this property: {concept_desc(c)}? Answer Yes or No."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok.encode(txt, add_special_tokens=False)

    llm = LLM(model=a.model, gpu_memory_utilization=a.mem, max_model_len=4096,
              enable_prefix_caching=True, tensor_parallel_size=a.tp,
              enforce_eager=a.eager, max_num_seqs=64)
    yt = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
    nt = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
    rep, allp = {}, []
    for s, pairs in strata.items():
        outs = llm.generate([TokensPrompt(prompt_token_ids=prompt(i, c)) for i, c in pairs],
                            SamplingParams(temperature=0.0, max_tokens=1, logprobs=20),
                            use_tqdm=False)
        big = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            py = sum(math.exp(lp[t].logprob) for t in yt if t in lp)
            pn = sum(math.exp(lp[t].logprob) for t in nt if t in lp)
            big.append(py / (py + pn) >= 0.5 if py + pn > 0 else False)
        small = [lab.get(i, {}).get(c, 0.0) >= 0.5 for i, c in pairs]
        agree = sum(x == y for x, y in zip(big, small, strict=True)) / max(1, len(pairs))
        rep[s] = {"n": len(pairs), "big_yes_rate": round(sum(big) / max(1, len(big)), 3),
                  "small_yes_rate": round(sum(small) / max(1, len(small)), 3),
                  "agreement": round(agree, 3)}
        allp += list(zip(big, small, strict=True))
    n = len(allp)
    po = sum(x == y for x, y in allp) / n
    pb, ps = sum(x for x, _ in allp) / n, sum(y for _, y in allp) / n
    pe = pb * ps + (1 - pb) * (1 - ps)
    rep["kappa_pooled"] = round((po - pe) / (1 - pe), 3) if pe < 1 else None
    rep["meta"] = {"model": a.model, "labels": a.labels, "data": a.data}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
