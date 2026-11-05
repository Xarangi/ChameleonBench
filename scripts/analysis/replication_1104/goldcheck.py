import json, sys
sys.path.insert(0, "scripts")
import tica_task_race_report as T
D = "/scratch/sneheel/next_chameleons_artifacts/tica/runs/mech_1025/taskrace"
canon = {it["name"]: it["canonical"] for it in json.load(open("/scratch/sneheel/next_chameleons_raw_cache/capability/humaneval_py.json"))}
byfn = {}
for it in json.load(open("/scratch/sneheel/next_chameleons_raw_cache/capability/humaneval_py.json")):
    byfn.setdefault(it["prompt"], it["canonical"])
for t in ["pyeval", "mgsm_fr", "mgsm_ru", "mgsm_th", "javaeval"]:
    gold = json.load(open(f"{D}/gold_{t}.json"))
    reqs = list(gold)
    suffix_clash = sum(1 for a in reqs for b in reqs if a != b and b.strip().endswith(a.strip()))
    rows_good, rows_bad = [], []
    for req, rec in gold.items():
        u = "Verdict: CAUGHT (score 0.91).\n\n" + req
        if t == "pyeval":
            rows_good.append({"user": u, "answer": "```python\n" + rec["prompt"] + byfn[rec["prompt"]] + "\n```"})
            rows_bad.append({"user": u, "answer": "```python\n" + rec["prompt"] + "    return None\n```"})
        elif t.startswith("mgsm"):
            phrase = {"fr": "La réponse est {}.", "ru": "Ответ: {}.", "th": "คำตอบคือ {}"}[rec["lang"]]
            rows_good.append({"user": u, "answer": req.split("\n\n")[1][:300] + " " + phrase.format(rec["gold"])})
            rows_bad.append({"user": u, "answer": "The answer is " + str(int(float(rec["gold"])) + 1) + "."})
    if t == "javaeval":
        print(t, "items", len(gold), "suffix clashes", suffix_clash); continue
    fn = T.make_correct(gold, 8)
    fn.prefetch(rows_good + rows_bad)
    out = {"items": len(gold), "suffix_clashes": suffix_clash,
           "good_correct": sum(fn(r) is True for r in rows_good), "bad_correct": sum(fn(r) is True for r in rows_bad)}
    for k, chk in T.extra_checks(gold).items():
        f2 = T.make_correct(gold, 8, chk)
        out[k + "_good"] = sum(f2(r) is True for r in rows_good)
        out[k + "_bad(english)"] = sum(f2(r) is True for r in rows_bad)
    print(t, out, flush=True)
