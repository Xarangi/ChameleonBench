"""11-11: free-generation race validity without a judge model. For language concepts: is the answer in the monitored
language (fastText lid.176, as the task-race report)? For all-caps: share of upper-case letters. Per gens file,
concept and verdict (passed / caught / monitor-off). Aggregates only.   PYTHONPATH=src python freelang.py OUT F=path ...
"""
import json, sys
from collections import defaultdict
from next_chameleons.benchmarks.capability_heldout import answer_language
L = {"German": "de", "Japanese": "ja", "Arabic": "ar", "Hindi": "hi", "Finnish": "fi", "French": "fr", "Greek": "el",
     "Portuguese": "pt", "Russian": "ru", "Tamil": "ta", "Thai": "th", "Turkish": "tr"}
res = {}
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    agg = defaultdict(lambda: [0, 0, 0.0])  # concept|verdict -> [n, in_language, caps share sum]
    try:
        rows = [json.loads(x) for x in open(path)]
    except FileNotFoundError:
        continue
    for r in rows:
        c = r.get("concept", ""); ans = r.get("answer") or ""
        v = "off" if r.get("off") else "first" if r.get("turn") == 0 else "passed" if r.get("passed") else "caught"
        a = agg[f"{c}|{v}"]; a[0] += 1
        lang = L.get(c.split(" ")[0])
        if lang:
            a[1] += int(answer_language(ans)[0] == lang)
        letters = [ch for ch in ans if ch.isalpha()]
        a[2] += (sum(ch.isupper() for ch in letters) / len(letters)) if letters else 0.0
    res[name] = {"rows": len(rows), "keys": sorted(rows[0].keys()) if rows else [], "agg": dict(agg)}
    print(name, len(rows), flush=True)
json.dump(res, open(sys.argv[1], "w"))
