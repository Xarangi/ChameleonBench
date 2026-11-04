import json, sys, os
R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/races"
def f(x, d=2):
    return "-" if x is None else (f"{x[0]:.{d}f}" if isinstance(x, list) else str(x))
for run in sys.argv[1:]:
    for t in ("pyeval", "javaeval", "mgsm_fr", "mgsm_ru", "mgsm_th"):
        p = f"{R}/{run}_{t}/report.json"
        if not os.path.exists(p):
            print(f"### {run} {t}: no report"); continue
        rep = json.load(open(p))
        for m, d in rep.items():
            mg = t.startswith("mgsm")
            fa = f"first-answer acc {f(d['first_answer_acc'])} (n={d['first_answer_acc'][3] if d['first_answer_acc'] else 0})"
            if mg:
                fa += f", in-language {f(d.get('first_answer_in_language'))}, both {f(d.get('first_answer_correct_and_in_language'))}"
            row = d["beaten_in_a_row"]
            print(f"### {t} | {m} | episodes {d.get('episodes')} pieces {len(d.get('pieces', []))} incomplete {d.get('pieces_incomplete')} | {fa} | beaten in a row mean {f(row['mean'])} hist {row['hist']}")
            hdr = "| k | beaten/faced | turns to beat (mean; hist) | passing correct |" + (" in-lang | both |" if mg else "") + " fluke (mean) |"
            print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
            for k in sorted([k for k in d if k.startswith("probe_")], key=lambda x: int(x.split("_")[1])):
                q = d[k]
                tb = f"{f(q['turns_to_beat'], 1)}; {q['turns_to_beat_hist']}" if q["beaten"] else "-"
                line = f"| {k.split('_')[1]} | {q['beaten']}/{q['faced']} | {tb} | {q.get('passing_correct_n') or '-'} |"
                if mg:
                    line += f" {q.get('passing_in_language_n') or '-'} | {q.get('passing_correct_and_in_language_n') or '-'} |"
                line += f" {f(q.get('fluke_share_mean'))} |"
                print(line)
            print()
