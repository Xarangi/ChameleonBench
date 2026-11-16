"""Build the eval battery TSV for a second-seed run from the first seed's own task lines (11-13).

Same commands, same config name (the seed config differs only in train.seed and out), same race seeds; only the adapter
and the output paths change. Battery = what the paper uses: fixed-text held-out val race, gated free-generation races
(train / held-out), held-out-kind races, persistence x2, capability, never-shown panel x8, and the task races for the 17
valid tasks (MMLU x4 and MGSM-ja are not in the paper). Task-race context is 64k for every piece (the first seed needed
64k on some pieces after 32k overflowed; the limit does not change results).

  /usr/bin/python3 configs/tica/mech_1109/seed_tsv.py 2b   -> runs/mech_1109/seed2b.tsv + seed2b_smoke.tsv
  /usr/bin/python3 configs/tica/mech_1109/seed_tsv.py 4b   -> runs/mech_1109/seed4b.tsv + seed4b_smoke.tsv

4b (11-15): the first seed's task races are races/gate_x2500_<task>/kl03_1800_p{1,2} (2 x 16 episodes, 33 turns, 6-of-8
gate, ChameleonBench-gate checkout); their piece commands live in that directory's pieces*.tsv (2 columns) and are
wrapped in race_piece.sh here. The two gated quality races run from ChameleonBench-emptyfix (= final + an empty sampled
answer counts as a catch instead of crashing, which is what killed the 2B seed's trained race once).
"""
import glob
import os
import shlex
import sys

R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs"
M = f"{R}/mech_1109"
SPEC = {
    "2b": dict(
        cfg="final_v4d_tamia_2b_x1800",
        old_adapter=f"{R}/final_v4d_tamia_2b_x1800/resume_1270/adaptive/step_530",
        new_adapter=f"{R}/final_v4d_tamia_2b_x1800_s1/resume_1220/adaptive/step_580",
        new_run="final_v4d_tamia_2b_x1800_s1",
        race_dir="final2b_", race_tag="k2b1800_p",
        sources=[f"{R}/mech_1025/x2b_drain.tsv", f"{M}/paperB.tsv", f"{M}/paperE.tsv"],
        smokes=[(f"{M}/paperQ_smoke.tsv", "q"), (f"{M}/paperB_smoke.tsv", "race")],
    ),
    "4b": dict(
        cfg="final_v4d_k03_tamia_4b",
        old_adapter=f"{R}/final_v4d_k03_tamia_4b/resume_1660/adaptive/step_140",
        new_adapter=f"{R}/final_v4d_k03_tamia_4b_s1/resume_1700/adaptive/step_100",
        new_run="final_v4d_k03_tamia_4b_s1",
        race_dir="gate_x2500_", race_tag="kl03_1800_p", new_race_dir="final4bs1_", new_race_tag="k4b1800s1_p",
        sources=[f"{R}/mech_1025/k034b_drain.tsv", f"{M}/fix4.tsv", f"{M}/paperE.tsv"],
        pieces=f"{R}/races/gate_x2500_*/pieces*.tsv",
        gated_repo="/scratch/s/sneheel/ChameleonBench-emptyfix",
        smokes=[(f"{M}/seed2b_smoke.tsv", "q"), (f"{M}/seed2b_smoke.tsv", "race")],
        smoke_from=("final_v4d_tamia_2b_x1800", f"{R}/final_v4d_tamia_2b_x1800_s1/resume_1220/adaptive/step_580", "smoke_s1_2b"),
    ),
    # 9b (11-16): first-seed task races are races/final9b_<task>/k9b1800_p{1,2}; rerun files first so a piece that
    # was rerun with changed settings uses the settings that produced the stored result.
    "9b": dict(
        cfg="final_v4d_k03_tamia_9b",
        old_adapter=f"{R}/final_v4d_k03_tamia_9b/resume_1660/adaptive/step_140",
        new_adapter=f"{R}/final_v4d_k03_tamia_9b_s1/resume_1590/adaptive/step_210",
        new_run="final_v4d_k03_tamia_9b_s1",
        race_dir="final9b_", race_tag="k9b1800_p",
        sources=[f"{M}/k9b_rr2.tsv", f"{M}/k9b_rr.tsv", f"{M}/k9b.tsv", f"{M}/fix4.tsv", f"{M}/paperG.tsv", f"{M}/paperE.tsv"],
        pieces=f"{R}/races/final9b_*/pieces*.tsv",
        gated_repo="/scratch/s/sneheel/ChameleonBench-emptyfix",
        smokes=[(f"{M}/seed2b_smoke.tsv", "q"), (f"{M}/seed2b_smoke.tsv", "race")],
        smoke_from=("final_v4d_tamia_2b_x1800", f"{R}/final_v4d_tamia_2b_x1800_s1/resume_1220/adaptive/step_580", "smoke_s1_2b"),
        smoke_sub=[("TICA_VLLM_MEM=0.30", "TICA_VLLM_MEM=0.35"), ("TICA_MEM_FRACTION=0.65", "TICA_MEM_FRACTION=0.60")],  # the 9B pieces' memory split
    ),
}
SKIP = ("mmlu_", "mgsm_ja", "/hs/", "/race_train.json", "/race_heldout.json", "step_1400", "step_1000")


def conv(line, s):
    assert s["old_adapter"] in line, line[:200]
    line = line.replace(s["old_adapter"], s["new_adapter"])
    line = line.replace(f"{R}/{s['cfg']}/", f"{R}/{s['new_run']}/")
    line = line.replace(f"races/{s['race_dir']}", f"races/{s.get('new_race_dir', s['race_dir'][:-1] + 's1_')}")
    line = line.replace(s["race_tag"], s.get("new_race_tag", s["race_tag"][:-2] + "s1_p"))
    if s.get("gated_repo") and line.startswith("q\t"):
        line = line.replace("/scratch/s/sneheel/ChameleonBench-final", s["gated_repo"])
    return line.replace("TICA_VLLM_MAXLEN=32768", "TICA_VLLM_MAXLEN=65536")


def main():
    key = sys.argv[1]
    s = SPEC[key]
    out, seen = [], set()
    lines = [l for src in s["sources"] for l in open(src)]
    if s.get("pieces"):  # task races come only from the pieces files (wrapped in race_piece.sh), never from a bare line
        lines = [l for l in lines if "/races/" not in l]
    for src in sorted(glob.glob(s.get("pieces", "")), key=lambda x: ("_rr" not in os.path.basename(x), x)):  # reruns first; 2-column race_launch pieces -> race_piece.sh lines
        for l in open(src):
            f = l.rstrip("\n").split("\t")
            if len(f) != 2 or s["race_tag"] not in f[0]:
                continue
            both = [f]
            if "gate_x2500_json/" in f[0]:  # the first seed's SQL pieces file was not kept; SQL = JSON line with the
                g = l.rstrip("\n")          # task swapped (checked against the 2B lines and the stored kl03_1800 args)
                for a, b in (("gate_x2500_json/", "gate_x2500_sql/"), ("data_json.json", "data_sql.json"),
                             ("inputs_json.json", "inputs_sql.json"), ("JSON data", "SQL code")):
                    assert a in g
                    g = g.replace(a, b)
                both.append(g.split("\t"))
            for f in both:
                lines.append(f"race\t{f[0]}\tbash configs/tica/mech_1109/race_piece.sh {f[0]} {shlex.quote(f[1])}\n")
    if True:
        for line in lines:
            f = line.rstrip("\n").split("\t")
            if len(f) != 3 or f"{R}/{s['cfg']}/" not in f[2] or s["old_adapter"] not in f[2]:
                continue
            if any(k in f[1] for k in SKIP):
                continue
            new = conv(line.rstrip("\n"), s)
            o = new.split("\t")[1]
            if o in seen:
                continue
            seen.add(o)
            os.makedirs(os.path.dirname(o), exist_ok=True)
            if "--gen-out " in new:  # tica_switch_eval.py does not create the gens directory (seed2b: 2 pieces failed)
                os.makedirs(os.path.dirname(new.split("--gen-out ")[1].split()[0]), exist_ok=True)
            out.append(new)
    smoke = [f"rl\t{M}/smoke_9b_race.json\ttrue"]  # rl paths are the first seed's, unchanged code; file exists
    for src, grp in s["smokes"]:
        for line in open(src):
            f = line.rstrip("\n").split("\t")
            if len(f) == 3 and f[0] == grp and "smoke_from" in s and s["smoke_from"][1] in f[2]:
                c, a, n = s["smoke_from"]
                new = line.rstrip("\n").replace(a, s["new_adapter"]).replace(c, s["cfg"]).replace(n, f"smoke_s1_{key}")
                for x, y in s.get("smoke_sub", []):
                    new = new.replace(x, y)
                smoke.append(new)
                break
            if len(f) == 3 and f[0] == grp and s["old_adapter"] in f[2]:
                smoke.append(line.rstrip("\n").replace(s["old_adapter"], s["new_adapter"]).replace("smoke_", f"smoke_s1_"))
                break
    open(f"{M}/seed{key}.tsv", "w").write("\n".join(out) + "\n")
    open(f"{M}/seed{key}_smoke.tsv", "w").write("\n".join(smoke) + "\n")
    groups = {}
    for l in out:
        groups[l.split("\t")[0]] = groups.get(l.split("\t")[0], 0) + 1
    print(len(out), "pieces", groups, "| smoke groups", [l.split("\t")[0] for l in smoke])
    for l in out:
        print(l.split("\t")[1].replace(R + "/", ""))


if __name__ == "__main__":
    main()
