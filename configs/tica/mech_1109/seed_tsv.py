"""Build the eval battery TSV for a second-seed run from the first seed's own task lines (11-13).

Same commands, same config name (the seed config differs only in train.seed and out), same race seeds; only the adapter
and the output paths change. Battery = what the paper uses: fixed-text held-out val race, gated free-generation races
(train / held-out), held-out-kind races, persistence x2, capability, never-shown panel x8, and the task races for the 17
valid tasks (MMLU x4 and MGSM-ja are not in the paper). Task-race context is 64k for every piece (the first seed needed
64k on some pieces after 32k overflowed; the limit does not change results).

  /usr/bin/python3 configs/tica/mech_1109/seed_tsv.py 2b   -> runs/mech_1109/seed2b.tsv + seed2b_smoke.tsv
"""
import os
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
}
SKIP = ("mmlu_", "mgsm_ja", "/hs/", "/race_train.json", "/race_heldout.json", "step_1400", "step_1000")


def conv(line, s):
    assert s["old_adapter"] in line, line[:200]
    line = line.replace(s["old_adapter"], s["new_adapter"])
    line = line.replace(f"{R}/{s['cfg']}/", f"{R}/{s['new_run']}/")
    line = line.replace(f"races/{s['race_dir']}", f"races/{s['race_dir'][:-1]}s1_")
    line = line.replace(s["race_tag"], s["race_tag"][:-2] + "s1_p")
    return line.replace("TICA_VLLM_MAXLEN=32768", "TICA_VLLM_MAXLEN=65536")


def main():
    key = sys.argv[1]
    s = SPEC[key]
    out, seen = [], set()
    for src in s["sources"]:
        for line in open(src):
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
            out.append(new)
    smoke = [f"rl\t{M}/smoke_9b_race.json\ttrue"]  # rl paths are the first seed's, unchanged code; file exists
    for src, grp in s["smokes"]:
        for line in open(src):
            f = line.rstrip("\n").split("\t")
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
