"""Eval battery for the Olmo-3-7B KL0.3 run (11-17), built from the 9B second seed's battery (seed9b.tsv).

Same 50 pieces and arguments as the 9B (same depth and width, so the same memory settings are the starting point).
Only the model-specific parts change:
  - config name, adapter and output paths;
  - code checkout. Olmo 3 has no grouped-query attention, so turn-0 fits with short shared prefixes need the
    one-conversation-per-call read (REPLICATION 11-12). Fixed-text / persistence / capability / panel pieces ran from the
    main checkout for the Qwen models -> here from ChameleonBench-olmofix (= main + that fix) through the launch-script
    copies in configs/tica/mech_1109/olmo/. Gated and task races ran from ChameleonBench-final / -emptyfix -> here
    from ChameleonBench-olmoeval (= final-evals eb2e8bb + the same fix + the empty-answer fix + the Olmo config).

  /usr/bin/python3 configs/tica/mech_1109/olmo_tsv.py [ADAPTER]  -> runs/mech_1109/olmo.tsv + olmo_smoke.tsv
"""
import os
import re
import sys

R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs"
M = f"{R}/mech_1109"
CFG = "final_v4d_k03_olmo3_7b_tamia"
OLD_AD = f"{R}/final_v4d_k03_tamia_9b_s1/resume_1590/adaptive/step_210"
OLD_AD0 = f"{R}/final_v4d_k03_tamia_9b/resume_1660/adaptive/step_140"
AD = sys.argv[1] if len(sys.argv) > 1 else f"{R}/{CFG}/resume_1560/adaptive/step_240"
FIX = "/scratch/s/sneheel/ChameleonBench-olmofix"
EVAL = "/scratch/s/sneheel/ChameleonBench-olmoeval"


def conv(line):
    line = line.replace(OLD_AD, AD).replace(OLD_AD0, AD)
    line = line.replace("final_v4d_k03_tamia_9b_s1", CFG).replace("final_v4d_k03_tamia_9b", CFG)
    line = line.replace("final9bs1_", "finalolmo_").replace("k9b1800s1", "olmo1800")
    line = line.replace("smoke_s1_9b", "smoke_olmo").replace("smoke_9b", "smoke_olmo")
    line = re.sub(r"/scratch/s/sneheel/ChameleonBench-(emptyfix|final|gate|graded)\b", EVAL, line)
    if line.startswith("rl\t"):  # only the pieces that ran from the main checkout; races keep the final-evals script
        line = re.sub(r"bash scripts/slurm/tica_(val|switch_eval|capability)\.sh", r"bash configs/tica/mech_1109/olmo/tica_\1.sh", line)
        line = line.replace("bash scripts/slurm/tica_probe_panel.sh", f"TICA_REPO={FIX} bash scripts/slurm/tica_probe_panel.sh")
    assert "9b" not in line.replace("mech_1109", ""), line[:300]
    return line


def main():
    out = [conv(l.rstrip("\n")) for l in open(f"{M}/seed9b.tsv") if l.strip()]
    for l in out:
        f = l.split("\t")
        os.makedirs(os.path.dirname(f[1]), exist_ok=True)
        if "--gen-out " in l:
            os.makedirs(os.path.dirname(l.split("--gen-out ")[1].split()[0]), exist_ok=True)
    smoke = [conv(l.rstrip("\n")) for l in open(f"{M}/k9b_smoke.tsv") if l.startswith("rl\t")]  # a real fixed-text smoke
    smoke = [l.replace("smoke_olmo_race", "smoke_olmo_rl") for l in smoke]
    smoke += [conv(l.rstrip("\n")) for l in open(f"{M}/seed9b_smoke.tsv") if l.split("\t")[0] in ("q", "race")]
    open(f"{M}/olmo.tsv", "w").write("\n".join(out) + "\n")
    open(f"{M}/olmo_smoke.tsv", "w").write("\n".join(smoke) + "\n")
    g = {}
    for l in out:
        g[l.split("\t")[0]] = g.get(l.split("\t")[0], 0) + 1
    print(len(out), "pieces", g, "| smoke", [l.split("\t")[0] for l in smoke], "| adapter exists:", os.path.isdir(AD))


if __name__ == "__main__":
    main()
