"""Untrained Olmo-3-7B control (user approved 11-19), built from the trained Olmo task-race pieces.

As for the untrained 2B / 9B controls (launch_paperD.sh): the 17 gated task races at HALF size = piece p1 (16 episodes,
seed 1), identical commands to the trained Olmo p1 pieces except `--adapter PATH` -> `--base` and the output names
(races/finalbaseolmo_<task>/baseolmo_p1*). Plus base capability (GSM8K / MMLU clean + the two briefs), same command as
the trained Olmo capability piece with --base. Smokes reuse the passed Olmo smoke outputs (same scripts, same model).

  /usr/bin/python3 configs/tica/mech_1109/baseolmo_tsv.py  -> runs/mech_1109/baseolmo.tsv + baseolmo_smoke.tsv
Node: NG=2 SLOTS=2 (each race piece uses its GPU and GPU+2 for the vLLM server).
"""
import os
import re

R = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs"
M = f"{R}/mech_1109"
AD = f"{R}/final_v4d_k03_olmo3_7b_tamia/resume_1560/adaptive/step_240"


def conv(line):
    assert f"--adapter {AD}" in line, line[:200]
    line = line.replace(f"--adapter {AD}", "--base")
    line = line.replace("/finalolmo_", "/finalbaseolmo_").replace("/olmo1800_p1", "/baseolmo_p1")
    assert "olmo1800" not in line and "step_240" not in line, line[:300]
    return line


out = []
for f in ("olmo_race2a.tsv", "olmo_race2b.tsv"):
    out += [conv(l.rstrip("\n")) for l in open(f"{M}/{f}") if "_p1.json\t" in l]
assert len(out) == 17, len(out)
cap = [l.rstrip("\n") for l in open(f"{M}/olmo.tsv") if "tica_capability.sh" in l]
assert len(cap) == 1
c = cap[0].replace(f"--adapter {AD}", "--base").replace("quality/step_1800/capability.json", "quality/base/capability.json")
assert "step_240" not in c and "step_1800" not in c
out.append(c)
for l in out:
    os.makedirs(os.path.dirname(l.split("\t")[1]), exist_ok=True)
open(f"{M}/baseolmo.tsv", "w").write("\n".join(out) + "\n")
smoke = [l for l in open(f"{M}/olmo_race2_smoke.tsv") if l.strip()] + [l for l in open(f"{M}/olmo_smoke.tsv") if l.startswith("rl\t")]
open(f"{M}/baseolmo_smoke.tsv", "w").write("".join(smoke))
print(len(out), "pieces | smoke groups", [l.split("\t")[0] for l in smoke], [os.path.exists(l.split("\t")[1]) for l in smoke])
