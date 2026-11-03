#!/usr/bin/env python3
"""ONE-COMMAND interactive race (11-03): models x task -> piece TSV -> GPU node job(s) -> dependent
CPU scoring job. Stdlib only (runs on the login node: it writes files and calls sbatch).

Each piece = one tica_switch_eval process on one GPU running --batch episodes AT ONCE with colocated
vLLM generation (TICA_VLLM=1, awake: TICA_VLLM_SLEEP=0) and the fast fork panel (--fast-forks).
Refits follow the training rule only (a new probe after each pass). Pieces of a model get seeds
seed, seed + 1, ...; the same seeds are used for every model (paired episodes). scripts/slurm/
race_node.sh runs the pieces one per GPU; the report (tica_task_race_report.py, or
tica_math_race_report.py for --task math) runs after the node job(s) end (afterany) and writes
RUN/report.json: per refit probe k, beat rate, turns to beat, accuracy of the passing solution and
the fork-panel fluke check.

  python3 scripts/race_launch.py --name mbpp_1103 --task mbpp --episodes 64 \\
      --model strict4b=final_v4d_klx_tamia_4b:/scratch/.../resume_1280/adaptive/step_520 \\
      --model strict2b=final_v4d_klx_tamia_2b:/scratch/.../resume_1200/adaptive/step_600 \\
      --model kl03_2b=final_v4d_tamia_2b_x1800:/scratch/.../resume_1270/adaptive/step_530
  python3 scripts/race_launch.py --spec configs/race/mbpp_strict_kl.json      # the same, from JSON

--model NAME=CONFIG:ADAPTER (ADAPTER "base" = the base model). --spec FILE.json holds the same
fields (CLI flags override; "models": {NAME: [CONFIG, ADAPTER]}). --dry-run writes the files and
prints the sbatch commands without submitting. Resume a timed-out node: rerun the printed node
command (finished pieces are skipped). Outputs (raw text on scratch only, never in the repo):
RUN = --root/NAME: NAME_pJ.json (aggregates), NAME_pJ.jsonl (main turns), NAME_pJ_forks.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART = "/scratch/s/sneheel/next_chameleons_artifacts/tica/runs"
TASK_DIR = f"{ART}/mech_1025/taskrace"
MATH_DIR = f"{ART}/mech_1025/mathrace"
# task -> (topic, data, inputs, gold, answer_max, report script)
TASKS = {
    "mbpp": ("Python code", f"{TASK_DIR}/data_mbpp.json", f"{TASK_DIR}/inputs_mbpp.json",
             f"{TASK_DIR}/gold_mbpp.json", 768, "tica_task_race_report.py"),
    "sql": ("SQL code", f"{TASK_DIR}/data_sql.json", f"{TASK_DIR}/inputs_sql.json",
            f"{TASK_DIR}/gold_sql.json", 768, "tica_task_race_report.py"),
    "json": ("JSON data", f"{TASK_DIR}/data_json.json", f"{TASK_DIR}/inputs_json.json",
             f"{TASK_DIR}/gold_json.json", 768, "tica_task_race_report.py"),
    "jseval": ("JavaScript code", f"{TASK_DIR}/data_jseval.json", f"{TASK_DIR}/inputs_jseval.json",
               f"{TASK_DIR}/gold_jseval.json", 768, "tica_task_race_report.py"),
    "math": ("mathematical reasoning", f"{MATH_DIR}/final_v5_gsm8ktest.json",
             f"{MATH_DIR}/elicit_inputs_gsm8ktest.json", f"{MATH_DIR}/gold.json", 512,
             "tica_math_race_report.py"),
}
PY_TRAIN = "/scratch/s/sneheel/venvs/q35t29/bin/python"


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", help="JSON file with any of the fields below")
    ap.add_argument("--name", help="run name (output dir RUN = ROOT/NAME)")
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--model", action="append", default=[], metavar="NAME=CONFIG:ADAPTER")
    ap.add_argument("--episodes", type=int, help="episodes per model (default 64)")
    ap.add_argument("--turns", type=int, help="turns per episode (default 33)")
    ap.add_argument("--fork-panel", type=int, help="forks per monitored turn (default 8)")
    ap.add_argument("--batch", type=int, help="episodes at once per GPU process (default 16)")
    ap.add_argument("--piece-episodes", type=int,
                    help="episodes per piece process (default = --batch; a multiple of it runs "
                         "several batches in one process, one vLLM start)")
    ap.add_argument("--seed", type=int, help="first piece seed (default 1)")
    ap.add_argument("--max-k", type=int, help="refit levels reported separately (default 12)")
    ap.add_argument("--answer-max", type=int, help="max new tokens (default per task)")
    ap.add_argument("--extra", help="extra tica_switch_eval flags, e.g. \"--set k=v\"")
    ap.add_argument("--gpus", type=int, help="GPUs per node job (default 4)")
    ap.add_argument("--jobs", type=int, help="node jobs to split the pieces over (default 1)")
    ap.add_argument("--time", help="node job wall time (default: estimate)")
    ap.add_argument("--min-per-piece", type=float,
                    help="minutes per piece for the wall-time estimate (default 0.7 x turns + 5)")
    ap.add_argument("--vllm-mem", help="TICA_VLLM_MEM: vLLM's GPU share (default 0.35)")
    ap.add_argument("--hf-mem", help="TICA_MEM_FRACTION: cap of the HF process (default 0.60; "
                                     "with vLLM's share < 1 so an overflow is an HF OOM error, "
                                     "not a silent vLLM hang as seen at --batch 32)")
    ap.add_argument("--vllm-maxlen", help="TICA_VLLM_MAXLEN (default 32768)")
    ap.add_argument("--vllm-venv", help="TICA_VLLM_VENV (default vllm-latest)")
    ap.add_argument("--account", help="Slurm account (default aip-rrabba)")
    ap.add_argument("--exclude", help="nodes to avoid (default tg11105)")
    ap.add_argument("--root", help=f"output root (default {ART}/races)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text()) if a.spec else {}
    defaults = {"episodes": 64, "turns": 33, "fork_panel": 8, "batch": 16, "seed": 1,
                "max_k": 12, "gpus": 4, "jobs": 1, "vllm_mem": "0.35", "hf_mem": "0.60",
                "vllm_maxlen": "32768",
                "vllm_venv": "vllm-latest", "account": "aip-rrabba", "exclude": "tg11105",
                "root": f"{ART}/races", "extra": ""}
    for k, v in {**defaults, **spec}.items():
        if k != "models" and getattr(a, k, None) in (None, []):
            setattr(a, k.replace("-", "_"), v)
    models = dict(spec.get("models", {}))
    for m in a.model:
        name, rest = m.split("=", 1)
        cfg, adapter = rest.split(":", 1)
        models[name] = [cfg, adapter]
    a.models = models
    if not (a.name and a.task and a.models):
        ap.error("--name, --task and at least one --model are required (or a --spec)")
    a.piece_episodes = a.piece_episodes or a.batch
    if a.piece_episodes % a.batch:
        ap.error("--piece-episodes must be a multiple of --batch")
    return a


def sbatch(args: list[str], dry: bool) -> str:
    print("$ sbatch " + " ".join(shlex.quote(x) for x in args))
    if dry:
        return "DRYRUN"
    out = subprocess.run(["sbatch", "--parsable", *args], check=True, capture_output=True,
                         text=True).stdout.strip()
    return out.split(";")[0]


def main() -> None:
    a = parse()
    topic, data, inputs, gold, amax, report = TASKS[a.task]
    run = Path(a.root) / a.name
    run.mkdir(parents=True, exist_ok=True)
    env = (f"TICA_VLLM=1 TICA_VLLM_SLEEP=0 TICA_VLLM_MEM={a.vllm_mem} "
           f"TICA_MEM_FRACTION={a.hf_mem} TICA_VLLM_MAXLEN={a.vllm_maxlen} "
           f"TICA_VLLM_VENV={a.vllm_venv} "
           f"TICA_REPO={shlex.quote(str(REPO))}")
    n_pieces = math.ceil(a.episodes / a.piece_episodes)
    lines = []
    for j in range(n_pieces):  # piece-major: every model's piece j before piece j + 1
        eps = min(a.piece_episodes, a.episodes - j * a.piece_episodes)
        for name, (cfg, adapter) in a.models.items():
            stem = run / f"{name}_p{j + 1}"
            who = ["--base"] if adapter == "base" else ["--adapter", adapter]
            cmd = ["bash", "scripts/slurm/tica_switch_eval.sh", cfg, f"{stem}.json", *who,
                   "--topics", topic, "--answers", "free",
                   "--answer-max", str(a.answer_max or amax), "--switch-prob", "0",
                   "--cond", "real_score", "--max-k", str(a.max_k), "--data", data,
                   "--inputs", inputs, "--batch", str(min(a.batch, eps)), "--turns", str(a.turns),
                   "--fork-panel", str(a.fork_panel), "--fast-forks", "--episodes", str(eps),
                   "--seed", str(a.seed + j), "--fork-out", f"{stem}_forks.jsonl",
                   "--gen-out", f"{stem}.jsonl"]
            lines.append((f"{stem}.json",
                          env + " " + " ".join(shlex.quote(x) for x in cmd)
                          + (" " + a.extra if a.extra else "")))
    per_job = [lines[k :: a.jobs] for k in range(a.jobs)]
    rounds = max(math.ceil(len(x) / a.gpus) for x in per_job)
    # 11-03 smoke (33 turns, batch 16, fork 8): 10-13.5 min per piece incl. startup
    mins = rounds * (a.min_per_piece or 0.7 * a.turns + 5) * (a.piece_episodes / a.batch) + 20
    wall = a.time or f"{int(mins // 60)}:{int(mins % 60):02d}:00"
    (run / "race.json").write_text(json.dumps({**{k: v for k, v in vars(a).items()
                                                   if k not in ("model", "dry_run")},
                                               "repo": str(REPO), "pieces": len(lines)}, indent=1))
    jobs = []
    for k, chunk in enumerate(per_job):
        if not chunk:
            continue
        tsv = run / (f"pieces_{k}.tsv" if a.jobs > 1 else "pieces.tsv")
        tsv.write_text("".join(f"{o}\t{c}\n" for o, c in chunk))
        jobs.append(sbatch([f"--account={a.account}", f"--gres=gpu:h100:{a.gpus}",
                            f"--exclude={a.exclude}", f"--time={wall}",
                            f"--job-name=race-{a.name}",
                            f"--output={REPO}/logs/%x-%j.out",
                            f"--export=ALL,TICA_REPO={REPO},TICA_VLLM_VENV={a.vllm_venv}",
                            str(REPO / "scripts/slurm/race_node.sh"), str(tsv)], a.dry_run))
    races = " ".join(shlex.quote(f"{name}={run}/{name}_p*.jsonl") for name in a.models)
    score = (f"cd {shlex.quote(str(REPO))} && OMP_NUM_THREADS=1 PYTHONPATH=src {PY_TRAIN} "
             f"scripts/{report} --gold {gold} --race {races} "
             + ("--workers 8 " if report == "tica_task_race_report.py" else "")
             + f"--out {run}/report.json")
    sid = sbatch([f"--account={a.account}", "--cpus-per-task=8", "--mem=16G", "--time=1:00:00",
                  f"--dependency=afterany:{':'.join(jobs)}", f"--job-name=race-score-{a.name}",
                  f"--output={REPO}/logs/%x-%j.out", "--wrap", score], a.dry_run)
    print(json.dumps({"run": str(run), "pieces": len(lines), "node_jobs": jobs,
                      "score_job": sid, "wall_per_node": wall, "report": f"{run}/report.json"},
                     indent=1))


if __name__ == "__main__":
    main()
