# FINAL v4 on TamIA: quickstart

Everything the final v4 run needs, from a fresh login on TamIA (or any Alliance cluster). Background,
budget and the whole-node launcher: `docs/tica-other-hpc.md`. Recipe and decisions:
`docs/tica-v4-recipe.md`. Config: `configs/tica/final_v4_4b.yaml`.

## 1. Code

```bash
git clone -b tica-v3 git@github.com:Xarangi/ChameleonBench.git ~/chameleons
```

All configs, builders, labelers, tests, docs and REPLICATION.md are in git. Data, labels, checkpoints
and the model are not (raw text / size): step 3 pulls them from Narval.

## 2. Environments (login node; uv needs internet)

```bash
cd ~/chameleons
VENVS=/scratch/$USER/venvs bash scripts/env/q35t29_build.sh       # Qwen3.5 venv (transformers 5.x)
tar -cf /scratch/$USER/venvs/q35t29.tar -C /scratch/$USER/venvs q35t29   # jobs unpack it node-local
uv sync --extra dev                                                # main .venv (CPU tests; transformers 4.57.6)
```

Never install transformers 5.x into the main `.venv`.

## 3. Model and data

```bash
bash scripts/tica_sync_v4.sh narval.alliancecan.ca
```

## 4. Paths

Configs and scripts use `/home/sneheel/...` and `/scratch/sneheel/...`. Same username on TamIA: nothing to
do. Otherwise run the rewrite in `docs/tica-other-hpc.md` section 4 on a local branch (do not push it).
Also `scripts/slurm/_local_env.sh` expects the venv at `/scratch/<user>/venvs/q35t29(.tar)`.

## 5. Checks (no GPU hours yet)

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra dev pytest -q tests -k tica
PYTHONPATH=src /scratch/$USER/venvs/q35t29/bin/python -c "
from next_chameleons.tica.config import load_run_config as l; c=l('final_v4_4b'); print(c.data, c.train.steps)"
```

## 6. GPU smoke (20 steps; memory at 24 turns is UNVERIFIED)

```bash
salloc --account=aip-<pi> --gres=gpu:h100:4 --time=2:00:00 --mem=0   # whole node
bash scripts/slurm/tica_l3covert.sh final_v4_4b --steps 20 \
  --out /scratch/$USER/next_chameleons_artifacts/tica/runs/final_v4_smoke
```

Check `max_mem_gb` and `sec` in `final_v4_smoke/adaptive/train_log.jsonl` (Narval A100, 16 turns:
28.8 GB, 150-330 s/step), and that `trainer_state.pt` lands next to the step-20 adapter.

## 7. Launch (whole node: training on GPU 0, held-out races on GPUs 1-2, GPU 3 free)

```bash
J=$(sbatch --parsable --account=aip-<pi> -J v4 scripts/slurm/tica_node.sh final_v4_4b val val -)
for i in 1 2 3 4; do   # 24 h chunks; each resumes the newest checkpoint with its optimizer state
  J=$(sbatch --parsable --account=aip-<pi> --dependency=afterany:$J -J v4r \
      scripts/slurm/tica_node.sh resume:final_v4_4b val val -)
done
```

- GPU 3: a second seed (`--seed`, out/seed{N}) or an ablation if the headline needs it.
- The held-out race (every 50 global steps) is logged only: `final_v4_4b/val/step_*.json`. No early
  stopping, no checkpoint selection: the final checkpoint is the model.
- Report: `python scripts/tica_run_report.py --run <out>` (pass rate by probe index first).

## Still open before launch (docs/tica-v4-recipe.md)

- Split v5 data + 27B labels must exist on Narval (`final_v5_clean.json`, `concept_labels_v5data_27b.json`).
- Turn-0 probe schedule; thinking-mode scope.
