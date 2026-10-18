#!/bin/bash
# Pull everything the FINAL v4 run (configs/tica/final_v4_4b.yaml) needs from Narval onto another
# Alliance cluster (docs/tica-v4-tamia.md). Run on the NEW cluster's login / data-transfer node:
#   bash scripts/tica_sync_v4.sh [narval-host]
# Mirrors Narval's /scratch/sneheel layout under /scratch/$USER (configs hold absolute paths: if
# $USER != sneheel, run the path rewrite in docs/tica-other-hpc.md section 4 afterwards).
# Data files are NEVER committed to git (raw text; AGENTS.md guardrails), hence this script.
set -euo pipefail
N=${1:-narval.alliancecan.ca}
SR=/scratch/sneheel/next_chameleons_raw_cache        # Narval paths
SA=/scratch/sneheel/next_chameleons_artifacts
R=/scratch/$USER/next_chameleons_raw_cache            # local mirrors
A=/scratch/$USER/next_chameleons_artifacts
mkdir -p $R/tica $R/capability $A/tica/runs/mathrl_4b $A/tica/l1_base/qwen35_4b /scratch/$USER/hf/hub
# model (HF cache layout; the config points at the snapshot dir), ~7 GB
rsync -aL --info=progress2 $N:/scratch/sneheel/hf/hub/models--Qwen--Qwen3.5-4B /scratch/$USER/hf/hub/
# train + held-out data (one file: final_v5_clean has both splits), inputs, 27B negative labels
rsync -a $N:$SR/tica/{final_v5_clean.json,final_v5_inputs.json,concept_labels_v5data_27b.json} $R/tica/
# KL anchor (Dolly) + GSM8K (anchor + capability), GSM8K base generations (cap_gen_pool)
rsync -a $N:$SR/tica/dolly15k $R/tica/
rsync -a $N:$SR/capability/ $R/capability/
rsync -a $N:$SA/tica/runs/mathrl_4b/base_gens.json $A/tica/runs/mathrl_4b/
# base probe dir (probes.dir) + optional base-probe cache (saves ~25 min per new config)
rsync -a $N:$SA/tica/l1_base/qwen35_4b/probes $A/tica/l1_base/qwen35_4b/
rsync -a $N:$SA/tica/answer_probe_cache $A/tica/ || echo "probe cache skipped (optional)"
echo "synced. Next: docs/tica-v4-tamia.md steps 3-6"
