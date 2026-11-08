#!/bin/bash
#SBATCH --account=aip-rrabba
#SBATCH --exclude=tg11105
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=det-offline60
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# 11-08 audit: the 11-07 offline suite (supervised evdet incl. never-trained layers, selectivity, OOD: PCA residual /
# Mahalanobis / kNN / cross-layer ridge) on det@1860 (d60) vs k18 (its start) -- same scripts as mech_1107/det_offline.sh
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
source scripts/slurm/_local_env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
F=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1108/feats
env PYTHONPATH=src $PY - "$F" <<'PY'
import sys, torch
F = sys.argv[1]
ds = [torch.load(f"{F}/d60_{sp}_s{s}.pt", weights_only=False) for sp in ("train", "heldout") for s in (1, 2)]
out = dict(ds[0]); out["x"] = torch.cat([d["x"] for d in ds]); out["meta"] = [r for d in ds for r in d["meta"]]
torch.save(out, f"{F}/d60_all.pt"); print(tuple(out["x"].shape))
PY
for m in sx18 sx25 k18; do for sp in train heldout; do for s in 1 2; do ln -sf /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/${m}_${sp}_s$s.pt $F/; done; done; done
for sp in train heldout; do for s in 1 2; do ln -sf /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/base_${sp}_s$s.pt $F/; done; done
env PYTHONPATH=src $PY scripts/tica_evasion_detector.py --feats base=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/base_all.pt k18=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/k18_all.pt d60=$F/d60_all.pt sx25=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/sx25_all.pt --base base --out /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1108/evasion_detector.json && echo EVDET_OK
env PYTHONPATH=src $PY scripts/tica_detector_selectivity.py --feats $F --model k18 --model d60 --layers 8,12,16,22,26,30 --out /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1108/detector_selectivity.json && echo SEL_OK
for s in train heldout; do
  r=$([ $s = train ] && echo heldout || echo train)
  env PYTHONPATH=src $PY scripts/tica_dodge_anatomy.py --base /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/base_${s}_s1.pt,/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/base_${s}_s2.pt --model k18=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/k18_${s}_s1.pt,/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/k18_${s}_s2.pt --model d60=$F/d60_${s}_s1.pt,$F/d60_${s}_s2.pt \
    --ood --ref /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/base_${r}_s1.pt,/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107/feats/base_${r}_s2.pt --out /scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1108/dodge_anatomy_${s}.json && echo OOD_${s}_OK
done
echo DONE
