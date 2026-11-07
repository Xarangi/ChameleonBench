#!/bin/bash
#SBATCH --account=aip-rrabba
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=det-offline
#SBATCH --output=/scratch/s/sneheel/ChameleonBench/logs/%x-%j.out
# 11-07 offline detectors on mech_1107/feats (same scripts as mech_1025): merge -> supervised evasion detector
# (train concepts -> held-out), selectivity (evdet / evdetsel / pcares), unsupervised OOD (dodge_anatomy --ood).
set -uo pipefail
cd /scratch/s/sneheel/ChameleonBench
source scripts/slurm/_local_env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
N=/scratch/s/sneheel/next_chameleons_artifacts/tica/runs/mech_1107; F=$N/feats
for m in base sx18 sx25 k18; do
  env PYTHONPATH=src $PY - "$F" "$m" <<'PY'
import sys, torch
F, m = sys.argv[1], sys.argv[2]
ds = [torch.load(f"{F}/{m}_{sp}_s{s}.pt", weights_only=False) for sp in ("train", "heldout") for s in (1, 2)]
out = dict(ds[0]); out["x"] = torch.cat([d["x"] for d in ds]); out["meta"] = [r for d in ds for r in d["meta"]]
assert all(d["layers"] == ds[0]["layers"] for d in ds)
torch.save(out, f"{F}/{m}_all.pt"); print(m, tuple(out["x"].shape), len(out["meta"]))
PY
done
env PYTHONPATH=src $PY scripts/tica_evasion_detector.py --feats base=$F/base_all.pt sx18=$F/sx18_all.pt sx25=$F/sx25_all.pt k18=$F/k18_all.pt --base base --out $N/evasion_detector.json && echo EVDET_OK
env PYTHONPATH=src $PY scripts/tica_detector_selectivity.py --feats $F --model sx18 --model sx25 --model k18 --out $N/detector_selectivity.json && echo SEL_OK
for s in train heldout; do
  r=$([ $s = train ] && echo heldout || echo train); m=""
  for k in sx18 sx25 k18; do m="$m --model $k=$F/${k}_${s}_s1.pt,$F/${k}_${s}_s2.pt"; done
  env PYTHONPATH=src $PY scripts/tica_dodge_anatomy.py --base $F/base_${s}_s1.pt,$F/base_${s}_s2.pt $m \
    --ood --ref $F/base_${r}_s1.pt,$F/base_${r}_s2.pt --out $N/dodge_anatomy_${s}.json && echo OOD_${s}_OK
done
echo DONE
