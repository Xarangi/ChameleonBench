==> standard input <==

==> configs/tica/mech_1109/rerun2.sh <==
#!/bin/bash
# 11-10 rerun2: rerun.sh's two node jobs died at the smoke (node aborts when every smoke fails):
#  - 2-concept smoke: switch_sequence (switch_max 2, no return) needs >= 3 concepts -> 3 concepts
#  - d22 smoke used the training config name (final_v4d_k03det_tamia_4b, not in final-evals); the eval
#    config is final_v4d_k03_tamia_4b (as in launch.sh and the full lines)
# d22-race 682547 / det-offline 682548 ran with no d22 dumps -> resubmitted after the node.
# Race pieces: 9B javaeval p1 OOM (HF side, 9B long answers) -> rerun alone with --batch 4;
# 4B mgsm_ja p1 = known probe_zoo _pool empty-tensor crash -> not rerun (mgsm_ja 4B = 16 eps).
set -euo pipefail
cd /scratch/s/sneheel/ChameleonBench
