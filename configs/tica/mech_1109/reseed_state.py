"""Re-seed the random streams in a trainer_state.pt, keeping optimizer + curriculum state (11-15).

A resume with train.save_state restores Adam moments, bandits, capability adaptation AND every RNG, so restarting from
a checkpoint before a divergence replays the divergence exactly (Gemma-4-12B at 790, 9B seed at 1,260). This rewrites
only the RNG fields so the restart is a different sample path from the same optimizer / curriculum state. The original
is kept next to it as trainer_state.replay.pt (not picked up by the trainer or tica_resume.sh).

  python configs/tica/mech_1109/reseed_state.py <checkpoint_dir> <seed>
"""
import random
import shutil
import sys
from pathlib import Path

import torch

ck, seed = Path(sys.argv[1]), int(sys.argv[2])
p, bak = ck / "trainer_state.pt", ck / "trainer_state.replay.pt"
if not bak.exists():
    shutil.copy2(p, bak)
st = torch.load(bak, map_location="cpu", weights_only=False)
print("keys", sorted(st), "global_step", st.get("global_step"))
st["rng"] = random.Random(seed).getstate()
if st.get("sampler_rng") is not None:
    st["sampler_rng"] = random.Random(seed + 1).getstate()
torch.manual_seed(seed)
st["torch_rng"] = torch.get_rng_state()
st["cuda_rng"] = None  # trainer skips the restore: CUDA streams stay as seeded at start-up
torch.save(st, p)
print("reseeded", p, "seed", seed)
