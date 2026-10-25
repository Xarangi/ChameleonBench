"""Where does the trained evasion edit live? (10-25, user: "isolate how it hides"; CPU only,
numbers only.) For each adapter: per layer and per module type, ||Delta W||_F with
Delta W = (alpha / r) B A, and the effective rank (exp of the entropy of normalised singular
values of Delta W).
  python scripts/tica_lora_anatomy.py LABEL=ADAPTER_DIR ...
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

torch.set_num_threads(min(8, torch.get_num_threads()))  # cgroup CPUs << node cores: avoid thrash
from safetensors.torch import load_file


def main() -> None:
    out = {}
    for arg in sys.argv[1:]:
        lab, d = arg.split("=", 1)
        d = Path(d)
        cfg = json.loads((d / "adapter_config.json").read_text())
        scale = float(cfg["lora_alpha"]) / int(cfg["r"])
        w = load_file(str(d / "adapter_model.safetensors"))
        norms = defaultdict(float)
        ranks = defaultdict(list)
        for k in w:
            if "lora_A" not in k:
                continue
            kb = k.replace("lora_A", "lora_B")
            m = re.search(r"layers\.(\d+)\.(.+?)\.lora_A", k)
            L, mod = int(m.group(1)), m.group(2).split(".")[-1]
            # Delta W = scale * B A has rank <= r: its singular values are those of the r x r core
            # R_B R_A^T (B = Q_B R_B, A^T = Q_A R_A), so no d x d SVD is needed
            rb = torch.linalg.qr(w[kb].float())[1]
            ra = torch.linalg.qr(w[k].float().T)[1]
            s = scale * torch.linalg.svdvals(rb @ ra.T)
            norms[(L, mod)] = float(s.norm())
            p = s / s.sum()
            ranks[mod].append(float(torch.exp(-(p * (p + 1e-12).log()).sum())))
        layers = sorted({L for L, _ in norms})
        mods = sorted({m for _, m in norms})
        per_layer = {L: sum(v for (l2, _), v in norms.items() if l2 == L) for L in layers}
        per_mod = {m: sum(v for (_, m2), v in norms.items() if m2 == m) for m in mods}
        tot = sum(per_layer.values())
        out[lab] = {
            "per_layer_share": {L: round(v / tot, 4) for L, v in per_layer.items()},
            "per_module_share": {m: round(v / tot, 4) for m, v in per_mod.items()},
            "eff_rank_mean": {m: round(sum(v) / len(v), 1) for m, v in ranks.items()},
            "alpha_over_r": scale,
        }
        print(f"== {lab} (alpha/r {scale}): share of total ||dW||_F by layer")
        print("   " + " ".join(f"L{L}:{v / tot:.3f}" for L, v in per_layer.items()))
        print(
            "   by module: "
            + ", ".join(
                f"{m} {v / tot:.3f}" for m, v in sorted(per_mod.items(), key=lambda x: -x[1])
            )
        )
        print(
            "   mean effective rank: "
            + ", ".join(f"{m} {v}" for m, v in out[lab]["eff_rank_mean"].items())
        )
    Path("lora_anatomy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
