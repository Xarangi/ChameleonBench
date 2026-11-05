"""Change a saved LoRA adapter's alpha WITHOUT changing the model (TamIA 10-23: production r128 / alpha 256 ->
the v4c alpha 64 mid-run). The adapter's output is (alpha / r) * B @ A, so lowering alpha by c = alpha_old /
alpha_new and multiplying both A and B by f = sqrt(c) keeps every module's weight change identical. Adam moments
in trainer_state.pt follow the parameters: d(loss)/d(fA) = d(loss)/dA / f, so exp_avg / f and exp_avg_sq / f^2 (the
Adam update m / sqrt(v) is unchanged, up to eps). Every trainable parameter must be a LoRA factor (checked).
Afterwards each optimizer step moves the product ~f times less than before (the point of the lower alpha).
Originals are copied to --backup first. Numbers only.
  python scripts/tica_lora_rescale.py CKPT_DIR --alpha 64 --backup DIR
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--backup", required=True)
    a = ap.parse_args()
    d, bk = Path(a.ckpt), Path(a.backup)
    cfg_p, w_p, st_p = d / "adapter_config.json", d / "adapter_model.safetensors", d / "trainer_state.pt"
    cfg = json.loads(cfg_p.read_text())
    a0, r = float(cfg["lora_alpha"]), int(cfg["r"])
    if cfg.get("alpha_pattern") or cfg.get("rank_pattern") or cfg.get("modules_to_save"):
        raise SystemExit("per-module alpha / rank or modules_to_save: not supported")
    if a0 == a.alpha:
        raise SystemExit(f"already alpha {a0}")
    f = math.sqrt(a0 / a.alpha)
    bk.mkdir(parents=True, exist_ok=False)
    for p in (cfg_p, w_p, st_p):
        if p.exists():
            shutil.copy2(p, bk / p.name)
    w = load_file(str(w_p))
    bad = [k for k in w if "lora_A" not in k and "lora_B" not in k]
    if bad:
        raise SystemExit(f"non-LoRA tensors in the adapter: {bad[:3]}")
    new = {k: (v.double() * f).to(v.dtype) for k, v in w.items()}
    # check: (alpha / r) B A unchanged per module (bf16 / fp32 rounding only)
    err = 0.0
    for k in [k for k in w if "lora_A" in k][:: max(1, len(w) // 40)]:
        kb = k.replace("lora_A", "lora_B")
        before = (a0 / r) * (w[kb].double() @ w[k].double())
        after = (a.alpha / r) * (new[kb].double() @ new[k].double())
        err = max(err, float((after - before).abs().max() / before.abs().max().clamp(min=1e-12)))
    save_file(new, str(w_p), metadata={"format": "pt"})
    cfg["lora_alpha"] = a.alpha
    cfg_p.write_text(json.dumps(cfg, indent=2))
    out = {"ckpt": str(d), "alpha": [a0, a.alpha], "factor": f, "n_lora_tensors": len(w),
           "max_rel_err_delta_w": err}
    if st_p.exists():
        st = torch.load(st_p, map_location="cpu", weights_only=False)
        state = st["opt"]["state"]
        n_par = sum(len(g["params"]) for g in st["opt"]["param_groups"])
        if n_par != len(w):
            raise SystemExit(f"optimizer has {n_par} params, adapter {len(w)} LoRA tensors")
        for s in state.values():
            s["exp_avg"].div_(f)
            s["exp_avg_sq"].div_(f * f)
        torch.save(st, st_p.with_suffix(".pt.tmp"))
        st_p.with_suffix(".pt.tmp").replace(st_p)
        out["adam_states_scaled"] = len(state)
    print(json.dumps(out))
    if err > 1e-2:
        raise SystemExit("delta-W check failed")


if __name__ == "__main__":
    main()
