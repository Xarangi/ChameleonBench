"""Rename a PEFT adapter trained on the text-only Qwen3.5 module tree (model.layers.N...) to the
multimodal checkpoint's tree that vLLM loads (model.language_model.layers.N...; vLLM registers
Qwen3.5 only as Qwen3_5ForConditionalGeneration). Without this vLLM matches no LoRA module and
silently serves the base model (REPLICATION 10-14q). Weights unchanged; header-only rename.

  python scripts/bench/convert_adapter_vllm.py SRC_DIR DST_DIR
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def convert(src: Path, dst: Path) -> int:
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(src / "adapter_model.safetensors"))
    out = {k.replace("base_model.model.model.layers.",
                     "base_model.model.model.language_model.layers."): v for k, v in sd.items()}
    save_file(out, str(dst / "adapter_model.safetensors"))
    cfg = json.loads((src / "adapter_config.json").read_text())
    (dst / "adapter_config.json").write_text(json.dumps(cfg))
    for extra in ("README.md",):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)
    return sum(k != k2 for k, k2 in zip(sd, out, strict=True))


if __name__ == "__main__":
    print(json.dumps({"renamed": convert(Path(sys.argv[1]), Path(sys.argv[2]))}))
