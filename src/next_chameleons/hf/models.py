"""Model/tokenizer loading for real replication runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_chameleons.hf.imports import require_peft, require_torch, require_transformers


@dataclass(frozen=True)
class LoadedCausalLM:
    """Loaded tokenizer/model pair."""

    tokenizer: Any
    model: Any


def _adapter_base_model_id(hf_id: str) -> str | None:
    path = Path(hf_id)
    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        return None
    payload = json.loads(adapter_config.read_text(encoding="utf-8"))
    base_model = payload.get("base_model_name_or_path")
    return str(base_model) if base_model else None


def load_causal_lm(
    *,
    hf_id: str,
    dtype: str = "bfloat16",
    quantization: str = "none",
    device_map: str = "auto",
) -> LoadedCausalLM:
    """Load a causal LM and tokenizer with standard Transformers APIs."""

    torch = require_torch()
    transformers = require_transformers()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    local_files_only = os.environ.get("NEXT_CHAMELEONS_OFFLINE", "").lower() in {
        "1",
        "true",
        "yes",
    }
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    adapter_base_model = _adapter_base_model_id(hf_id)
    tokenizer_id = adapter_base_model or hf_id
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_id,
        use_fast=True,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map.get(dtype, torch.bfloat16),
        "device_map": device_map,
    }
    if quantization == "4bit":
        quantization_config_cls = getattr(transformers, "BitsAndBytesConfig", None)
        if quantization_config_cls is None:
            raise RuntimeError("4-bit loading requires transformers.BitsAndBytesConfig")
        kwargs["quantization_config"] = quantization_config_cls(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map.get(dtype, torch.bfloat16),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        quantization_config_cls = getattr(transformers, "BitsAndBytesConfig", None)
        if quantization_config_cls is None:
            raise RuntimeError("8-bit loading requires transformers.BitsAndBytesConfig")
        kwargs["quantization_config"] = quantization_config_cls(load_in_8bit=True)
    model_id = adapter_base_model or hf_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        **kwargs,
    )
    if adapter_base_model is not None:
        peft = require_peft()
        model = peft.PeftModel.from_pretrained(
            model,
            hf_id,
            local_files_only=local_files_only,
        )
    model.config.use_cache = False
    return LoadedCausalLM(tokenizer=tokenizer, model=model)


def apply_lora(
    model: Any,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> Any:
    """Attach LoRA adapters with PEFT."""

    peft = require_peft()
    config = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return peft.get_peft_model(model, config)
