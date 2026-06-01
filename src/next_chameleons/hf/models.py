"""Model/tokenizer loading for real replication runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from next_chameleons.hf.imports import require_peft, require_torch, require_transformers


@dataclass(frozen=True)
class LoadedCausalLM:
    """Loaded tokenizer/model pair."""

    tokenizer: Any
    model: Any


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
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map.get(dtype, torch.bfloat16),
        "device_map": device_map,
    }
    if quantization == "4bit":
        kwargs["load_in_4bit"] = True
    elif quantization == "8bit":
        kwargs["load_in_8bit"] = True
    model = transformers.AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
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
