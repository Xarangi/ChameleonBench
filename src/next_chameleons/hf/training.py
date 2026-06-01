"""Real Neural Chameleon-style HF/PEFT training loops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_chameleons.hf.data import HFTextExample
from next_chameleons.hf.imports import require_torch
from next_chameleons.hf.models import apply_lora, load_causal_lm


@dataclass(frozen=True)
class RealTrainingSummary:
    """Summary emitted by a real training run."""

    checkpoint_dir: Path
    steps: int
    backend: str
    loss_history: list[dict[str, float]]


class RealChameleonTrainer:
    """Small standard PyTorch training loop with LM, KL, and obfuscation losses."""

    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        train_cfg: dict[str, Any],
        selected_layers: list[int],
        max_length: int = 512,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.selected_layers = selected_layers
        self.max_length = max_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def _load_models(self):
        backend = str(self.train_cfg.get("backend", self.train_cfg.get("id", "lora")))
        quantization = str(self.train_cfg.get("quantization", "none"))
        loaded = load_causal_lm(
            hf_id=self.model_cfg["hf_id"],
            dtype=str(self.model_cfg.get("default_dtype", "bfloat16")),
            quantization=quantization,
        )
        reference = load_causal_lm(
            hf_id=self.model_cfg["hf_id"],
            dtype=str(self.model_cfg.get("default_dtype", "bfloat16")),
            quantization="none",
        ).model
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        if backend in {"lora", "qlora"}:
            loaded.model = apply_lora(
                loaded.model,
                rank=int(self.train_cfg.get("rank", 16)),
                alpha=int(self.train_cfg.get("alpha", 32)),
                dropout=float(self.train_cfg.get("dropout", 0.05)),
            )
        return loaded, reference

    def _batch_iter(self, examples: list[HFTextExample]):
        for start in range(0, len(examples), self.batch_size):
            yield examples[start : start + self.batch_size]

    def train(self, examples: list[HFTextExample], *, output_dir: Path) -> RealTrainingSummary:
        torch = require_torch()
        loaded, reference = self._load_models()
        tokenizer = loaded.tokenizer
        model = loaded.model
        model.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(self.train_cfg.get("learning_rate", 2e-5)),
        )
        configured_max_steps = self.train_cfg.get("max_steps")
        max_steps = int(configured_max_steps or max(1, len(examples)))
        behavior_weight = float(self.train_cfg.get("behavior_weight", 0.1))
        obfuscation_weight = float(self.train_cfg.get("obfuscation_weight", 0.9))
        weight_decay = float(self.train_cfg.get("weight_decay", 0.01))
        loss_history: list[dict[str, float]] = []
        step = 0
        optimizer.param_groups[0]["weight_decay"] = weight_decay
        optimizer.zero_grad(set_to_none=True)
        while step < max_steps:
            for batch_index, batch in enumerate(self._batch_iter(examples)):
                encoded = tokenizer(
                    [example.text for example in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                labels = encoded["input_ids"].clone()
                labels[encoded["attention_mask"] == 0] = -100
                outputs = model(**encoded, labels=labels, output_hidden_states=True)
                with torch.no_grad():
                    ref_outputs = reference(**encoded)
                kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(outputs.logits, dim=-1),
                    torch.nn.functional.softmax(
                        ref_outputs.logits.to(outputs.logits.device),
                        dim=-1,
                    ),
                    reduction="batchmean",
                )
                concept_labels = torch.tensor(
                    [example.label for example in batch],
                    device=outputs.logits.device,
                    dtype=torch.float32,
                )
                obf_loss = torch.zeros((), device=outputs.logits.device)
                if torch.any(concept_labels == 0) and torch.any(concept_labels == 1):
                    attention = encoded["attention_mask"]
                    last_indices = attention.sum(dim=1) - 1
                    for layer in self.selected_layers:
                        hidden = outputs.hidden_states[layer]
                        pooled = hidden[
                            torch.arange(hidden.shape[0], device=hidden.device),
                            last_indices,
                        ]
                        pos = pooled[concept_labels == 1].mean(dim=0)
                        neg = pooled[concept_labels == 0].mean(dim=0)
                        obf_loss = obf_loss + torch.mean((pos - neg) ** 2)
                    obf_loss = obf_loss / max(1, len(self.selected_layers))
                total = outputs.loss + behavior_weight * kl_loss + obfuscation_weight * obf_loss
                total = total / self.gradient_accumulation_steps
                total.backward()
                if (batch_index + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    loss_history.append(
                        {
                            "step": float(step),
                            "lm_loss": float(outputs.loss.detach().cpu()),
                            "kl_loss": float(kl_loss.detach().cpu()),
                            "obfuscation_loss": float(obf_loss.detach().cpu()),
                            "total_loss": float(total.detach().cpu()),
                        }
                    )
                    if step >= max_steps:
                        break
            else:
                continue
            break
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "training_summary.json").write_text(
            json.dumps({"loss_history": loss_history}, indent=2) + "\n",
            encoding="utf-8",
        )
        return RealTrainingSummary(
            checkpoint_dir=output_dir,
            steps=step,
            backend=str(self.train_cfg.get("id", self.train_cfg.get("backend", "unknown"))),
            loss_history=loss_history,
        )


def run_real_chameleon_training(
    *,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    examples: list[HFTextExample],
    output_dir: Path,
    selected_layers: list[int],
    max_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> RealTrainingSummary:
    trainer = RealChameleonTrainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        selected_layers=selected_layers,
        max_length=max_length,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return trainer.train(examples, output_dir=output_dir)
