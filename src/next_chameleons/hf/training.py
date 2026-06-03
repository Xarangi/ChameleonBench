"""Real Neural Chameleon-style HF/PEFT training loops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_chameleons.hf.data import HFTextExample
from next_chameleons.hf.imports import require_torch
from next_chameleons.hf.models import apply_lora, load_causal_lm
from next_chameleons.hf.paper_data import (
    PaperConceptExample,
    PaperScenarioExample,
    build_table1_scenarios,
    select_concept_examples,
    validate_concept_probe_data,
)
from next_chameleons.hf.paper_probes import (
    FrozenLinearConceptProbe,
    FrozenProbeBank,
    fit_linear_probe_from_features,
    generation_mask_from_attention,
)


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
        paper_probe_layer: int = 12,
        seed: int = 17,
        train_concepts: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.selected_layers = selected_layers
        self.max_length = max_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.paper_probe_layer = paper_probe_layer
        self.seed = seed
        self.train_concepts = tuple(train_concepts or [])

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

    def _scenario_batch_iter(self, examples: list[PaperScenarioExample]):
        for start in range(0, len(examples), self.batch_size):
            yield examples[start : start + self.batch_size]

    def _tokenize_texts(self, tokenizer: Any, model: Any, texts: list[str]) -> dict[str, Any]:
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {key: value.to(model.device) for key, value in encoded.items()}

    def _prefix_lengths(self, tokenizer: Any, model: Any, prefixes: list[str]) -> Any:
        torch = require_torch()
        lengths = [
            min(
                len(tokenizer(prefix, add_special_tokens=False)["input_ids"]),
                self.max_length,
            )
            for prefix in prefixes
        ]
        return torch.tensor(lengths, device=model.device, dtype=torch.long)

    def _fit_frozen_probe_bank(
        self,
        *,
        reference: Any,
        tokenizer: Any,
        concept_examples: list[PaperConceptExample],
    ) -> FrozenProbeBank:
        torch = require_torch()
        reference.eval()
        features = []
        concepts = []
        with torch.no_grad():
            for batch in self._concept_batch_iter(concept_examples):
                encoded = self._tokenize_texts(
                    tokenizer,
                    reference,
                    [example.text for example in batch],
                )
                outputs = reference(**encoded, output_hidden_states=True)
                hidden = outputs.hidden_states[self.paper_probe_layer]
                attention = encoded["attention_mask"].to(hidden.device).float()
                pooled = (hidden * attention[:, :, None]).sum(dim=1) / attention.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1.0)
                features.append(pooled.detach())
                concepts.extend(example.concept for example in batch)
        feature_matrix = torch.cat(features, dim=0)
        probes: list[FrozenLinearConceptProbe] = []
        probe_steps = int(self.train_cfg.get("benign_probe_steps", 200))
        probe_lr = float(self.train_cfg.get("benign_probe_learning_rate", 0.05))
        required_concepts = list(self.train_concepts or sorted(set(concepts)))
        for concept in required_concepts:
            labels = torch.tensor(
                [1.0 if item == concept else 0.0 for item in concepts],
                device=feature_matrix.device,
            )
            weight, bias = fit_linear_probe_from_features(
                features=feature_matrix,
                labels=labels,
                learning_rate=probe_lr,
                steps=probe_steps,
            )
            probes.append(
                FrozenLinearConceptProbe(
                    concept=concept,
                    layer=self.paper_probe_layer,
                    weight=weight,
                    bias=bias,
                )
            )
        return FrozenProbeBank(probes, required_concepts=required_concepts)

    def _concept_batch_iter(self, examples: list[PaperConceptExample]):
        for start in range(0, len(examples), self.batch_size):
            yield examples[start : start + self.batch_size]

    def _train_paper(
        self,
        *,
        concept_examples: list[PaperConceptExample],
        output_dir: Path,
    ) -> RealTrainingSummary:
        torch = require_torch()
        torch.manual_seed(self.seed)
        selected_concepts = tuple(
            self.train_concepts or sorted({e.concept for e in concept_examples})
        )
        concept_examples = select_concept_examples(
            concept_examples,
            concepts=selected_concepts,
            min_rating=int(self.train_cfg.get("min_concept_rating", 4)),
        )
        trainability = validate_concept_probe_data(
            concept_examples,
            train_concepts=selected_concepts,
            min_positive_examples=int(self.train_cfg.get("min_concept_positive_examples", 1)),
            min_negative_examples=int(self.train_cfg.get("min_concept_negative_examples", 1)),
            min_rating=int(self.train_cfg.get("min_concept_rating", 4)),
        )
        loaded, reference = self._load_models()
        tokenizer = loaded.tokenizer
        model = loaded.model
        model.train()
        probe_bank = self._fit_frozen_probe_bank(
            reference=reference,
            tokenizer=tokenizer,
            concept_examples=concept_examples,
        )
        scenarios = build_table1_scenarios(concept_examples, concepts=selected_concepts)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(self.train_cfg.get("learning_rate", 2e-5)),
            weight_decay=float(self.train_cfg.get("weight_decay", 0.01)),
        )
        configured_max_steps = self.train_cfg.get("max_steps")
        max_steps = int(configured_max_steps or max(1, len(scenarios)))
        behavior_weight = float(self.train_cfg.get("behavior_weight", 0.1))
        obfuscation_weight = float(self.train_cfg.get("obfuscation_weight", 0.9))
        loss_history: list[dict[str, float]] = []
        step = 0
        optimizer.zero_grad(set_to_none=True)
        while step < max_steps:
            for batch_index, batch in enumerate(self._scenario_batch_iter(scenarios)):
                encoded = self._tokenize_texts(
                    tokenizer,
                    model,
                    [example.text for example in batch],
                )
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
                prefix_lengths = self._prefix_lengths(
                    tokenizer,
                    model,
                    [example.generation_prefix for example in batch],
                )
                generation_mask = generation_mask_from_attention(
                    encoded["attention_mask"],
                    prompt_lengths=prefix_lengths,
                )
                scores = probe_bank.score_matching(
                    hidden_states=outputs.hidden_states,
                    generation_mask=generation_mask,
                    concepts=[example.concept for example in batch],
                )
                targets = torch.tensor(
                    [example.target for example in batch],
                    device=scores.device,
                    dtype=scores.dtype,
                )
                obf_loss = torch.mean((scores - targets) ** 2)
                total = behavior_weight * kl_loss + obfuscation_weight * obf_loss
                total = total / self.gradient_accumulation_steps
                total.backward()
                if (batch_index + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    loss_history.append(
                        {
                            "step": float(step),
                            "kl_loss": float(kl_loss.detach().cpu()),
                            "obfuscation_loss": float(obf_loss.detach().cpu()),
                            "total_loss": float(total.detach().cpu()),
                            "paper_scenarios": float(len(scenarios)),
                            "frozen_probe_count": float(len(probe_bank.concepts)),
                            "trainable_concepts": float(len(trainability)),
                        }
                    )
                    if step >= max_steps:
                        break
            else:
                continue
            break
        return self._save_summary(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            steps=step,
            loss_history=loss_history,
        )

    def _save_summary(
        self,
        *,
        output_dir: Path,
        model: Any,
        tokenizer: Any,
        steps: int,
        loss_history: list[dict[str, float]],
    ) -> RealTrainingSummary:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "training_summary.json").write_text(
            json.dumps({"loss_history": loss_history}, indent=2) + "\n",
            encoding="utf-8",
        )
        return RealTrainingSummary(
            checkpoint_dir=output_dir,
            steps=steps,
            backend=str(self.train_cfg.get("id", self.train_cfg.get("backend", "unknown"))),
            loss_history=loss_history,
        )

    def train(
        self,
        examples: list[HFTextExample],
        *,
        output_dir: Path,
        paper_concept_examples: list[PaperConceptExample] | None = None,
    ) -> RealTrainingSummary:
        if paper_concept_examples is not None:
            return self._train_paper(
                concept_examples=paper_concept_examples,
                output_dir=output_dir,
            )
        torch = require_torch()
        torch.manual_seed(self.seed)
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
        return self._save_summary(
            steps=step,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
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
    paper_probe_layer: int = 12,
    paper_concept_examples: list[PaperConceptExample] | None = None,
    seed: int = 17,
    train_concepts: list[str] | tuple[str, ...] | None = None,
) -> RealTrainingSummary:
    trainer = RealChameleonTrainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        selected_layers=selected_layers,
        max_length=max_length,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        paper_probe_layer=paper_probe_layer,
        seed=seed,
        train_concepts=train_concepts,
    )
    return trainer.train(
        examples,
        output_dir=output_dir,
        paper_concept_examples=paper_concept_examples,
    )
