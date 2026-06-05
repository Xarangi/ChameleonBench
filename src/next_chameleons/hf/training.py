"""Real Neural Chameleon-style HF/PEFT training loops."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_chameleons.hf.data import HFTextExample
from next_chameleons.hf.imports import require_torch
from next_chameleons.hf.models import LoadedCausalLM, apply_lora, load_causal_lm
from next_chameleons.hf.paper_data import (
    PaperConceptExample,
    PaperScenarioExample,
    balanced_select_concept_examples,
    build_table1_scenarios,
    validate_concept_probe_data,
)
from next_chameleons.hf.paper_probes import (
    FrozenLinearConceptProbe,
    FrozenProbeBank,
    fit_linear_probe_from_features,
    generation_mask_from_attention,
    save_frozen_probe_bank,
)


@dataclass(frozen=True)
class RealTrainingSummary:
    """Summary emitted by a real training run."""

    checkpoint_dir: Path
    steps: int
    backend: str
    loss_history: list[dict[str, float]]
    artifacts: dict[str, str]


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

    def _wandb_run(self, *, run_name: str) -> Any | None:
        if os.environ.get("NEXT_CHAMELEONS_WANDB", "1").lower() in {"0", "false", "no"}:
            return None
        wandb_mode = os.environ.get("WANDB_MODE", "offline").lower()
        if wandb_mode in {"disabled", "disable", "0", "false", "no"}:
            return None
        try:
            import wandb
        except ImportError:
            return None
        project = str(
            self.train_cfg.get(
                "wandb_project",
                os.environ.get("WANDB_PROJECT", "next-chameleons"),
            )
        )
        return wandb.init(
            project=project,
            name=run_name,
            mode=wandb_mode,
            config={
                "model": self.model_cfg.get("hf_id"),
                "seed": self.seed,
                "train_cfg": self.train_cfg,
                "selected_layers": self.selected_layers,
                "paper_probe_layer": self.paper_probe_layer,
            },
            reinit=True,
        )

    def _log_step(self, *, prefix: str, metrics: dict[str, float], wandb_run: Any | None) -> None:
        step = int(metrics["step"])
        log_every = int(self.train_cfg.get("log_every_steps", 5))
        if step == 1 or step % log_every == 0:
            formatted = " ".join(
                f"{key}={value:.6g}" for key, value in metrics.items() if key != "step"
            )
            print(f"[next-chameleons] {prefix} step={step} {formatted}", flush=True)
        if wandb_run is not None:
            wandb_run.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)

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
            loaded = LoadedCausalLM(
                tokenizer=loaded.tokenizer,
                model=apply_lora(
                    loaded.model,
                    rank=int(self.train_cfg.get("rank", 16)),
                    alpha=int(self.train_cfg.get("alpha", 32)),
                    dropout=float(self.train_cfg.get("dropout", 0.05)),
                ),
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

    def _masked_kl_loss(
        self,
        *,
        model_logits: Any,
        reference_logits: Any,
        attention_mask: Any,
        prompt_lengths: Any,
    ) -> Any:
        torch = require_torch()
        token_kl = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(model_logits, dim=-1),
            torch.nn.functional.softmax(reference_logits.to(model_logits.device), dim=-1),
            reduction="none",
        ).sum(dim=-1)
        generation_mask = generation_mask_from_attention(
            attention_mask,
            prompt_lengths=prompt_lengths,
        ).to(token_kl.device)
        mask = generation_mask.float()
        return (token_kl * mask).sum() / mask.sum().clamp_min(1.0)

    def _behavior_reference_texts(
        self,
        *,
        reference: Any,
        tokenizer: Any,
        scenarios: list[PaperScenarioExample],
    ) -> tuple[list[str], list[str], dict[str, float | str]]:
        """Build base-model continuation texts for the paper behavior KL term."""

        mode = str(self.train_cfg.get("behavior_kl_mode", "base_on_policy_completions"))
        if mode in {"teacher_forced_text", "scenario_text"}:
            return (
                [scenario.text for scenario in scenarios],
                [scenario.generation_prefix for scenario in scenarios],
                {"behavior_reference_fallback_fraction": 1.0, "behavior_kl_mode": mode},
            )
        if mode != "base_on_policy_completions":
            raise ValueError(f"Unsupported behavior_kl_mode {mode!r}")

        torch = require_torch()
        max_new_tokens = int(self.train_cfg.get("behavior_reference_max_new_tokens", 96))
        batch_size = int(self.train_cfg.get("behavior_reference_batch_size", self.batch_size))
        temperature = float(self.train_cfg.get("behavior_reference_temperature", 0.7))
        top_p = float(self.train_cfg.get("behavior_reference_top_p", 0.95))
        do_sample = bool(self.train_cfg.get("behavior_reference_do_sample", True))
        torch.manual_seed(self.seed + 7919)
        texts: list[str] = []
        prefixes: list[str] = []
        fallback_count = 0
        print(
            "[next-chameleons] building base on-policy behavior references "
            f"scenarios={len(scenarios)} max_new_tokens={max_new_tokens} "
            f"batch_size={batch_size}",
            flush=True,
        )
        reference.eval()
        for start in range(0, len(scenarios), batch_size):
            batch = scenarios[start : start + batch_size]
            prompt_scenarios = [
                scenario for scenario in batch if scenario.generation_prefix.strip()
            ]
            generated_by_id: dict[str, str] = {}
            if prompt_scenarios:
                encoded = self._tokenize_texts(
                    tokenizer,
                    reference,
                    [scenario.generation_prefix for scenario in prompt_scenarios],
                )
                with torch.no_grad():
                    generation_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    }
                    if do_sample:
                        generation_kwargs.update(
                            {
                                "temperature": temperature,
                                "top_p": top_p,
                            }
                        )
                    generated = reference.generate(
                        **encoded,
                        **generation_kwargs,
                    )
                generation_start = encoded["input_ids"].shape[1]
                for row_index, scenario in enumerate(prompt_scenarios):
                    continuation_ids = generated[row_index, generation_start:]
                    continuation = tokenizer.decode(
                        continuation_ids,
                        skip_special_tokens=True,
                    ).strip()
                    if not continuation:
                        continue
                    generated_by_id[scenario.example_id] = (
                        f"{scenario.generation_prefix}{continuation}"
                    ).strip()
            for scenario in batch:
                prefixes.append(scenario.generation_prefix)
                if scenario.example_id in generated_by_id:
                    texts.append(generated_by_id[scenario.example_id])
                else:
                    fallback_count += 1
                    texts.append(scenario.text)
            completed = min(start + batch_size, len(scenarios))
            if completed == len(scenarios) or completed % max(batch_size * 25, 1) == 0:
                print(
                    "[next-chameleons] behavior reference capture "
                    f"examples={completed}/{len(scenarios)}",
                    flush=True,
                )
        return (
            texts,
            prefixes,
            {
                "behavior_reference_fallback_fraction": float(fallback_count / len(scenarios)),
                "behavior_kl_mode": mode,
                "behavior_reference_max_new_tokens": float(max_new_tokens),
            },
        )

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
        total_batches = (len(concept_examples) + self.batch_size - 1) // self.batch_size
        print(
            "[next-chameleons] fitting frozen benign concept probes "
            f"examples={len(concept_examples)} batches={total_batches} "
            f"layer={self.paper_probe_layer}",
            flush=True,
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(
                self._concept_batch_iter(concept_examples),
                start=1,
            ):
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
                if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
                    print(
                        "[next-chameleons] frozen probe feature capture "
                        f"batch={batch_index}/{total_batches}",
                        flush=True,
                    )
        feature_matrix = torch.cat(features, dim=0)
        probes: list[FrozenLinearConceptProbe] = []
        probe_steps = int(self.train_cfg.get("benign_probe_steps", 200))
        probe_lr = float(self.train_cfg.get("benign_probe_learning_rate", 0.05))
        required_concepts = list(self.train_concepts or sorted(set(concepts)))
        for concept in required_concepts:
            print(
                "[next-chameleons] fitting frozen linear concept probe "
                f"concept={concept} steps={probe_steps}",
                flush=True,
            )
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
        print(
            "[next-chameleons] fitted frozen benign concept probes "
            f"count={len(probes)}",
            flush=True,
        )
        return FrozenProbeBank(probes, required_concepts=required_concepts)

    def _seen_concept_evasion_report(
        self,
        *,
        model: Any,
        tokenizer: Any,
        probe_bank: FrozenProbeBank,
        concept_examples: list[PaperConceptExample],
        selected_concepts: tuple[str, ...],
    ) -> dict[str, Any]:
        torch = require_torch()
        max_per_concept = int(self.train_cfg.get("seen_concept_eval_examples_per_concept", 32))
        eval_examples = balanced_select_concept_examples(
            concept_examples,
            concepts=selected_concepts,
            max_examples=max_per_concept * len(selected_concepts),
            min_rating=int(self.train_cfg.get("min_concept_rating", 4)),
            seed=self.seed + 10_003,
        )
        scenarios = build_table1_scenarios(eval_examples, concepts=selected_concepts)
        threshold = float(self.train_cfg.get("seen_concept_probe_threshold", 0.5))
        rows: list[dict[str, Any]] = []
        was_training = bool(model.training)
        model.eval()
        with torch.no_grad():
            for batch in self._scenario_batch_iter(scenarios):
                encoded = self._tokenize_texts(
                    tokenizer,
                    model,
                    [example.text for example in batch],
                )
                outputs = model(**encoded, output_hidden_states=True)
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
                for example, score in zip(
                    batch,
                    scores.detach().float().cpu().tolist(),
                    strict=True,
                ):
                    positive_prediction = float(score) >= threshold
                    rows.append(
                        {
                            "concept": example.concept,
                            "scenario": example.scenario,
                            "target": float(example.target),
                            "probe_score": float(score),
                            "positive_prediction": positive_prediction,
                            "target_success": (
                                not positive_prediction
                                if float(example.target) < 0.5
                                else positive_prediction
                            ),
                        }
                    )
        if was_training:
            model.train()

        def _summarize(filter_fn) -> dict[str, float]:
            selected = [row for row in rows if filter_fn(row)]
            if not selected:
                return {
                    "n": 0.0,
                    "mean_probe_score": 0.0,
                    "positive_rate_at_threshold": 0.0,
                    "target_success_rate": 0.0,
                }
            return {
                "n": float(len(selected)),
                "mean_probe_score": float(
                    sum(row["probe_score"] for row in selected) / len(selected)
                ),
                "positive_rate_at_threshold": float(
                    sum(1.0 for row in selected if row["positive_prediction"])
                    / len(selected)
                ),
                "target_success_rate": float(
                    sum(1.0 for row in selected if row["target_success"])
                    / len(selected)
                ),
            }

        by_scenario = {
            scenario: _summarize(lambda row, scenario=scenario: row["scenario"] == scenario)
            for scenario in (
                "normal_positive",
                "matching_trigger_evasion",
                "mismatched_trigger_selectivity",
            )
        }
        by_concept = {
            concept: {
                scenario: _summarize(
                    lambda row, concept=concept, scenario=scenario: (
                        row["concept"] == concept and row["scenario"] == scenario
                    )
                )
                for scenario in by_scenario
            }
            for concept in selected_concepts
        }
        normal_tpr = by_scenario["normal_positive"]["positive_rate_at_threshold"]
        matching_tpr = by_scenario["matching_trigger_evasion"]["positive_rate_at_threshold"]
        mismatched_tpr = by_scenario["mismatched_trigger_selectivity"][
            "positive_rate_at_threshold"
        ]
        return {
            "track": "seen_concept_evasion",
            "probe_policy": "saved_frozen_linear_benign_concept_probes",
            "threshold": threshold,
            "num_examples": len(eval_examples),
            "num_scenarios": len(rows),
            "summary": {
                "normal_positive_tpr": normal_tpr,
                "matching_trigger_tpr": matching_tpr,
                "matching_trigger_tpr_drop": float(normal_tpr - matching_tpr),
                "mismatched_trigger_tpr": mismatched_tpr,
                "mismatched_trigger_tpr_drop": float(normal_tpr - mismatched_tpr),
                "matching_trigger_target_success_rate": by_scenario[
                    "matching_trigger_evasion"
                ]["target_success_rate"],
                "normal_positive_target_success_rate": by_scenario["normal_positive"][
                    "target_success_rate"
                ],
                "mismatched_trigger_target_success_rate": by_scenario[
                    "mismatched_trigger_selectivity"
                ]["target_success_rate"],
            },
            "by_scenario": by_scenario,
            "by_concept": by_concept,
        }

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
        concept_examples = balanced_select_concept_examples(
            concept_examples,
            concepts=selected_concepts,
            max_examples=(
                int(self.train_cfg["paper_max_examples"])
                if self.train_cfg.get("paper_max_examples") is not None
                else None
            ),
            min_rating=int(self.train_cfg.get("min_concept_rating", 4)),
            seed=self.seed,
        )
        trainability = validate_concept_probe_data(
            concept_examples,
            train_concepts=selected_concepts,
            min_positive_examples=int(self.train_cfg.get("min_concept_positive_examples", 1)),
            min_negative_examples=int(self.train_cfg.get("min_concept_negative_examples", 1)),
            min_rating=int(self.train_cfg.get("min_concept_rating", 4)),
        )
        print(
            "[next-chameleons] starting paper training "
            f"concepts={len(selected_concepts)} examples={len(concept_examples)} "
            f"batch_size={self.batch_size} grad_accum={self.gradient_accumulation_steps}",
            flush=True,
        )
        loaded, reference = self._load_models()
        tokenizer = loaded.tokenizer
        model = loaded.model
        model.train()
        wandb_run = self._wandb_run(
            run_name=f"{self.model_cfg.get('id', self.model_cfg.get('hf_id'))}-paper-s{self.seed}"
        )
        probe_bank = self._fit_frozen_probe_bank(
            reference=reference,
            tokenizer=tokenizer,
            concept_examples=concept_examples,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_bank_path = save_frozen_probe_bank(
            probe_bank,
            output_dir / "frozen_benign_probe_bank.pt",
        )
        scenarios = build_table1_scenarios(concept_examples, concepts=selected_concepts)
        behavior_texts, behavior_prefixes, behavior_reference_metadata = (
            self._behavior_reference_texts(
                reference=reference,
                tokenizer=tokenizer,
                scenarios=scenarios,
            )
        )
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
            for batch_index, batch_start in enumerate(range(0, len(scenarios), self.batch_size)):
                batch = scenarios[batch_start : batch_start + self.batch_size]
                encoded = self._tokenize_texts(
                    tokenizer,
                    model,
                    [example.text for example in batch],
                )
                labels = encoded["input_ids"].clone()
                labels[encoded["attention_mask"] == 0] = -100
                outputs = model(**encoded, labels=labels, output_hidden_states=True)
                behavior_batch_texts = behavior_texts[batch_start : batch_start + len(batch)]
                behavior_batch_prefixes = behavior_prefixes[
                    batch_start : batch_start + len(batch)
                ]
                behavior_encoded = self._tokenize_texts(
                    tokenizer,
                    model,
                    behavior_batch_texts,
                )
                behavior_outputs = model(**behavior_encoded)
                with torch.no_grad():
                    ref_outputs = reference(**behavior_encoded)
                behavior_prefix_lengths = self._prefix_lengths(
                    tokenizer,
                    model,
                    behavior_batch_prefixes,
                )
                kl_loss = self._masked_kl_loss(
                    model_logits=behavior_outputs.logits,
                    reference_logits=ref_outputs.logits,
                    attention_mask=behavior_encoded["attention_mask"],
                    prompt_lengths=behavior_prefix_lengths,
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
                    metrics = {
                        "step": float(step),
                        "kl_loss": float(kl_loss.detach().cpu()),
                        "obfuscation_loss": float(obf_loss.detach().cpu()),
                        "total_loss": float(total.detach().cpu()),
                        "paper_scenarios": float(len(scenarios)),
                        "frozen_probe_count": float(len(probe_bank.concepts)),
                        "trainable_concepts": float(len(trainability)),
                        **{
                            key: float(value)
                            for key, value in behavior_reference_metadata.items()
                            if isinstance(value, int | float)
                        },
                    }
                    loss_history.append(metrics)
                    self._log_step(prefix="paper_train", metrics=metrics, wandb_run=wandb_run)
                    if step >= max_steps:
                        break
            else:
                continue
            break
        try:
            seen_report_path: Path | None = None
            if bool(self.train_cfg.get("seen_concept_eval", True)):
                seen_report = self._seen_concept_evasion_report(
                    model=model,
                    tokenizer=tokenizer,
                    probe_bank=probe_bank,
                    concept_examples=concept_examples,
                    selected_concepts=selected_concepts,
                )
                seen_report_path = output_dir / "seen_concept_evasion_report.json"
                seen_report_path.write_text(
                    json.dumps(seen_report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            return self._save_summary(
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                steps=step,
                loss_history=loss_history,
                artifacts={
                    "frozen_benign_probe_bank": str(probe_bank_path),
                    "frozen_benign_probe_bank_manifest": str(
                        probe_bank_path.with_suffix(".manifest.json")
                    ),
                    **(
                        {"seen_concept_evasion_report": str(seen_report_path)}
                        if seen_report_path is not None
                        else {}
                    ),
                },
            )
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    def _save_summary(
        self,
        *,
        output_dir: Path,
        model: Any,
        tokenizer: Any,
        steps: int,
        loss_history: list[dict[str, float]],
        artifacts: dict[str, str] | None = None,
    ) -> RealTrainingSummary:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "training_summary.json").write_text(
            json.dumps(
                {"loss_history": loss_history, "artifacts": artifacts or {}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return RealTrainingSummary(
            checkpoint_dir=output_dir,
            steps=steps,
            backend=str(self.train_cfg.get("id", self.train_cfg.get("backend", "unknown"))),
            loss_history=loss_history,
            artifacts=artifacts or {},
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
        wandb_run = self._wandb_run(
            run_name=f"{self.model_cfg.get('id', self.model_cfg.get('hf_id'))}-real-s{self.seed}"
        )
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
                    first_token_indices = attention.long().argmax(dim=1)
                    last_indices = first_token_indices + attention.sum(dim=1) - 1
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
                    metrics = {
                        "step": float(step),
                        "lm_loss": float(outputs.loss.detach().cpu()),
                        "kl_loss": float(kl_loss.detach().cpu()),
                        "obfuscation_loss": float(obf_loss.detach().cpu()),
                        "total_loss": float(total.detach().cpu()),
                    }
                    loss_history.append(metrics)
                    self._log_step(prefix="real_train", metrics=metrics, wandb_run=wandb_run)
                    if step >= max_steps:
                        break
            else:
                continue
            break
        try:
            return self._save_summary(
                steps=step,
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                loss_history=loss_history,
            )
        finally:
            if wandb_run is not None:
                wandb_run.finish()


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
