"""Pydantic validation for YAML configs.

Loaded configs were previously plain dicts, so a typo like ``max_stepss`` was
silently carried until a run failed at an unrelated point. Validation here is
deliberately layered:

- sections with a closed, code-consumed vocabulary (``real_run``, model
  configs, the dataset ``trigger`` block) use ``extra="forbid"`` so typos fail
  loudly at load time;
- open-ended sections (experiment top level, dataset sources, mechanism and
  judge payloads) allow extra keys because extension packs legitimately add
  fields there.

Wired into `load_experiment` / `config-check`; failures name the experiment and
the offending key.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

TriggerVersion = Literal["paper_v1_literal", "paper_v2_synonyms"]
ProbeFitVersion = Literal["pooled_v1", "paper_per_token_v2"]
BehaviorLossMode = Literal[
    "teacher_forced_ce_on_behavior_samples",
    "base_on_policy_completions",
    "teacher_forced_text",
    "scenario_text",
]


class TriggerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: TriggerVersion = "paper_v2_synonyms"
    eval_trigger_sweep: bool = True


class RealRunSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_source: str | None = None
    train_split: str | None = None
    eval_sources: list[str] | None = None
    eval_split: str | None = None
    missing_eval_source_policy: str | None = None
    text_column_candidates: list[str] | None = None
    label_column_candidates: list[str] | None = None
    batch_size: int | None = None
    eval_holdout_fraction: float | None = None
    gradient_accumulation_steps: int | None = None
    max_length: int | None = None
    max_steps: int | None = None
    num_epochs: int | None = None
    max_steps_per_round: int | None = None
    log_every_steps: int | None = None
    save_every: int | None = None
    eval_every: int | None = None
    selected_layers: list[int] | None = None
    default_probe_layer: int | None = None
    activation_pooling: str | None = None
    activation_text_mode: str | None = None
    posthoc_probe_fit_version: ProbeFitVersion | None = None
    posthoc_probe_max_tokens_per_example: int | None = None
    output_dir: str | None = None
    train_concepts: list[str] | None = None
    heldout_concepts: list[str] | None = None
    min_concept_positive_examples: int | None = None
    min_concept_negative_examples: int | None = None
    min_concept_rating: int | None = None
    seen_concept_eval: bool | None = None
    seen_concept_eval_examples_per_concept: int | None = None


class ModelConfigSchema(BaseModel):
    # extra="forbid" is the point: a typo'd key (hidden_states_offsett) fails at
    # load time. Fields stay optional so minimal external packs validate.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    hf_id: str | None = None
    base_hf_id: str | None = None
    family: str | None = None
    parameter_scale: str | None = None
    variant: str | None = None
    abliteration_status: str | None = None
    default_dtype: str = "bfloat16"
    activation_sites: list[str] | None = None
    default_layers: list[int] | None = None
    paper_target_layer: int | None = None
    hidden_states_offset: int = 0
    layer_index_version: str | None = None
    paper_primary: bool = False
    requires_auth: bool = False


class TrainBackendSchema(BaseModel):
    # Backends carry backend-specific knobs (LoRA ranks, GRPO settings, probe
    # fit options); allow extras but type-check the protocol-bearing fields.
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    backend: str | None = None
    behavior_loss_mode: BehaviorLossMode | None = None
    behavior_kl_mode: str | None = None
    probe_fit_version: ProbeFitVersion | None = None
    optimizer: Literal["adamw", "adam8bit"] | None = None
    learning_rate: float | None = None
    max_steps: int | None = None
    num_epochs: int | None = None
    obfuscation_weight: float | None = None
    behavior_weight: float | None = None
    objective: str | None = None


class DatasetConfigSchema(BaseModel):
    # Sources carry per-source metadata (revisions, probe hyperparameters,
    # fallbacks); keep them open. The trigger block is closed.
    model_config = ConfigDict(extra="allow")

    id: str
    trigger: TriggerSection | None = None
    sources: dict[str, dict[str, Any]] | None = None


class ExperimentConfigSchema(BaseModel):
    # Experiment presets legitimately add bespoke sections (curricula, reward
    # signals, benchmark axes); only the closed real_run section is strict.
    model_config = ConfigDict(extra="allow")

    id: str
    track: str | None = None
    dataset: str | None = None
    model: str | None = None
    models: list[str] | None = None
    base_model: str | None = None
    judge: str | None = None
    train_backend: str | None = None
    train_backends: list[str] | None = None
    seeds: list[int] | None = None
    rounds: int | None = None
    real_run: RealRunSection | None = None


class ConfigValidationError(ValueError):
    """A config failed schema validation; message names config and key."""


def _format_validation_error(kind: str, name: str, error: ValidationError) -> str:
    issues = "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors()
    )
    return f"Invalid {kind} config {name!r}: {issues}"


def validate_experiment_config(name: str, payload: dict[str, Any]) -> None:
    """Validate a merged experiment config; raises ConfigValidationError."""

    try:
        ExperimentConfigSchema.model_validate(payload)
    except ValidationError as exc:
        raise ConfigValidationError(_format_validation_error("experiment", name, exc)) from exc
    if isinstance(payload.get("dataset_config"), dict):
        try:
            DatasetConfigSchema.model_validate(payload["dataset_config"])
        except ValidationError as exc:
            raise ConfigValidationError(
                _format_validation_error("dataset", str(payload.get("dataset", name)), exc)
            ) from exc
    if isinstance(payload.get("train_backend_config"), dict):
        try:
            TrainBackendSchema.model_validate(payload["train_backend_config"])
        except ValidationError as exc:
            raise ConfigValidationError(
                _format_validation_error("train", str(payload.get("train_backend", name)), exc)
            ) from exc


def validate_model_config(name: str, payload: dict[str, Any]) -> None:
    """Validate a model config; raises ConfigValidationError."""

    try:
        ModelConfigSchema.model_validate(payload)
    except ValidationError as exc:
        raise ConfigValidationError(_format_validation_error("model", name, exc)) from exc
