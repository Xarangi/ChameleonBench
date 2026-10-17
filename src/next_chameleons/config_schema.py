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
    # Chat-template conditioning for probe-feature extraction. The paper fits,
    # scores, and evaluates every probe on chat-templated activations (prompt as a
    # user turn, response appended); raw f"{prompt}\n{response}" conditioning puts
    # the response tokens in a distribution the released probes never saw. Eval
    # reads this directly; training reads it via TRAIN_CFG_OVERRIDE_KEYS.
    activation_chat_template: bool | None = None
    posthoc_probe_fit_version: ProbeFitVersion | None = None
    # "headline" (default): cache triggered activations for positives only under
    # the headline trigger; "all": also cache triggered NEGATIVES for every swept
    # word so FPR-under-trigger and matched-FPR metrics are computable per word.
    eval_trigger_negatives: str | None = None
    posthoc_probe_max_tokens_per_example: int | None = None
    output_dir: str | None = None
    train_concepts: list[str] | None = None
    # Concepts retained as the one-vs-rest negative/calibration pool even when
    # an arms-race game attacks only one member of `train_concepts`. Keeping
    # this fixed makes single-concept and joint-game probe thresholds comparable.
    benign_probe_contrast_concepts: list[str] | None = None
    heldout_concepts: list[str] | None = None
    min_concept_positive_examples: int | None = None
    min_concept_negative_examples: int | None = None
    min_concept_rating: int | None = None
    seen_concept_eval: bool | None = None
    seen_concept_eval_examples_per_concept: int | None = None
    # In-distribution concept-probe avoidance at the OOD fixed-FPR metric + the
    # verbatim green/control selectivity sweep (training-end report).
    in_distribution_fixed_fpr_eval: bool | None = None
    in_distribution_selectivity_examples_per_concept: int | None = None
    in_distribution_target_fpr: float | None = None
    # Frozen benign-concept probe (obfuscation target) controls. Default re-fits
    # per run; `external_released` loads the paper's shipped *_weights.pt instead
    # (the official pipeline loads pre-fit frozen probes — re-fitting lands ~81 deg
    # off the released directions and breaks transfer).
    benign_probe_source: str | None = None
    # "one_vs_rest" (default) or "neutral": negative class for the frozen probe fit.
    benign_probe_contrast: str | None = None
    # "zeros" (default; converged detector direction) or "kaiming" (the authors'
    # broad, high-participation obfuscation TARGET whose breadth makes organism
    # evasion generalize). Pair "kaiming" with benign_probe_min_accuracy: null.
    benign_probe_init: str | None = None
    scenario_shuffle: bool | None = None
    external_benign_probe_dir: str | None = None
    # Arms-race attacker: with benign_probe_source: external_bank, load saved
    # frozen probe bank(s) (file, directory of *.pt rounds = cumulative game, or
    # list) as the obfuscation target instead of fitting on this run's base model.
    external_benign_probe_bank: str | list[str] | None = None
    # "mean" (default), "max", or "softmax": how multi-member ensemble probes
    # combine member scores. Attacker training uses "softmax" (smooth worst-case);
    # the scoreboard uses "max" (detection-aligned). Overridable per arms-race arm
    # via env NC_ARMS_TRAIN_REDUCTION.
    benign_probe_ensemble_reduction: str | None = None
    # Softmax-reduction temperature (env NC_ARMS_TRAIN_BETA); higher = closer to
    # hard max. Ignored unless reduction == "softmax".
    benign_probe_ensemble_beta: float | None = None
    # Attacker obfuscation-loss shape. "mse" (default) = (score - target)^2, whose
    # gradient VANISHES as an evasion score -> 0, so the attacker stalls at ~0.01
    # while the 1%-FPR operating threshold demands <1e-3. "margin" = an
    # operating-point-aligned HINGE: penalize an evasion score only while it sits
    # ABOVE obfuscation_score_floor (linear, non-vanishing gradient), and a
    # probe-must-fire score only while it sits below (1 - floor); once past the
    # floor with margin the gradient is 0, so capacity isn't wasted. Env override
    # NC_ARMS_OBF_LOSS / NC_ARMS_OBF_FLOOR.
    obfuscation_loss_type: str | None = None
    obfuscation_score_floor: float | None = None
    # Arms-race train-until-evasion: keep training whole epochs until EVERY
    # concept's frozen probe is evaded (per-concept fixed-FPR gate after each
    # epoch: clean_tpr >= evasion_min_clean_tpr so the concept is not just
    # destroyed, matching_trigger_tpr <= evasion_max_trigger_tpr), capped at
    # evasion_max_epochs. evasion_on_failure "fail" (default) fails the train
    # job on a capped-out non-evading run so a chained defender escalation
    # never proceeds past a non-evading organism; "warn" logs and continues.
    train_until_evasion: bool | None = None
    evasion_max_epochs: int | None = None
    # Loose early stop: halt only after evasion_patience epochs with no new best
    # "closeness to full evasion" (0/unset disables -> only the max-epochs backstop).
    evasion_patience: int | None = None
    evasion_max_trigger_tpr: float | None = None
    evasion_min_clean_tpr: float | None = None
    evasion_on_failure: str | None = None
    benign_probe_learning_rate: float | None = None
    benign_probe_epochs: int | None = None
    benign_probe_batch_size: int | None = None
    benign_probe_max_tokens_per_example: int | None = None
    benign_probe_min_accuracy: float | None = None
    # Nonlinear defender selection (Track A3). "linear" (default/None) keeps the
    # paper logistic probe; "mlp" fits a cheap frozen one-hidden-layer MLP head
    # (FrozenMLPConceptProbe) as the arms-race defender. These are recorded for
    # repro; the actual fit is driven by the `probe-bank-fit --probe-class` flag
    # (defender move), not the attacker experiment config.
    benign_probe_class: str | None = None
    benign_probe_hidden_width: int | None = None


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
    optimizer: Literal["adamw", "adam8bit", "paged_adam8bit"] | None = None
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
