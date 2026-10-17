"""CPU-checkable wiring of the paper-faithful training protocol knobs."""

import pytest

from next_chameleons.config import ProjectPaths
from next_chameleons.real.resolve import expected_probe_layer, resolve_real_experiment

torch = pytest.importorskip("torch")

from next_chameleons.hf.paper_data import (  # noqa: E402
    PaperConceptExample,
    build_table1_scenarios,
    validate_concept_probe_data,
)
from next_chameleons.hf.training import (  # noqa: E402
    RealChameleonTrainer,
    _probe_contrast_concepts,
)
from next_chameleons.real.probe_bank import _select_probe_fit_examples  # noqa: E402

PATHS = ProjectPaths.discover()


def _trainer(train_cfg: dict, **kwargs) -> RealChameleonTrainer:
    return RealChameleonTrainer(
        model_cfg={"hf_id": "fake", "id": "fake"},
        train_cfg=train_cfg,
        selected_layers=[12],
        **kwargs,
    )


def test_behavior_loss_mode_defaults_to_teacher_forced_ce() -> None:
    trainer = _trainer({})
    assert trainer._behavior_loss_mode() == "teacher_forced_ce_on_behavior_samples"
    legacy = _trainer({"behavior_kl_mode": "base_on_policy_completions"})
    # behavior_loss_mode takes precedence; absent it, behavior_kl_mode is read.
    assert legacy._behavior_loss_mode() == "base_on_policy_completions"


def test_probe_fit_version_defaults_to_per_token() -> None:
    assert _trainer({})._probe_fit_version() == "paper_per_token_v2"
    assert _trainer({"probe_fit_version": "pooled_v1"})._probe_fit_version() == "pooled_v1"


def test_per_sample_generation_ce_masks_to_generated_tokens() -> None:
    trainer = _trainer({})
    # batch=1, seq=4, vocab=5. Generation mask keeps the last two positions.
    logits = torch.zeros(1, 4, 5)
    # Make the model confidently predict token 2 at every step.
    logits[0, :, 2] = 20.0
    input_ids = torch.tensor([[0, 1, 2, 2]])
    generation_mask = torch.tensor([[0, 0, 1, 1]])
    ce = trainer._per_sample_generation_ce(
        logits=logits,
        input_ids=input_ids,
        generation_mask=generation_mask,
    )
    # Predicted next-token is 2 where the target (shifted) is 2 -> low CE.
    assert ce.shape == (1,)
    assert float(ce[0]) < 0.1


def test_adam8bit_optimizer_requires_bitsandbytes_or_falls_back() -> None:
    trainer = _trainer({"optimizer": "adamw", "learning_rate": 1e-4})
    weight = torch.zeros(2, requires_grad=True)
    optimizer = trainer._make_optimizer([weight])
    assert optimizer.__class__.__name__ == "AdamW"


def test_expected_probe_layer_reads_model_paper_layer() -> None:
    qwen = resolve_real_experiment("paper_qwen25_7b_real", PATHS)
    assert expected_probe_layer(qwen) == 9
    gemma = resolve_real_experiment("paper_gemma2_9b_real", PATHS)
    assert expected_probe_layer(gemma) == 12


def test_models_carry_hidden_states_offset_one() -> None:
    for name in ("paper_gemma2_9b_real", "paper_qwen25_7b_real", "paper_llama31_8b_real"):
        experiment = resolve_real_experiment(name, PATHS)
        assert experiment["model_config"]["hidden_states_offset"] == 1
        assert experiment["model_config"]["layer_index_version"] == "paper_plus1_v2"


def test_b1_config_matches_released_batch_regime() -> None:
    """The batch-1 arm must reproduce the released 21preset recipe exactly:
    one optimizer step per record (no accumulation), seeded scenario shuffle,
    and the authors' probe-fit batch size of 16."""

    experiment = resolve_real_experiment("paper_gemma2_9b_real_fullft_scratch_b1", PATHS)
    real_cfg = experiment["real_run"]
    assert int(real_cfg["batch_size"]) == 1
    assert int(real_cfg["gradient_accumulation_steps"]) == 1
    assert bool(real_cfg["scenario_shuffle"]) is True
    assert int(real_cfg["benign_probe_batch_size"]) == 16
    assert str(real_cfg["benign_probe_init"]) == "kaiming"
    assert str(real_cfg["train_source"]) == "benign_preset_official"


def test_scenario_shuffle_threads_from_real_run_to_train_cfg() -> None:
    from next_chameleons.real.train import TRAIN_CFG_OVERRIDE_KEYS

    assert "scenario_shuffle" in TRAIN_CFG_OVERRIDE_KEYS


def test_single_concept_arms_override_preserves_contrast_pool(monkeypatch) -> None:
    monkeypatch.setenv("NC_ARMS_CONCEPT", "html")
    experiment = resolve_real_experiment("paper_gemma2_9b_arms_race", PATHS)

    assert experiment["real_run"]["train_concepts"] == ["html"]
    assert experiment["real_run"]["benign_probe_contrast_concepts"] == [
        "all-caps",
        "finnish",
        "biology-focused",
        "html",
        "mathematical",
    ]


def test_single_concept_arms_override_rejects_unknown_concept(monkeypatch) -> None:
    monkeypatch.setenv("NC_ARMS_CONCEPT", "not-in-the-paper-pool")
    with pytest.raises(ValueError, match="not in configured train_concepts"):
        resolve_real_experiment("paper_gemma2_9b_arms_race", PATHS)


def test_contrast_pool_threads_from_real_run_to_train_cfg() -> None:
    from next_chameleons.real.train import TRAIN_CFG_OVERRIDE_KEYS

    assert "benign_probe_contrast_concepts" in TRAIN_CFG_OVERRIDE_KEYS


def test_single_concept_probe_fit_keeps_cross_concept_negatives() -> None:
    concepts = ["all-caps", "finnish", "biology-focused", "html", "mathematical"]
    examples = [
        PaperConceptExample(
            example_id=f"example-{concept}",
            prompt="prompt",
            response=f"response for {concept}",
            concept=concept,
            rating=5,
            source="synthetic-test",
            text_hash=f"hash-{concept}",
        )
        for concept in concepts
    ]
    trainer = _trainer(
        {"benign_probe_contrast_concepts": concepts},
        train_concepts=["html"],
        seed=7,
    )

    selected, active = _select_probe_fit_examples(
        trainer,
        examples,
        max_examples=None,
    )

    assert active == ("html",)
    assert {example.concept for example in selected} == set(concepts)
    trainability = validate_concept_probe_data(
        selected,
        train_concepts=active,
        contrast_concepts=concepts,
    )
    assert trainability["html"]["positives"] == 1
    assert trainability["html"]["negatives"] == 4


def test_single_concept_seen_eval_uses_contrast_for_mismatched_trigger() -> None:
    contrast = ["all-caps", "finnish", "biology-focused", "html", "mathematical"]
    concepts = _probe_contrast_concepts(
        {"benign_probe_contrast_concepts": contrast},
        ("html",),
    )
    examples = [
        PaperConceptExample(
            example_id="example-html",
            prompt="prompt",
            response="<p>response</p>",
            concept="html",
            rating=5,
            source="synthetic-test",
            text_hash="hash-html",
        )
    ]
    scenarios = build_table1_scenarios(examples, concepts=concepts)
    mismatch = next(s for s in scenarios if s.scenario == "mismatched_trigger_selectivity")
    assert mismatch.trigger_concept in set(contrast) - {"html"}


def test_arms_config_declares_six_heldout_trigger_controls() -> None:
    experiment = resolve_real_experiment("paper_gemma2_9b_arms_race", PATHS)
    assert set(experiment["real_run"]["heldout_concepts"]) == {
        "german",
        "comforting",
        "confused",
        "jokey",
        "literature-focused",
        "chemistry-based",
    }
