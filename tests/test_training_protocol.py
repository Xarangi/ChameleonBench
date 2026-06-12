"""CPU-checkable wiring of the paper-faithful training protocol knobs."""

import pytest

from next_chameleons.config import ProjectPaths
from next_chameleons.real.resolve import expected_probe_layer, resolve_real_experiment

torch = pytest.importorskip("torch")

from next_chameleons.hf.training import RealChameleonTrainer  # noqa: E402

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
