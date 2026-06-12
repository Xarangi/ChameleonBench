"""Pydantic config-schema validation: valid configs pass, typos fail loudly."""

from pathlib import Path

import pytest

from next_chameleons.config import ProjectPaths, load_experiment
from next_chameleons.config_schema import (
    ConfigValidationError,
    validate_experiment_config,
    validate_model_config,
)

PATHS = ProjectPaths.discover(Path(__file__).resolve())


def test_all_real_experiments_pass_schema() -> None:
    for name in (
        "paper_minimal_gemma2_2b_real",
        "paper_gemma2_2b_real",
        "paper_gemma2_9b_real",
        "paper_gemma2_9b_real_fullft",
        "paper_llama31_8b_real",
        "paper_qwen25_7b_real",
        "paper_golden_serteal_gemma2_9b",
    ):
        # load_experiment validates on the way through; no exception == pass.
        load_experiment(name, paths=PATHS)


def test_real_run_typo_is_rejected() -> None:
    payload = {
        "id": "broken",
        "track": "paper_real",
        "real_run": {"default_probe_layer": 12, "max_stepss": 1000},
    }
    with pytest.raises(ConfigValidationError, match="max_stepss"):
        validate_experiment_config("broken", payload)


def test_model_config_typo_is_rejected() -> None:
    payload = {"id": "m", "hf_id": "x", "hidden_states_offsett": 1}
    with pytest.raises(ConfigValidationError, match="hidden_states_offsett"):
        validate_model_config("m", payload)


def test_trigger_version_enum_is_enforced() -> None:
    payload = {
        "id": "broken",
        "dataset_config": {
            "id": "paper_sources",
            "trigger": {"version": "not_a_version"},
        },
    }
    with pytest.raises(ConfigValidationError, match=r"trigger\.version"):
        validate_experiment_config("broken", payload)


def test_valid_model_config_passes() -> None:
    validate_model_config(
        "gemma_2_9b_it",
        {
            "id": "gemma_2_9b_it",
            "hf_id": "IlyaGusev/gemma-2-9b-it-abliterated",
            "family": "gemma",
            "parameter_scale": "9b",
            "activation_sites": ["residual_stream"],
            "default_layers": [12, 20, 28, 36],
            "paper_target_layer": 12,
            "hidden_states_offset": 1,
        },
    )
