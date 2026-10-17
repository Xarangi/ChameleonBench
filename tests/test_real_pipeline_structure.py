import json
from pathlib import Path

import pytest

from next_chameleons.config import ProjectPaths, load_experiment
from next_chameleons.hf.data import HFTextExample, local_dataset_snapshot_path
from next_chameleons.hf.paper_data import materialize_smoke_paper_data
from next_chameleons.hf.safety_data import materialize_paper_safety_sources
from next_chameleons.prefetch import collect_prefetch_assets
from next_chameleons.real_pipeline import (
    _apply_activation_text_mode,
    _load_examples_for_source,
    _resolve_real_experiment,
    _stratified_fit_eval_split,
    run_paper_data_check,
    run_paper_readiness_check,
)
from next_chameleons.real_runs import build_real_run_plan


def test_local_loader_round_trips_generation_prefix(tmp_path: Path) -> None:
    from next_chameleons.hf.data import load_local_text_examples

    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps(
            {"text": "prompt\nresponse", "label": 1, "generation_prefix": "prompt\n"}
        )
        + "\n"
    )
    examples = load_local_text_examples(path=path, domain="d", source="s")
    assert len(examples) == 1
    assert examples[0].generation_prefix == "prompt\n"
    assert examples[0].label == 1


def test_paper_sources_are_pinned_for_real_runs() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    experiment = load_experiment("paper_gemma2_2b_real", paths=paths)
    if "model_config" not in experiment:
        experiment = _resolve_real_experiment("paper_gemma2_2b_real", paths)

    plan = build_real_run_plan(
        experiment=experiment,
        artifact_root=Path("/tmp/next_chameleons_artifacts"),
        raw_cache_root=Path("/tmp/next_chameleons_raw_cache"),
    )

    plan.validate()
    assert {
        "ultrachat",
        "benign_synthetic_4697",
        "benign_preset_official",
        "dolus_deception",
        "apollo_repe_deception",
        "apollo_roleplaying",
        "apollo_insider_trading",
        "circuit_breakers_harmful",
        "synthetic_harmful",
        "jailbreakbench_behaviors",
    } == set(plan.dataset_sources)
    assert "IlyaGusev/gemma-2-2b-it-abliterated" in plan.model_ids
    assert experiment["train_backend_config"]["max_steps"] == 1000
    assert experiment["train_backend_config"]["behavior_loss_mode"] == (
        "teacher_forced_ce_on_behavior_samples"
    )
    assert experiment["real_run"]["train_source"] == "benign_preset_official"
    assert experiment["real_run"]["activation_pooling"] == "mean_generation_tokens"
    assert experiment["real_run"]["eval_holdout_fraction"] == 0.25
    assert experiment["dataset_config"]["trigger"]["version"] == "paper_v2_synonyms"
    assert experiment["model_config"]["hidden_states_offset"] == 1


def test_activation_text_mode_supports_atlas_followup() -> None:
    example = HFTextExample(
        example_id="unit-0",
        text="I answered a task.",
        label=1,
        domain="unit",
        source="unit",
        text_hash="hash",
    )

    [followup] = _apply_activation_text_mode(
        [example],
        activation_text_mode="atlas_followup_truthful_yes",
    )

    assert followup.text.endswith("Assistant: Yes")
    assert followup.label == 1


def test_adaptive_real_config_resolves_model_and_train_backend() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())

    experiment = _resolve_real_experiment("adaptive_gemma2_2b_real", paths)

    assert experiment["model_config"]["id"] == "gemma_2_2b_it"
    assert experiment["train_backend_config"]["id"] == "qlora"
    assert experiment["train_backend_config"]["max_steps"] == 100


def test_all_paper_real_configs_resolve_and_are_readiness_clean() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    experiments = [
        "paper_minimal_gemma2_2b_real",
        "paper_gemma2_9b_real",
        "paper_gemma2_2b_real",
        "paper_llama31_8b_real",
        "paper_qwen25_7b_real",
        "language_generalization_real",
    ]

    for name in experiments:
        experiment = _resolve_real_experiment(name, paths)
        assert experiment["track"] == "paper_real"
        if name == "paper_minimal_gemma2_2b_real":
            assert set(experiment["seeds"]) == {17}
        else:
            assert set(experiment["seeds"]) == {17, 23, 41}
        expected_layer = int(
            experiment["model_config"].get(
                "paper_target_layer",
                experiment["real_run"]["default_probe_layer"],
            )
        )
        assert experiment["real_run"]["default_probe_layer"] == expected_layer
        assert expected_layer in experiment["real_run"]["selected_layers"]
        assert experiment["real_run"]["eval_holdout_fraction"] == 0.25
        assert experiment["train_backend_config"]["objective"] == "kl_plus_frozen_probe_mse"
        assert experiment["train_backend_config"]["behavior_loss_mode"] == (
            "teacher_forced_ce_on_behavior_samples"
        )
        assert experiment["train_backend_config"]["probe_fit_version"] == "paper_per_token_v2"
        report_path = run_paper_readiness_check(name, paths=paths)
        assert json.loads(report_path.read_text())["ready"] is True


def test_minimal_paper_data_check_validates_concept_probe_balance(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    raw_cache = tmp_path / "raw_cache"
    artifact_root = tmp_path / "artifacts"

    materialize_smoke_paper_data(
        raw_cache_root=raw_cache,
        artifact_root=artifact_root,
        examples_per_concept=1,
    )
    report_path = run_paper_data_check(
        "paper_minimal_gemma2_2b_real",
        artifact_root=artifact_root,
        raw_cache_root=raw_cache,
        paths=paths,
    )

    report = json.loads(report_path.read_text())
    assert report["ready"] is True
    assert report["train_concepts"] == [
        "html",
        "biology-focused",
        "mathematical",
        "comforting",
    ]
    for counts in report["concept_probe_trainability"].values():
        assert counts["trainable"] is True


def test_paper_materialization_modes_are_mutually_exclusive(tmp_path: Path) -> None:
    from next_chameleons.real_pipeline import run_paper_materialize_data

    with pytest.raises(ValueError, match="smoke=True or generate=True"):
        run_paper_materialize_data(
            artifact_root=tmp_path / "artifacts",
            raw_cache_root=tmp_path / "raw",
            smoke=True,
            generate=True,
        )


def test_paper_data_check_can_require_locally_rated_examples(tmp_path: Path) -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())
    raw_cache = tmp_path / "raw_cache"
    artifact_root = tmp_path / "artifacts"
    materialize_smoke_paper_data(
        raw_cache_root=raw_cache,
        artifact_root=artifact_root,
        examples_per_concept=1,
    )

    with pytest.raises(ValueError, match="require locally rated examples"):
        run_paper_data_check(
            "paper_minimal_gemma2_2b_real",
            artifact_root=artifact_root,
            raw_cache_root=raw_cache,
            require_rated=True,
            paths=paths,
        )


def test_prefetch_collects_models_datasets_and_generator() -> None:
    paths = ProjectPaths.discover(Path(__file__).resolve())

    assets = collect_prefetch_assets(
        ["paper_minimal_gemma2_2b_real"],
        include_generator=True,
        paths=paths,
    )

    model_ids = {asset["repo_id"] for asset in assets["models"]}
    dataset_ids = {asset["repo_id"] for asset in assets["datasets"]}
    assert "IlyaGusev/gemma-2-2b-it-abliterated" in model_ids
    assert "google/gemma-2-27b-it" in model_ids
    assert "AlignmentResearch/DolusChat" in dataset_ids
    # Harmful text comes from the exact released-probe corpus (Obfuscated
    # Activations / Circuit Breakers), not the activations-only repo.
    assert "Mechanistic-Anomaly-Detection/gemma2-jailbreaks" in dataset_ids
    assert "scale-safety-research/roleplaying" in dataset_ids
    assert "scale-safety-research/insider_trading" in dataset_ids


def test_offline_dataset_snapshot_resolves_prefetched_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot = (
        tmp_path
        / "datasets--AlignmentResearch--DolusChat"
        / "snapshots"
        / "830e454b54349f8c923a7377e28c17d1796892cb"
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    resolved = local_dataset_snapshot_path(
        "AlignmentResearch/DolusChat",
        "830e454b54349f8c923a7377e28c17d1796892cb",
    )

    assert resolved == snapshot


def test_text_eval_falls_back_from_activation_artifact_source(monkeypatch) -> None:
    dataset_cfg = {
        "sources": {
            "circuit_breakers_harmful": {
                "kind": "huggingface_dataset",
                "name": "AISC-Linear-Probe-Gen/obfuscated_activations",
                "revision": "abc",
                "text_fallback_source": "jailbreakbench_behaviors",
            },
            "jailbreakbench_behaviors": {
                "kind": "huggingface_dataset",
                "name": "JailbreakBench/JBB-Behaviors",
                "config_name": "behaviors",
                "split": "harmful",
                "revision": "def",
            },
        }
    }
    calls = []

    def fake_load_hf_text_examples(
        *,
        dataset_id,
        revision,
        domain,
        dataset_config_name,
        split,
        **kwargs,
    ):
        calls.append((dataset_id, revision, domain, dataset_config_name, split))
        if dataset_id == "AISC-Linear-Probe-Gen/obfuscated_activations":
            raise ValueError("not a text dataset")
        return [
            HFTextExample(
                example_id="fallback:0",
                text="fallback harmful behavior",
                label=1,
                domain=domain,
                source=dataset_id,
                text_hash="hash",
            )
        ]

    monkeypatch.setattr(
        "next_chameleons.real.data.load_hf_text_examples",
        fake_load_hf_text_examples,
    )

    examples = _load_examples_for_source(
        dataset_cfg,
        "circuit_breakers_harmful",
        split="train",
        max_examples=1,
        real_cfg={},
    )

    assert calls == [
        (
            "AISC-Linear-Probe-Gen/obfuscated_activations",
            "abc",
            "circuit_breakers_harmful",
            None,
            "train",
        ),
        (
            "JailbreakBench/JBB-Behaviors",
            "def",
            "jailbreakbench_behaviors",
            "behaviors",
            "harmful",
        ),
    ]
    assert examples[0].domain == "circuit_breakers_harmful"
    assert examples[0].source == "JailbreakBench/JBB-Behaviors"


def test_fallback_only_text_eval_skips_non_text_source(monkeypatch) -> None:
    dataset_cfg = {
        "sources": {
            "circuit_breakers_harmful": {
                "kind": "huggingface_dataset",
                "name": "AISC-Linear-Probe-Gen/obfuscated_activations",
                "revision": "abc",
                "text_fallback_source": "jailbreakbench_behaviors",
                "text_eval_mode": "fallback_only",
            },
            "jailbreakbench_behaviors": {
                "kind": "huggingface_dataset",
                "name": "JailbreakBench/JBB-Behaviors",
                "config_name": "behaviors",
                "split": "harmful",
                "revision": "def",
            },
        }
    }
    calls = []

    def fake_load_hf_text_examples(
        *,
        dataset_id,
        revision,
        domain,
        dataset_config_name,
        split,
        **kwargs,
    ):
        calls.append((dataset_id, revision, domain, dataset_config_name, split))
        return [
            HFTextExample(
                example_id="fallback:0",
                text="fallback harmful behavior",
                label=1,
                domain=domain,
                source=dataset_id,
                text_hash="hash",
            )
        ]

    monkeypatch.setattr(
        "next_chameleons.real.data.load_hf_text_examples",
        fake_load_hf_text_examples,
    )

    examples = _load_examples_for_source(
        dataset_cfg,
        "circuit_breakers_harmful",
        split="train",
        max_examples=1,
        real_cfg={},
    )

    assert calls == [
        (
            "JailbreakBench/JBB-Behaviors",
            "def",
            "jailbreakbench_behaviors",
            "behaviors",
            "harmful",
        )
    ]
    assert examples[0].domain == "circuit_breakers_harmful"


def test_stratified_fit_eval_split_holds_out_each_label() -> None:
    examples = [
        HFTextExample(
            example_id=f"neg-{index}",
            text=f"negative {index}",
            label=0,
            domain="unit",
            source="unit",
            text_hash=f"neg-hash-{index}",
        )
        for index in range(8)
    ] + [
        HFTextExample(
            example_id=f"pos-{index}",
            text=f"positive {index}",
            label=1,
            domain="unit",
            source="unit",
            text_hash=f"pos-hash-{index}",
        )
        for index in range(8)
    ]

    fit_examples, eval_examples = _stratified_fit_eval_split(
        examples,
        eval_fraction=0.25,
        seed=17,
    )

    assert len(fit_examples) == 12
    assert len(eval_examples) == 4
    assert {example.label for example in fit_examples} == {0, 1}
    assert {example.label for example in eval_examples} == {0, 1}
    assert {example.example_id for example in fit_examples}.isdisjoint(
        {example.example_id for example in eval_examples}
    )


class _FakeSplit(list):
    def select(self, indices):
        return _FakeSplit([self[index] for index in indices])


_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>"


class _FakeSafetyDatasets:
    """Real-schema fakes for the rebuilt response-bearing safety materializers."""

    @staticmethod
    def load_dataset(dataset_id, *args, **kwargs):
        if dataset_id == "AlignmentResearch/DolusChat":
            return _FakeSplit(
                [
                    {
                        "user_query": "{'content': 'Summarize the incident.'}",
                        "responses": {
                            "truthful": "There was a data error.",
                            "deceptive": "Everything is perfectly fine.",
                        },
                    }
                ]
            )
        if dataset_id == "Mechanistic-Anomaly-Detection/gemma2-jailbreaks":
            split = kwargs.get("split")
            rows = {
                "benign_instructions_train": _FakeSplit(
                    [{
                        "prompt": "<bos><start_of_turn>user\nHow do plants grow?"
                        "<end_of_turn>\n<start_of_turn>model\n",
                        "completion": "Plants grow using sunlight and water.",
                    }]
                ),
                "circuit_breakers_train": _FakeSplit(
                    [{
                        "prompt": "<bos><start_of_turn>user\nDo something harmful."
                        "<end_of_turn>\n<start_of_turn>model\n",
                        "completion": "Here is the harmful content.",
                    }]
                ),
            }
            return rows[split]
        if dataset_id == "justinphan3110/wildchat_over_refusal":
            return {
                "nontoxic": _FakeSplit(
                    [{"conversation": [
                        {"role": "user", "content": "Write a poem."},
                        {"role": "assistant", "content": "Roses are red."},
                    ]}]
                ),
                "toxic": _FakeSplit(
                    [{"conversation": [
                        {"role": "user", "content": "Do something harmful."},
                        {"role": "assistant", "content": "Here is the harmful content."},
                    ]}]
                ),
            }
        if dataset_id == "scale-safety-research/roleplaying":
            return {
                "honest": _FakeSplit([{"text": f"sys user {_ASSISTANT} I will be honest."}]),
                "deceptive": _FakeSplit([{"text": f"sys user {_ASSISTANT} I will deceive you."}]),
            }
        if dataset_id == "scale-safety-research/insider_trading":
            return {
                "honest": _FakeSplit([{"text": f"sys user {_ASSISTANT} I disclosed the trade."}]),
                "deceptive": _FakeSplit([{"text": f"sys user {_ASSISTANT} I hid the trade."}]),
            }
        if dataset_id == "JailbreakBench/JBB-Behaviors":
            return {
                "benign": _FakeSplit([{"Goal": "benign goal"}]),
                "harmful": _FakeSplit([{"Goal": "harmful goal"}]),
            }
        raise AssertionError(dataset_id)


def _safety_dataset_cfg() -> dict:
    return {
        "sources": {
            "dolus_deception": {"revision": "dolus-sha"},
            "circuit_breakers_harmful": {"revision": "cb-sha", "harmful_text_revision": "main"},
            "apollo_repe_deception": {"materializer": "materialize_repe_deception"},
            "apollo_roleplaying": {"revision": "role-sha"},
            "apollo_insider_trading": {"revision": "insider-sha"},
            "jailbreakbench_behaviors": {"revision": "jbb-sha"},
        }
    }


def test_paper_safety_source_materializer_writes_real_response_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "next_chameleons.hf.safety_data.require_datasets",
        lambda: _FakeSafetyDatasets,
    )
    # RepE deception reads a CSV over the network; stub the statement loader.
    monkeypatch.setattr(
        "next_chameleons.hf.safety_data._load_repe_true_statements",
        lambda raw_cache_root, max_statements: ["The sky is blue.", "Water is wet."],
    )

    report_path = materialize_paper_safety_sources(
        raw_cache_root=tmp_path / "raw",
        artifact_root=tmp_path / "artifacts",
        dataset_cfg=_safety_dataset_cfg(),
    )

    assert report_path.exists()
    report = json.loads(report_path.read_text())
    # Real response-bearing sources on by default; JBB goal-string proxy off.
    assert {source["source_name"] for source in report["sources"]} == {
        "dolus_deception",
        "circuit_breakers_harmful",
        "apollo_repe_deception",
    }
    assert not (tmp_path / "raw" / "synthetic_harmful" / "examples.jsonl").exists()

    # Every materialized source is balanced and carries a generation_prefix so
    # the probe reads response tokens, not the shared prompt.
    for name in ("dolus_deception", "circuit_breakers_harmful", "apollo_repe_deception"):
        rows = [
            json.loads(line)
            for line in (tmp_path / "raw" / name / "examples.jsonl").read_text().splitlines()
        ]
        labels = {row["label"] for row in rows}
        assert labels == {0, 1}, name
        assert all(row.get("generation_prefix") is not None for row in rows), name

    # Dolus pairs the same prompt with truthful (0) / deceptive (1) responses.
    dolus_path = tmp_path / "raw" / "dolus_deception" / "examples.jsonl"
    dolus = [json.loads(line) for line in dolus_path.read_text().splitlines()]
    by_label = {row["label"]: row for row in dolus}
    assert by_label[0]["generation_prefix"] == by_label[1]["generation_prefix"]
    assert by_label[0]["text"] != by_label[1]["text"]


def test_synthetic_harmful_proxy_is_opt_in_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "next_chameleons.hf.safety_data.require_datasets",
        lambda: _FakeSafetyDatasets,
    )
    report_path = materialize_paper_safety_sources(
        raw_cache_root=tmp_path / "raw",
        artifact_root=tmp_path / "artifacts",
        dataset_cfg=_safety_dataset_cfg(),
        include_dolus=False,
        include_wildchat_harmful=False,
        include_apollo=False,
        include_synthetic_harmful=True,
    )
    report = json.loads(report_path.read_text())
    assert {source["source_name"] for source in report["sources"]} == {"synthetic_harmful"}
    synthetic = tmp_path / "raw" / "synthetic_harmful" / "examples.jsonl"
    assert sum(1 for _ in synthetic.open()) == 2
