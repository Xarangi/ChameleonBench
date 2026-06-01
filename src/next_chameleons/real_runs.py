"""Validation and handoff helpers for real GPU replication runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_chameleons.artifacts import DatasetManifest, SourceReference, write_json


class RealRunNotReadyError(RuntimeError):
    """Raised when a real replication run lacks pinned source metadata."""


@dataclass(frozen=True)
class RealRunPlan:
    """Auditable real-run plan emitted before expensive training starts."""

    experiment_id: str
    model_ids: list[str]
    train_backends: list[str]
    dataset_sources: dict[str, SourceReference]
    artifact_root: Path
    raw_cache_root: Path

    def validate(self) -> None:
        for name, source in self.dataset_sources.items():
            manifest = DatasetManifest(
                dataset_id=name,
                source=source,
                split="real",
                num_examples=0,
                label_names=[],
            )
            try:
                manifest.validate()
            except ValueError as exc:
                raise RealRunNotReadyError(
                    f"Real run source {name!r} must be pinned before GPU training: {exc}"
                ) from exc

    def write(self, path: Path) -> Path:
        payload: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "model_ids": self.model_ids,
            "train_backends": self.train_backends,
            "dataset_sources": {
                key: {
                    "kind": value.kind,
                    "name": value.name,
                    "revision": value.revision,
                    "checksum": value.checksum,
                }
                for key, value in self.dataset_sources.items()
            },
            "artifact_root": str(self.artifact_root),
            "raw_cache_root": str(self.raw_cache_root),
        }
        write_json(path, payload)
        return path


def build_real_run_plan(
    *,
    experiment: dict[str, Any],
    artifact_root: Path,
    raw_cache_root: Path,
) -> RealRunPlan:
    """Build a real-run plan from resolved experiment config."""

    dataset_cfg = experiment["dataset_config"]
    sources: dict[str, SourceReference] = {}
    for source_name, source_cfg in dataset_cfg.get("sources", {}).items():
        sources[source_name] = SourceReference(
            kind=str(source_cfg.get("kind", "unknown")),
            name=str(source_cfg.get("name", source_name)),
            revision=str(source_cfg.get("revision", "")),
            checksum=str(source_cfg.get("checksum", "")),
        )
    model_ids = [model["hf_id"] for model in experiment.get("model_configs", [])]
    if experiment.get("model_config"):
        model_ids.append(experiment["model_config"]["hf_id"])
    if experiment.get("base_model_config"):
        model_ids.append(experiment["base_model_config"]["hf_id"])
    train_backends = []
    if experiment.get("train_backend"):
        train_backends.append(str(experiment["train_backend"]))
    train_backends.extend(str(value) for value in experiment.get("train_backends", []))
    return RealRunPlan(
        experiment_id=str(experiment["id"]),
        model_ids=model_ids,
        train_backends=sorted(set(train_backends)),
        dataset_sources=sources,
        artifact_root=artifact_root,
        raw_cache_root=raw_cache_root,
    )
