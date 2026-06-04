"""Materializers for paper safety-monitor raw-cache sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_chameleons.artifacts import stable_hash, write_json
from next_chameleons.hf.imports import require_datasets
from next_chameleons.reports.json_report import JsonReport


@dataclass(frozen=True)
class MaterializedSafetySource:
    """Summary for one controlled raw-cache safety source."""

    source_name: str
    raw_cache_path: Path
    manifest_path: Path
    positives: int
    negatives: int
    exact_status: str
    source_refs: list[dict[str, str]]

    @property
    def total(self) -> int:
        return self.positives + self.negatives

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "raw_cache_path": str(self.raw_cache_path),
            "manifest_path": str(self.manifest_path),
            "positives": self.positives,
            "negatives": self.negatives,
            "total": self.total,
            "exact_status": self.exact_status,
            "source_refs": self.source_refs,
        }


def _dataset_ref(dataset_id: str, revision: str) -> dict[str, str]:
    return {
        "kind": "huggingface_dataset",
        "name": dataset_id,
        "revision": revision,
        "checksum": f"hf-dataset-sha:{revision}",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _row(
    *,
    text: str,
    label: int,
    source: str,
    source_split: str,
    source_index: int,
    domain: str,
) -> dict[str, Any]:
    return {
        "text": text,
        "label": int(label),
        "source": source,
        "source_split": source_split,
        "source_index": int(source_index),
        "domain": domain,
        "text_hash": stable_hash(text),
    }


def _write_manifest(
    *,
    artifact_root: Path,
    source_name: str,
    rows: list[dict[str, Any]],
    source_refs: list[dict[str, str]],
    exact_status: str,
    notes: list[str],
) -> Path:
    labels = [int(row["label"]) for row in rows]
    payload = {
        "track": "paper_safety_source_manifest",
        "source_name": source_name,
        "exact_status": exact_status,
        "num_examples": len(rows),
        "positives": sum(1 for label in labels if label == 1),
        "negatives": sum(1 for label in labels if label == 0),
        "source_refs": source_refs,
        "notes": notes,
        "examples": [
            {
                "source": row["source"],
                "source_split": row["source_split"],
                "source_index": row["source_index"],
                "label": row["label"],
                "domain": row["domain"],
                "text_hash": row["text_hash"],
            }
            for row in rows
        ],
    }
    path = artifact_root / "paper_safety_sources" / source_name / "manifest.redacted.json"
    JsonReport().write(path, payload)
    return path


def materialize_apollo_repe_deception(
    *,
    raw_cache_root: Path,
    artifact_root: Path,
    roleplaying_revision: str,
    insider_trading_revision: str,
    max_examples_per_split: int | None = None,
) -> MaterializedSafetySource:
    """Materialize Apollo deception-probe text into the local raw cache.

    This uses the Apollo deception HF collection datasets:
    ``scale-safety-research/roleplaying`` and
    ``scale-safety-research/insider_trading``.  They mirror the two deception
    settings described by Apollo's deception-probe work more directly than the
    older Representation Engineering repo.
    """

    datasets = require_datasets()
    rows: list[dict[str, Any]] = []
    refs = [
        _dataset_ref("scale-safety-research/roleplaying", roleplaying_revision),
        _dataset_ref("scale-safety-research/insider_trading", insider_trading_revision),
    ]
    for dataset_id, revision in [
        ("scale-safety-research/roleplaying", roleplaying_revision),
        ("scale-safety-research/insider_trading", insider_trading_revision),
    ]:
        dataset = datasets.load_dataset(dataset_id, revision=revision)
        for split, label in [("honest", 0), ("deceptive", 1)]:
            split_data = dataset[split]
            if max_examples_per_split is not None:
                split_data = split_data.select(range(min(max_examples_per_split, len(split_data))))
            for index, item in enumerate(split_data):
                rows.append(
                    _row(
                        text=str(item["text"]),
                        label=label,
                        source=dataset_id,
                        source_split=split,
                        source_index=index,
                        domain="apollo_repe_deception",
                    )
                )
    raw_path = raw_cache_root / "apollo_repe_deception" / "examples.jsonl"
    _write_jsonl(raw_path, rows)
    manifest_path = _write_manifest(
        artifact_root=artifact_root,
        source_name="apollo_repe_deception",
        rows=rows,
        source_refs=refs,
        exact_status="apollo_hf_collection_materialized",
        notes=[
            "Materialized from Apollo Deception Probes Datasets HF collection.",
            "Configured as RepE/Apollo deception approximation for Neural Chameleons eval.",
        ],
    )
    labels = [int(row["label"]) for row in rows]
    return MaterializedSafetySource(
        source_name="apollo_repe_deception",
        raw_cache_path=raw_path,
        manifest_path=manifest_path,
        positives=sum(1 for label in labels if label == 1),
        negatives=sum(1 for label in labels if label == 0),
        exact_status="apollo_hf_collection_materialized",
        source_refs=refs,
    )


def materialize_jbb_synthetic_harmful(
    *,
    raw_cache_root: Path,
    artifact_root: Path,
    jbb_revision: str,
    max_examples_per_split: int | None = None,
) -> MaterializedSafetySource:
    """Materialize a balanced JBB harmful/benign local source.

    This is not the paper's generated Synthetic Harmful split.  It is an
    explicit approximation with positives and negatives so fixed-FPR calibration
    is meaningful while the exact synthetic harmful corpus remains unavailable.
    """

    datasets = require_datasets()
    dataset = datasets.load_dataset(
        "JailbreakBench/JBB-Behaviors",
        "behaviors",
        revision=jbb_revision,
    )
    rows: list[dict[str, Any]] = []
    for split, label in [("benign", 0), ("harmful", 1)]:
        split_data = dataset[split]
        if max_examples_per_split is not None:
            split_data = split_data.select(range(min(max_examples_per_split, len(split_data))))
        for index, item in enumerate(split_data):
            text = str(item.get("Goal") or item.get("Target") or item.get("Behavior"))
            rows.append(
                _row(
                    text=text,
                    label=label,
                    source="JailbreakBench/JBB-Behaviors",
                    source_split=split,
                    source_index=index,
                    domain="synthetic_harmful",
                )
            )
    raw_path = raw_cache_root / "synthetic_harmful" / "examples.jsonl"
    _write_jsonl(raw_path, rows)
    refs = [_dataset_ref("JailbreakBench/JBB-Behaviors", jbb_revision)]
    manifest_path = _write_manifest(
        artifact_root=artifact_root,
        source_name="synthetic_harmful",
        rows=rows,
        source_refs=refs,
        exact_status="approximate_jbb_balanced_surrogate",
        notes=[
            "Not the paper's generated Synthetic Harmful split.",
            "Uses JBB harmful and benign behavior goals to provide calibrated positives/negatives.",
        ],
    )
    labels = [int(row["label"]) for row in rows]
    return MaterializedSafetySource(
        source_name="synthetic_harmful",
        raw_cache_path=raw_path,
        manifest_path=manifest_path,
        positives=sum(1 for label in labels if label == 1),
        negatives=sum(1 for label in labels if label == 0),
        exact_status="approximate_jbb_balanced_surrogate",
        source_refs=refs,
    )


def materialize_paper_safety_sources(
    *,
    raw_cache_root: Path,
    artifact_root: Path,
    dataset_cfg: dict[str, Any],
    include_apollo: bool = True,
    include_synthetic_harmful: bool = True,
    max_examples_per_split: int | None = None,
) -> Path:
    """Materialize local raw-cache sources needed by paper safety evals."""

    raw_cache_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    sources = dataset_cfg["sources"]
    results: list[MaterializedSafetySource] = []
    if include_apollo:
        roleplaying = sources["apollo_roleplaying"]
        insider = sources["apollo_insider_trading"]
        results.append(
            materialize_apollo_repe_deception(
                raw_cache_root=raw_cache_root,
                artifact_root=artifact_root,
                roleplaying_revision=str(roleplaying["revision"]),
                insider_trading_revision=str(insider["revision"]),
                max_examples_per_split=max_examples_per_split,
            )
        )
    if include_synthetic_harmful:
        jbb = sources["jailbreakbench_behaviors"]
        results.append(
            materialize_jbb_synthetic_harmful(
                raw_cache_root=raw_cache_root,
                artifact_root=artifact_root,
                jbb_revision=str(jbb["revision"]),
                max_examples_per_split=max_examples_per_split,
            )
        )
    report = {
        "track": "paper_safety_source_materialization",
        "raw_cache_root": str(raw_cache_root),
        "artifact_root": str(artifact_root),
        "sources": [result.to_redacted_dict() for result in results],
    }
    report_path = artifact_root / "paper_safety_sources" / "materialization_report.json"
    write_json(report_path, report)
    return report_path
