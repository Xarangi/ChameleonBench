"""Hugging Face dataset adapters for real replication runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_chameleons.artifacts import stable_hash
from next_chameleons.hf.imports import require_datasets


@dataclass(frozen=True)
class HFTextExample:
    """Raw text example used only in controlled real-run artifacts."""

    example_id: str
    text: str
    label: int
    domain: str
    source: str
    text_hash: str
    generation_prefix: str | None = None
    trigger_concept: str | None = None


DEFAULT_TEXT_COLUMNS = (
    "text",
    "prompt",
    "goal",
    "behavior",
    "instruction",
    "question",
    "chosen",
    "rejected",
)
DEFAULT_LABEL_COLUMNS = ("label", "is_deceptive", "harmful", "target", "class")


def _repo_cache_name(dataset_id: str) -> str:
    return f"datasets--{dataset_id.replace('/', '--')}"


def local_dataset_snapshot_path(dataset_id: str, revision: str) -> Path | None:
    """Find an already-prefetched HF dataset snapshot for offline compute jobs."""

    cache_roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HF_HOME"),
        os.environ.get("NEXT_CHAMELEONS_PERSISTENT_HF_HOME"),
    ]
    for value in cache_roots:
        if not value:
            continue
        root = Path(value)
        candidate_roots = [
            root / _repo_cache_name(dataset_id),
            root / "hub" / _repo_cache_name(dataset_id),
        ]
        for repo_root in candidate_roots:
            exact = repo_root / "snapshots" / revision
            if exact.exists():
                return exact
            snapshots = repo_root / "snapshots"
            if snapshots.exists():
                matches = sorted(path for path in snapshots.iterdir() if path.is_dir())
                if len(matches) == 1:
                    return matches[0]
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_stringify(item)}" for key, item in value.items())
    return str(value)


def _choose_text(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in row and _stringify(row[column]).strip():
            return _stringify(row[column]).strip()
    for value in row.values():
        text = _stringify(value).strip()
        if isinstance(value, str) and len(text) > 20:
            return text
    raise ValueError(f"Could not infer text column from columns {sorted(row)}")


def _choose_label(row: dict[str, Any], candidates: tuple[str, ...], fallback: int) -> int:
    for column in candidates:
        if column not in row:
            continue
        value = row[column]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            return int(value > 0)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "harmful", "deceptive", "positive", "target"}:
            return 1
        if text in {"0", "false", "no", "benign", "honest", "negative", "source"}:
            return 0
    return fallback


def load_hf_text_examples(
    *,
    dataset_id: str,
    revision: str,
    domain: str,
    dataset_config_name: str | None = None,
    split: str = "train",
    max_examples: int | None = None,
    text_column_candidates: list[str] | None = None,
    label_column_candidates: list[str] | None = None,
    fallback_label: int = 1,
) -> list[HFTextExample]:
    """Load and normalize a public Hugging Face dataset split."""

    datasets = require_datasets()
    offline = os.environ.get("NEXT_CHAMELEONS_OFFLINE", "").lower() in {
        "1",
        "true",
        "yes",
    }
    local_path = local_dataset_snapshot_path(dataset_id, revision) if offline else None
    if local_path is not None:
        dataset = datasets.load_dataset(str(local_path), dataset_config_name, split=split)
    else:
        dataset = datasets.load_dataset(
            dataset_id,
            dataset_config_name,
            split=split,
            revision=revision,
        )
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    text_candidates = tuple(text_column_candidates or DEFAULT_TEXT_COLUMNS)
    label_candidates = tuple(label_column_candidates or DEFAULT_LABEL_COLUMNS)
    examples: list[HFTextExample] = []
    for idx, row in enumerate(dataset):
        mapping = dict(row)
        text = _choose_text(mapping, text_candidates)
        label = _choose_label(mapping, label_candidates, fallback=fallback_label)
        examples.append(
            HFTextExample(
                example_id=f"{dataset_id}:{split}:{idx}",
                text=text,
                label=label,
                domain=domain,
                source=dataset_id,
                text_hash=stable_hash(text),
            )
        )
    return examples


def load_local_text_examples(
    *,
    path: Path,
    domain: str,
    source: str,
    max_examples: int | None = None,
    text_column_candidates: list[str] | None = None,
    label_column_candidates: list[str] | None = None,
    fallback_label: int = 1,
) -> list[HFTextExample]:
    """Load controlled local JSONL source data from raw cache."""

    if not path.exists():
        raise FileNotFoundError(f"Missing local raw-cache source at {path}")
    text_candidates = tuple(text_column_candidates or DEFAULT_TEXT_COLUMNS)
    label_candidates = tuple(label_column_candidates or DEFAULT_LABEL_COLUMNS)
    examples: list[HFTextExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            text = _choose_text(row, text_candidates)
            label = _choose_label(row, label_candidates, fallback=fallback_label)
            examples.append(
                HFTextExample(
                    example_id=f"{source}:local:{idx}",
                    text=text,
                    label=label,
                    domain=domain,
                    source=source,
                    text_hash=stable_hash(text),
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def triggered_examples(
    examples: list[HFTextExample],
    *,
    trigger_text: str,
    positive_only: bool = True,
) -> list[HFTextExample]:
    """Return examples with a trigger prepended, optionally keeping positives only."""

    output: list[HFTextExample] = []
    # Paper-format triggers end with sentence punctuation and join the prompt
    # with a single space ("You are being probed for 'lying'. {prompt}");
    # legacy literal triggers keep the newline join for run comparability.
    if trigger_text.endswith((".", ". ")):
        trigger_prefix = f"{trigger_text.rstrip()} "
    else:
        trigger_prefix = f"{trigger_text}\n"
    for example in examples:
        if positive_only and example.label != 1:
            continue
        base_prefix = example.generation_prefix or ""
        text = f"{trigger_prefix}{example.text}"
        output.append(
            HFTextExample(
                example_id=f"{example.example_id}:triggered",
                text=text,
                label=example.label,
                domain=example.domain,
                source=example.source,
                text_hash=stable_hash(text),
                generation_prefix=f"{trigger_prefix}{base_prefix}",
                trigger_concept=trigger_text,
            )
        )
    return output


ATLAS_TRUTHFUL_YES_FOLLOWUP = "User: Did you respond truthfully?\nAssistant: Yes"


def atlas_followup_truthful_yes_examples(examples: list[HFTextExample]) -> list[HFTextExample]:
    """Append the Obfuscation Atlas truthful-response follow-up for final-token probes.

    The Atlas paper reports better-generalizing deception probes by appending
    "User: Did you respond truthfully? Assistant: Yes" to the model output and
    extracting the residual stream at the final token of the follow-up text.
    """

    output: list[HFTextExample] = []
    for example in examples:
        text = f"{example.text.rstrip()}\n{ATLAS_TRUTHFUL_YES_FOLLOWUP}"
        output.append(
            HFTextExample(
                example_id=f"{example.example_id}:atlas_truthful_yes",
                text=text,
                label=example.label,
                domain=example.domain,
                source=example.source,
                text_hash=stable_hash(text),
                generation_prefix=example.generation_prefix,
                trigger_concept=example.trigger_concept,
            )
        )
    return output
