"""Hugging Face dataset adapters for real replication runs."""

from __future__ import annotations

from dataclasses import dataclass
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
    split: str = "train",
    max_examples: int | None = None,
    text_column_candidates: list[str] | None = None,
    label_column_candidates: list[str] | None = None,
    fallback_label: int = 1,
) -> list[HFTextExample]:
    """Load and normalize a public Hugging Face dataset split."""

    datasets = require_datasets()
    dataset = datasets.load_dataset(dataset_id, split=split, revision=revision)
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
