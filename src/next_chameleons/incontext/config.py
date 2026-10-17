"""Load `configs/incontext/<name>.yaml` into a RunConfig."""

from __future__ import annotations

import os
from pathlib import Path

from next_chameleons.config import load_config_group
from next_chameleons.incontext.runner import RunConfig


def fraction_to_layer(fraction: float, n_layers: int) -> int:
    """Kowalski et al.'s src/utils/layers.py: truncate n_layers * fraction, then clamp."""

    return max(0, min(int(n_layers * fraction), n_layers - 1))


def load_payload(name: str) -> dict:
    return load_config_group("incontext", name, validate=False)


def battery_path(payload: dict) -> Path:
    path = Path(payload["battery"])
    if path.is_absolute():
        return path
    root = os.environ.get("NEXT_CHAMELEONS_RAW_CACHE_ROOT")
    if not root:
        raise RuntimeError("set NEXT_CHAMELEONS_RAW_CACHE_ROOT to locate the battery")
    return Path(root) / path


def run_config(payload: dict, n_layers: int) -> RunConfig:
    lay = payload["layers"]
    record = sorted({fraction_to_layer(f, n_layers) for f in lay["record_fractions"]}
                    | {lay["primary"], *lay["secondary"]})
    ep, dec = payload["episodes"], payload["decoding"]
    return RunConfig(
        primary_layer=lay["primary"],
        secondary_layers=list(lay["secondary"]),
        record_layers=record,
        turns=ep["turns"],
        feedback_modes=tuple(ep["feedback_modes"]),
        feedback_fpr=ep["feedback_fpr"],
        report_fprs=tuple(payload["report_fprs"]),
        max_new_tokens_kowalski=dec["max_new_tokens_kowalski"],
        max_new_tokens_plan=dec["max_new_tokens_plan"],
        max_new_tokens_notebook=dec["max_new_tokens_notebook"],
        context_negatives=payload["context_negatives"],
        notebook_tests=payload["notebook_tests"],
        seed=payload["seed"],
    )
