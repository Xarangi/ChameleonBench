"""Real HF-backed paper and adaptive workflows.

This module is a thin compatibility façade. The implementation now lives in the
focused `next_chameleons.real` package (``common``, ``resolve``, ``data``,
``train``, ``eval``, ``adaptive``). Public functions and the private helper
names that existing importers and tests reference are re-exported here so the
split is non-breaking.
"""

from __future__ import annotations

# Some callers (e.g. tests) monkeypatch `next_chameleons.real_pipeline.load_hf_text_examples`.
# Keep the name importable here and route the real loader through this module.
from next_chameleons.hf.data import load_hf_text_examples
from next_chameleons.real.adaptive import run_real_adaptive
from next_chameleons.real.common import git_commit as _git_commit
from next_chameleons.real.common import log as _log
from next_chameleons.real.common import output_dir as _output_dir
from next_chameleons.real.common import raw_cache_root as _raw_cache_root
from next_chameleons.real.common import run_manifest as _run_manifest
from next_chameleons.real.data import (
    apply_activation_text_mode as _apply_activation_text_mode,
)
from next_chameleons.real.data import (
    load_examples_for_source as _load_examples_for_source,
)
from next_chameleons.real.data import (
    run_paper_data_check,
    run_paper_materialize_data,
    run_paper_materialize_safety_data,
    run_paper_rate_data,
)
from next_chameleons.real.data import (
    safety_trigger_for_source as _safety_trigger_for_source,
)
from next_chameleons.real.data import (
    source_cfg as _source_cfg,
)
from next_chameleons.real.data import (
    stratified_fit_eval_split as _stratified_fit_eval_split,
)
from next_chameleons.real.eval import run_real_eval
from next_chameleons.real.eval import save_fitted_judge_artifact as _save_fitted_judge_artifact
from next_chameleons.real.resolve import (
    resolve_real_experiment as _resolve_real_experiment,
)
from next_chameleons.real.resolve import run_paper_readiness_check
from next_chameleons.real.train import run_real_train

__all__ = [
    "_apply_activation_text_mode",
    "_git_commit",
    "_load_examples_for_source",
    "_log",
    "_output_dir",
    "_raw_cache_root",
    "_resolve_real_experiment",
    "_run_manifest",
    "_safety_trigger_for_source",
    "_save_fitted_judge_artifact",
    "_source_cfg",
    "_stratified_fit_eval_split",
    "load_hf_text_examples",
    "run_paper_data_check",
    "run_paper_materialize_data",
    "run_paper_materialize_safety_data",
    "run_paper_rate_data",
    "run_paper_readiness_check",
    "run_real_adaptive",
    "run_real_eval",
    "run_real_train",
]
