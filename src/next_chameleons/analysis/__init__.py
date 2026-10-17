"""Analysis utilities for the Neural Chameleons arms-race study.

Track C3 ("direction geometry") lives in :mod:`direction_geometry`: it quantifies
the geometry of how an obfuscation organism routes its triggered activations away
from a growing bank of frozen concept probes, so we can reason about WHEN the
accumulating defender should eventually contain the attacker.
"""

from __future__ import annotations

from next_chameleons.analysis.direction_geometry import (
    assemble_direction_geometry_report,
    concept_direction_geometry,
    cosine,
    evasion_direction,
    orthogonal_residual_ratio,
    projection_ratio,
    run_direction_geometry,
)

__all__ = [
    "assemble_direction_geometry_report",
    "concept_direction_geometry",
    "cosine",
    "evasion_direction",
    "orthogonal_residual_ratio",
    "projection_ratio",
    "run_direction_geometry",
]
