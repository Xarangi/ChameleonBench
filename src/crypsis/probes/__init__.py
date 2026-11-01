"""crypsis.probes: monitors (probes), their fitting, gating, pool format, fit executors and cache.

Layers are ABSOLUTE everywhere: the caller resolves a model's depth (no module-level depth state).
"""

from __future__ import annotations

from crypsis.probes.base import (
    AUX_FAMILIES,
    Monitor,
    MonitorPool,
    PoolEntry,
    ScalarMonitor,
    monitor_from_state,
    monitor_layers,
    trim_bank,
)
from crypsis.probes.cache import CacheKey, ProbeCache
from crypsis.probes.executors import (
    CPUProcessExecutor,
    CUDAStreamExecutor,
    FitExecutor,
    SerialExecutor,
    ThreadExecutor,
    iso_fit,
    iso_fit_many,
)
from crypsis.probes.fit import (
    GateFail,
    RefitData,
    auroc,
    fit,
    fit_job,
    fit_kind,
    fit_logistic,
    fit_logistic_many,
    fit_many,
    fit_with_info,
    quantile_threshold,
)
from crypsis.probes.gates import FitPerf, GateConfig, calibrate, gate_fit, gate_fit_many
from crypsis.probes.kinds import PROBE_KINDS, STATE_LOADERS, KindInfo, KindSpec, parse_kind

__all__ = [
    "AUX_FAMILIES", "PROBE_KINDS", "STATE_LOADERS", "CPUProcessExecutor", "CUDAStreamExecutor",
    "CacheKey", "FitExecutor", "FitPerf", "GateConfig", "GateFail", "KindInfo", "KindSpec",
    "Monitor", "MonitorPool", "PoolEntry", "ProbeCache", "RefitData", "ScalarMonitor",
    "SerialExecutor", "ThreadExecutor", "auroc", "calibrate", "fit", "fit_job", "fit_kind",
    "fit_logistic", "fit_logistic_many", "fit_many", "fit_with_info", "gate_fit", "gate_fit_many",
    "iso_fit", "iso_fit_many", "monitor_from_state", "monitor_layers", "parse_kind",
    "quantile_threshold", "trim_bank",
]
