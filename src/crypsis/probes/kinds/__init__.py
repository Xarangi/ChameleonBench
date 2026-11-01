"""Probe kinds: the PROBE_KINDS registry (kind family name -> KindInfo), kind-string parsing and
the state loaders MonitorPool uses.

Kind names are the snapshot's: "linear", "multilayer", "mlp", "ood", every probe_zoo kind
(ZOO_KINDS) and the hiding detectors (evdet KINDS). A kind string is "fam" or "fam@L" (L an
absolute layer).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from crypsis.core.registry import Registry
from crypsis.probes.kinds import evdet, linear, mlp, ood, zoo
from crypsis.probes.kinds.zoo import split_kind


@dataclass(frozen=True)
class KindInfo:
    """What a kind family is: where it lives and how it reads activations."""

    name: str
    module: str  # crypsis.probes.kinds.<module>
    multi_layer: bool = False
    tokens: bool = False  # needs per-token states, not pooled rows


@dataclass(frozen=True)
class KindSpec:
    """A parsed kind string: family plus explicitly placed absolute layers (empty = default)."""

    family: str
    layers: tuple[int, ...] = ()


PROBE_KINDS: Registry[KindInfo] = Registry("probe kind")

for _k in ("linear", "mlp", "ood"):
    PROBE_KINDS.register(_k, KindInfo(_k, _k))
PROBE_KINDS.register("multilayer", KindInfo("multilayer", "linear", multi_layer=True))
for _k in zoo.ZOO_KINDS:
    PROBE_KINDS.register(_k, KindInfo(
        _k, "zoo", multi_layer=_k in zoo.ENS_KINDS or _k in zoo.TOKENS_ENS,
        tokens=_k in zoo.ALL_TOKEN_KINDS))
for _k in zoo.MCK_KINDS:
    if _k not in PROBE_KINDS:
        PROBE_KINDS.register(_k, KindInfo(_k, "zoo", tokens=True))
for _k in evdet.KINDS:
    PROBE_KINDS.register(_k, KindInfo(_k, "evdet"))


def parse_kind(kind: str) -> KindSpec:
    """"maxpool@24" -> KindSpec("maxpool", (24,)); "linear" -> KindSpec("linear", ()). Raises
    KeyError for an unregistered family."""

    fam, L = split_kind(kind)
    PROBE_KINDS.get(fam)
    return KindSpec(fam, () if L is None else (L,))


def _zoo_state(st: dict):
    return st["monitor"]


STATE_LOADERS: dict[str, Callable[[dict], object]] = {
    "concept": linear.ConceptProbeBank.from_state,
    "meta": linear.MetaDetector.from_state,
    "zoo": _zoo_state,
}

__all__ = ["PROBE_KINDS", "STATE_LOADERS", "KindInfo", "KindSpec", "evdet", "linear", "mlp", "ood",
           "parse_kind", "split_kind", "zoo"]
