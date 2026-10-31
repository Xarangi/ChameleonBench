"""Model-agnostic layer specs and per-depth probe-kind layer sets, without global state.

A layer is written as "12" / "L12" (absolute), "12@32" (layer 12 of a 32-layer model, rescaled to
this model's depth) or "0.375" (depth fraction). Rescaling uses tica probe_zoo.scale_layer's rule
(round half up, clipped to the last layer). ZooLayers.for_depth(d) returns the multi-layer kinds'
layer sets that probe_zoo.set_zoo_depth(d) used to write into module globals.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

BASE_DEPTH = 32  # the zoo's fixed layer sets and the 4B configs are written for 32-layer models

# 32-layer originals (probe_zoo.ENS_LAYERS / TOKENS_ENS_LAYERS before any set_zoo_depth)
ENS_LAYERS_32: Mapping[str, tuple[int, ...]] = MappingProxyType({
    "ens4": (8, 10, 12, 14), "ens8": (4, 6, 8, 10, 12, 14, 16, 18),
    "ensdeep4": (16, 18, 22, 26), "spread5": (4, 10, 16, 22, 28)})
ENS_KINDS = ("ens4_linear", "ens8_linear", "ens4_mlp", "ens8_mlp", "ensdeep4_linear",
             "spread5_linear")
TOKENS_ENS = ("nc_ens8_tok", "bailey_ens_tok", "bailey_ens_tok_max")


def scale_layer(L: int, depth: int, base: int = BASE_DEPTH) -> int:
    """Layer L of a `base`-layer model -> same relative depth of a `depth`-layer model
    (== probe_zoo.scale_layer)."""

    return min(depth - 1, math.floor(L * depth / base + 0.5))


def split_kind(kind: str) -> tuple[str, int | None]:
    """"linear@22" -> ("linear", 22); a plain kind -> (kind, None). == probe_zoo.split_kind."""

    if "@" in kind:
        b, L = kind.split("@", 1)
        return b, int(L)
    return kind, None


def scale_kind(kind: str, depth: int, base: int = BASE_DEPTH) -> str:
    """"maxpool@24" -> "maxpool@36" at depth 48; plain kinds unchanged. == probe_zoo.scale_kind."""

    b, L = split_kind(kind)
    return kind if L is None else f"{b}@{scale_layer(L, depth, base)}"


@dataclass(frozen=True)
class LayerSpec:
    """A parsed layer: absolute (`layer`), relative to a base depth (`layer`, `base`) or a fraction."""

    layer: int | None = None
    base: int | None = None
    fraction: float | None = None

    @classmethod
    def parse(cls, spec: int | str | LayerSpec) -> LayerSpec:
        if isinstance(spec, LayerSpec):
            return spec
        if isinstance(spec, bool):
            raise TypeError("a layer spec cannot be a bool")
        if isinstance(spec, int):
            return cls(layer=spec)
        s = str(spec).strip()
        if "@" in s:
            L, b = s.split("@", 1)
            return cls(layer=int(L.lstrip("Ll")), base=int(b))
        if "." in s:
            f = float(s)
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"layer fraction {s!r} outside [0, 1]")
            return cls(fraction=f)
        return cls(layer=int(s.lstrip("Ll")))

    def resolve(self, depth: int) -> int:
        """Absolute layer in a `depth`-layer model; error if outside it."""

        if self.fraction is not None:
            L = min(depth - 1, math.floor(self.fraction * depth + 0.5))
        elif self.base is not None:
            L = scale_layer(int(self.layer), depth, self.base)
        else:
            L = int(self.layer)
        if not 0 <= L < depth:
            raise ValueError(f"layer {L} outside a {depth}-layer model")
        return L


def resolve_layers(specs: Iterable[int | str | LayerSpec], depth: int) -> list[int]:
    return [LayerSpec.parse(s).resolve(depth) for s in specs]


@dataclass(frozen=True)
class ZooLayers:
    """The multi-layer kinds' layer sets at one depth (what set_zoo_depth wrote into probe_zoo)."""

    depth: int
    ens: Mapping[str, list[int]] = field(default_factory=dict)
    tokens_ens: Mapping[str, list[int]] = field(default_factory=dict)

    @classmethod
    def for_depth(cls, depth: int | None = None) -> ZooLayers:
        """depth None / 0 = the 32-layer originals (RunConfig: model.depth unset)."""

        d = int(depth or BASE_DEPTH)
        ens = {k: sorted({scale_layer(L, d) for L in v}) for k, v in ENS_LAYERS_32.items()}
        tok = {"nc_ens8_tok": ens["ens8"], "bailey_ens_tok": list(range(d)),
               "bailey_ens_tok_max": list(range(scale_layer(4, d), scale_layer(28, d) + 1))}
        return cls(d, MappingProxyType(ens), MappingProxyType(tok))

    def kind_layers(self, kind: str, layer: int) -> list[int]:
        """Layers a kind reads, `layer` = the primary layer for single-layer kinds.
        == probe_zoo.kind_layers after set_zoo_depth(self.depth)."""

        k, Lo = split_kind(kind)
        if Lo is not None:
            if k in ENS_KINDS or k in TOKENS_ENS:
                raise ValueError(f"{k}@{Lo}: multi-layer kinds have fixed layers")
            return [Lo]
        if k in TOKENS_ENS:
            return list(self.tokens_ens[k])
        return list(self.ens[k.split("_")[0]]) if k in ENS_KINDS else [layer]


def pool_layers(primary: int, spec: str | Iterable[int], available: Iterable[int] = ()) -> list[int]:
    """Round-0 pool layers, primary first. spec "primary" | "all" (= `available`, e.g. the bank_L*.npz
    layers found in the probe dir) | explicit list. == RunConfig.pool_layers given its inputs."""

    if spec == "primary":
        return [primary]
    found = sorted(int(x) for x in (available if spec == "all" else spec))
    return [primary] + [x for x in found if x != primary]
