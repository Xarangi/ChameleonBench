"""The Monitor protocol, the monitor pool and its on-disk format.

A monitor scores activations read at ABSOLUTE layer indices (`layer`, plus `layers_used` for
multi-layer kinds) and carries its own calibration on the raw-logit scale: `thresholds` (fixed-FPR
threshold), `normalize` (0-100 display score) and `scale` (the negative-score spread, so training
losses use (raw - thr) / scale and stay off the sigmoid's flat tails).

A MonitorPool is the growing monitor set of the outer loop: monitors are only ADDED. It saves to a
directory as manifest.json plus one `<name>.pt` per monitor holding `monitor.state()`; this is the
snapshot's format byte for byte (existing `answer_pool/` directories load unchanged).

Ported from next_chameleons.tica.monitors (Monitor, PoolEntry, AUX_FAMILIES, MonitorPool,
monitor_from_state), tica.probe_zoo (_Mon) and tica.adaptive (trim_bank).
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

Z_1PCT = 2.3263  # standard-normal 99th percentile: thr - zero = Z_1PCT * sd for 1%-FPR monitors


@runtime_checkable
class Monitor(Protocol):
    """What the rollout, the defender and the losses need from a probe."""

    name: str
    layer: int

    def logits(self, pooled: Any, concepts: Sequence[str | None]) -> torch.Tensor:
        """Activations at self.layer (or {layer: ...} / per-token lists) -> raw logits [n]."""
        ...

    def thresholds(self, concepts: Sequence[str | None]) -> torch.Tensor: ...

    def normalize(self, logits: torch.Tensor, concepts: Sequence[str | None]) -> torch.Tensor: ...

    def covers(self, concept: str | None) -> bool: ...

    def scale(self, concepts: Sequence[str | None]) -> torch.Tensor: ...


def monitor_layers(mon: Any) -> list[int]:
    """Every absolute layer `mon` reads."""

    return sorted({mon.layer, *(getattr(mon, "layers_used", None) or [])})


class ScalarMonitor:
    """Base of the single-concept zoo monitors: one scalar threshold `thr` and calibration scale
    `scale_` (the SD of held-out negative scores; p = sigmoid((s - thr) / scale)). Subclasses set
    name / layer / concept and implement logits() and to()."""

    needs_tokens = False
    layers_used: list[int] | None = None
    scale_: float = 1.0

    def scale(self, concepts=None):
        return torch.tensor([self.scale_])

    def _keep(self, new):  # .to() copies keep the calibration scale
        new.scale_ = self.scale_
        return new

    def covers(self, concept) -> bool:
        return concept == self.concept

    def thresholds(self, concepts=None):
        return torch.tensor([self.thr])

    def normalize(self, t, concepts=None):
        return t

    def state(self) -> dict:
        """A "zoo" state: the CPU copy of the monitor itself (torch.save pickles it). crypsis adds
        it so every kind round-trips through MonitorPool; the snapshot could only save concept /
        meta monitors."""

        return {"kind": "zoo", "cls": type(self).__name__, "monitor": self.to("cpu")}


def monitor_from_state(st: dict):
    """A monitor from its `state()` dict (the snapshot's "concept" / "meta" states, plus the
    "zoo" states crypsis adds for every other kind)."""

    from crypsis.probes.kinds import STATE_LOADERS

    try:
        loader = STATE_LOADERS[st["kind"]]
    except KeyError:
        raise ValueError(f"unknown monitor kind {st['kind']!r}") from None
    return loader(st)


# ------------------------------------------------------------------------------------------ pool
@dataclass
class PoolEntry:
    monitor: Any  # a Monitor
    round: int
    family: str  # e.g. "concept_linear", "meta_linear", "panel"
    held_out: bool = False  # True: evaluation only, never sampled for training feedback


# pool families that are never an episode monitor, selectivity readout or sampled monitor
AUX_FAMILIES = ("panel", "turn0_panel")


def _is_concept_bank(mon: Any) -> bool:
    from crypsis.probes.kinds.linear import ConceptProbeBank

    return isinstance(mon, ConceptProbeBank)


class MonitorPool:
    """The growing monitor set of the outer loop. Monitors are only ever ADDED."""

    def __init__(self, entries: list[PoolEntry] | None = None) -> None:
        self.entries: list[PoolEntry] = list(entries or [])

    def add(self, monitor, *, round: int, family: str, held_out: bool = False) -> None:
        if any(e.monitor.name == monitor.name for e in self.entries):
            raise ValueError(f"monitor {monitor.name!r} already in the pool")
        self.entries.append(PoolEntry(monitor, round, family, held_out))

    def get(self, name: str):
        for e in self.entries:
            if e.monitor.name == name:
                return e.monitor
        raise KeyError(name)

    def trainable(self, concept: str | None = None) -> list[PoolEntry]:
        """Non-held-out, non-auxiliary entries covering `concept` ("panel" = fixed hide targets
        only; "turn0_panel" = probes that only judge turn 0)."""

        return [e for e in self.entries if not e.held_out and e.family not in AUX_FAMILIES
                and e.monitor.covers(concept)]

    def sample(self, rng: random.Random, concept: str | None) -> PoolEntry:
        """Uniform over trainable monitors that cover `concept`."""

        cands = self.trainable(concept)
        if not cands:
            raise LookupError(f"no trainable monitor covers {concept!r}")
        return rng.choice(cands)

    def concept_monitor(self, concept: str):
        """The earliest trainable ConceptProbeBank covering `concept` (selectivity readouts)."""

        for e in self.entries:
            if (
                _is_concept_bank(e.monitor)
                and not e.held_out
                and e.family not in AUX_FAMILIES
                and e.monitor.covers(concept)
            ):
                return e.monitor
        raise LookupError(f"no concept bank covers {concept!r}")

    def concept_monitors(self, concept: str) -> list:
        """Every trainable ConceptProbeBank covering `concept`, earliest first."""

        return [
            e.monitor
            for e in self.entries
            if _is_concept_bank(e.monitor)
            and not e.held_out
            and e.family not in AUX_FAMILIES
            and e.monitor.covers(concept)
        ]

    def panel(self, concept: str) -> list[str]:
        """Names of the depth-panel probes (family "panel") covering `concept`."""

        return [e.monitor.name for e in self.entries
                if e.family == "panel" and e.monitor.covers(concept)]

    @property
    def layers(self) -> list[int]:
        """Every layer a monitor reads (multi-layer monitors: all of their layers_used)."""
        out = set()
        for e in self.entries:
            out.add(e.monitor.layer)
            out.update(getattr(e.monitor, "layers_used", None) or [])
        return sorted(out)

    @property
    def token_layers(self) -> list[int]:
        """Layers whose PER-TOKEN states some monitor needs (token-pooled probes)."""
        out = set()
        for e in self.entries:
            if getattr(e.monitor, "needs_tokens", False):
                out.add(e.monitor.layer)
                out.update(getattr(e.monitor, "layers_used", None) or [])
        return sorted(out)

    def to(self, device) -> MonitorPool:
        return MonitorPool(
            [PoolEntry(e.monitor.to(device), e.round, e.family, e.held_out) for e in self.entries]
        )

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        manifest = []
        for e in self.entries:
            torch.save(e.monitor.state(), d / f"{e.monitor.name}.pt")
            manifest.append(
                {
                    "name": e.monitor.name,
                    "round": e.round,
                    "family": e.family,
                    "held_out": e.held_out,
                }
            )
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))

    @classmethod
    def load(cls, directory: Path | str) -> MonitorPool:
        d = Path(directory)
        out = cls()
        for m in json.loads((d / "manifest.json").read_text()):
            mon = monitor_from_state(torch.load(d / f"{m['name']}.pt", weights_only=False))
            out.entries.append(PoolEntry(mon, m["round"], m["family"], m["held_out"]))
        return out


def trim_bank(bank: MonitorPool, keep_base: set[str], cap: int) -> list[str]:
    """Drop the oldest online-refit monitors so the bank has at most `cap` entries; never drop a
    base member. Returns the names dropped."""

    if len(bank.entries) <= cap:
        return []
    refits = [
        e
        for e in bank.entries
        if e.monitor.name not in keep_base and e.monitor.meta.get("online_refit")
    ]
    refits.sort(key=lambda e: e.round)  # oldest first
    dropped = []
    while len(bank.entries) > cap and refits:
        e = refits.pop(0)
        bank.entries.remove(e)
        dropped.append(e.monitor.name)
    return dropped
