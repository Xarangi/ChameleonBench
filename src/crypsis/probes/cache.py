"""Content-addressed probe cache: a fitted probe (or None) + info stored under a hash of everything
the fit depends on.

`CacheKey(model_id, data_digest, kind, layers, code_version, extra)` is explicit; its digest is
sha256 of the sorted-key JSON of those fields. `ProbeCache(root)` stores one `<digest>.pt` per key
({"bank", "info", "key"}, written atomically) -- the snapshot's panel-cache file format.

Legacy keys: the snapshot's two caches hashed free-form JSON key dicts built from trainer / policy
state (supervised._panel_cached: sha256(key)[:24] -> `<hex>.pt` with {"bank", "info"};
answer_format.fit_answer_probes: sha256(key)[:16] -> a directory with L{L}.pt, quality.json,
key.json). Those dicts need the plan, the tokenizer and the AdaptiveConfig (higher layers), so
they are built by the callers that own that state; `legacy_digest`, `load_digest` / `save_digest`
and `load_answer_dir` / `save_answer_dir` read and write the old locations unchanged, so existing
caches keep hitting when the caller reproduces its old key JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from crypsis.core.io import write_text_atomic


@dataclass(frozen=True)
class CacheKey:
    """Everything a cached fit depends on. `extra`: any further JSON-able inputs (gate, labels)."""

    model_id: str
    data_digest: str
    kind: str
    layers: tuple[int, ...]
    code_version: str
    extra: dict = field(default_factory=dict, hash=False)

    def to_json(self) -> str:
        d = asdict(self)
        d["layers"] = list(self.layers)
        return json.dumps(d, sort_keys=True, default=str)

    def digest(self, n: int = 24) -> str:
        return legacy_digest(self.to_json(), n)


def legacy_digest(key_json: str, n: int) -> str:
    """sha256 hex prefix of a key's JSON text (the snapshot's rule; n = 24 panel, 16 answer)."""

    return hashlib.sha256(key_json.encode()).hexdigest()[:n]


def data_digest(*tensors: torch.Tensor) -> str:
    """A content digest of fit inputs (shape, dtype and bytes of each tensor, CPU)."""

    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(f"{tuple(t.shape)}{t.dtype}".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()[:24]


class ProbeCache:
    """Probes on disk under `root`, addressed by CacheKey digest (or a legacy digest)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: CacheKey | str) -> Path:
        return self.root / f"{key if isinstance(key, str) else key.digest()}.pt"

    def load(self, key: CacheKey | str) -> tuple[Any, dict] | None:
        """(probe or None, info) on a hit, None on a miss."""

        p = self.path(key)
        if not p.exists():
            return None
        st = torch.load(p, map_location="cpu", weights_only=False)
        return st["bank"], st["info"]

    def save(self, key: CacheKey | str, probe: Any, info: dict) -> Path:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        st = {"bank": probe.to("cpu") if probe is not None else None, "info": info}
        if isinstance(key, CacheKey):
            st["key"] = key.to_json()
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        torch.save(st, tmp)
        os.replace(tmp, p)  # atomic: concurrent jobs never read a partial file
        return p

    # snapshot panel cache: <sha256(key_json)[:24]>.pt
    def load_digest(self, key_json: str, n: int = 24):
        return self.load(legacy_digest(key_json, n))

    def save_digest(self, key_json: str, probe: Any, info: dict, n: int = 24) -> Path:
        return self.save(legacy_digest(key_json, n), probe, info)

    # snapshot answer-probe cache: <sha256(key_json)[:16]>/{L{L}.pt, quality.json, key.json}
    def load_answer_dir(self, key_json: str, layers: Sequence[int]):
        """(monitors per layer, {layer: quality}) on a hit, None on a miss."""

        from crypsis.probes.base import monitor_from_state

        d = self.root / legacy_digest(key_json, 16)
        if not (d / "quality.json").exists():
            return None
        quality = {int(k): v for k, v in json.loads((d / "quality.json").read_text()).items()}
        mons = [monitor_from_state(torch.load(d / f"L{L}.pt", weights_only=False)) for L in layers]
        return mons, quality

    def save_answer_dir(self, key_json: str, layers: Sequence[int], mons: Sequence[Any],
                        quality: dict) -> Path:
        d = self.root / legacy_digest(key_json, 16)
        d.mkdir(parents=True, exist_ok=True)
        for L, m in zip(layers, mons, strict=True):
            torch.save(m.state(), d / f"L{L}.pt")
        write_text_atomic(d / "key.json", key_json)
        write_text_atomic(d / "quality.json", json.dumps(quality))  # last: marks the entry done
        return d
