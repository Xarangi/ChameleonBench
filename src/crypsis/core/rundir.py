"""Run-directory layout and checkpoint discovery (one implementation for training, evals, analysis, runtime).

Layout: <out>/adaptive/step_<k> (first chunk, global = base_offset + k), <out>/resume_<N>/adaptive/
step_<k> (chunk resumed from global step N, global = N + k), <chunk>/final (global = offset + last
train_log step). Reproduces scripts/slurm/tica_resume.sh (newest), tamia_prod_v4c_switch.sh
(newest(prefer_state=False, include_final=False)), tica/run_report.py (chunks / rows) and
tamia_queue_quality.sh ckpt_at (checkpoint_at). Reads no environment variables: a branch's base
offset (formerly TICA_STEP_OFFSET) is an argument.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from crypsis.core.io import read_jsonl

_RESUME = re.compile(r"resume_(\d+)$")
STATE_FILE = "trainer_state.pt"


def _step_of(name: str) -> int | None:
    return int(name[5:]) if name.startswith("step_") and name[5:].isdigit() else None


def _last_logged_step(log: Path) -> int | None:
    """Last train_log row's step, parsed as tica_resume.sh does (text after the first '"step": ')."""

    text = log.read_text().strip() if log.exists() else ""
    if not text:
        return None
    return int(text.splitlines()[-1].split('"step": ')[1].split(",")[0])


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    global_step: int
    has_state: bool


@dataclass(frozen=True)
class RunDir:
    root: Path
    base_offset: int = 0

    def __init__(self, root: str | Path, base_offset: int = 0) -> None:
        object.__setattr__(self, "root", Path(root))
        object.__setattr__(self, "base_offset", int(base_offset))

    def chunks(self) -> list[tuple[int, Path]]:
        """(offset, <chunk>/adaptive) sorted by offset. == run_report.chunks (resume_<digits> only;
        names such as resume_260_k2_aborted are ignored)."""

        cs = []
        if (self.root / "adaptive").is_dir():
            cs.append((self.base_offset, self.root / "adaptive"))
        for d in self.root.glob("resume_*/adaptive"):
            m = _RESUME.match(d.parent.name)
            if m:
                cs.append((int(m.group(1)), d))
        return sorted(cs)

    def _offset_of_chunk_dir(self, adaptive: Path) -> int | None:
        if adaptive.parent == self.root:
            return self.base_offset
        m = _RESUME.match(adaptive.parent.name)
        return int(m.group(1)) if m and adaptive.parent.parent == self.root else None

    def global_step(self, path: str | Path) -> int:
        """Global step of a checkpoint path inside this run (lexical: symlinks are not resolved)."""

        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        off = self._offset_of_chunk_dir(p.parent)
        if off is None:
            raise ValueError(f"{path} is not <run>/[resume_N/]adaptive/<step_k|final> of {self.root}")
        k = _step_of(p.name)
        if k is not None:
            return off + k
        if p.name == "final":
            last = _last_logged_step(p.parent / "train_log.jsonl")
            if last is None:
                raise ValueError(f"{path}: final without a train_log step")
            return off + last
        raise ValueError(f"{path} is not a checkpoint directory")

    def checkpoints(self, include_final: bool = True) -> list[Checkpoint]:
        """Every step_<k> (and finished-chunk final) over all chunks, sorted by (global, state, path)."""

        out = []
        for off, d in self.chunks():
            for c in d.glob("step_*"):
                k = _step_of(c.name)
                if k is not None:
                    out.append(Checkpoint(c, off + k, (c / STATE_FILE).exists()))
            fin = d / "final"
            if include_final and fin.is_dir():
                last = _last_logged_step(d / "train_log.jsonl")
                if last is not None:  # a chunk that ran to its end
                    out.append(Checkpoint(fin, off + last, (fin / STATE_FILE).exists()))
        return sorted(out, key=lambda c: (c.global_step, c.has_state, str(c.path)))

    def newest(self, prefer_state: bool = True, include_final: bool = True) -> Checkpoint | None:
        """Newest checkpoint by global step; at equal steps the copy with trainer_state.pt wins
        (prefer_state), then the larger path string. == the tica_resume.sh heredoc
        (prefer_state=True, include_final=True) and tamia_prod_v4c_switch.sh (False, False)."""

        cks = self.checkpoints(include_final)
        if not cks:
            return None
        key = ((lambda c: (c.global_step, c.has_state, str(c.path))) if prefer_state
               else (lambda c: (c.global_step, str(c.path))))
        return max(cks, key=key)

    def checkpoint_at(self, global_step: int) -> Path | None:
        """The step dir saved at a global step: adaptive/step_g first, then resume chunks in name
        order (first match). == tamia_queue_quality.sh ckpt_at (base_offset 0)."""

        g = int(global_step)
        first = self.root / "adaptive" / f"step_{g - self.base_offset}"
        if first.is_dir():
            return first
        for d in sorted(self.root.glob("resume_*")):
            m = _RESUME.match(d.name)
            if not (m and d.is_dir()):
                continue
            off = int(m.group(1))
            c = d / "adaptive" / f"step_{g - off}"
            if g > off and c.is_dir():
                return c
        return None

    def train_logs(self) -> Iterator[tuple[int, Path]]:
        """(offset, train_log.jsonl) per chunk in offset order (files may be missing)."""

        for off, d in self.chunks():
            yield off, d / "train_log.jsonl"

    def rows(self, fname: str = "train_log.jsonl", plus: int = 0) -> list[dict]:
        """Rows of a per-chunk JSONL with global step "g" = step + offset + plus; rows past the next
        chunk's offset are superseded and dropped. == run_report.load_run (train_log plus=0,
        turns / refits plus=1)."""

        cs = self.chunks()
        out = []
        for j, (off, d) in enumerate(cs):
            nxt = cs[j + 1][0] if j + 1 < len(cs) else None
            for r in read_jsonl(d / fname):
                if "step" not in r:
                    continue
                g = int(r["step"]) + off + plus
                if nxt is not None and g > nxt:
                    continue
                out.append({**r, "g": g})
        return out

    def early_stopped(self) -> bool:
        """tica_resume.sh's check: adaptive/early_stop.json exists and some chunk's says "stop": true."""

        if not (self.root / "adaptive" / "early_stop.json").is_file():
            return False
        files = [self.root / "adaptive" / "early_stop.json",
                 *sorted(self.root.glob("resume_*/adaptive/early_stop.json"))]
        return any('"stop": true' in f.read_text() for f in files if f.is_file())
