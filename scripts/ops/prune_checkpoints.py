#!/usr/bin/env python3
"""Thin old tica run checkpoints while keeping a trend-analysis grid (stdlib only; run on a CPU node).

Keeps, per run: every adapter whose GLOBAL step is a multiple of --every (default 50), each chunk's newest
checkpoint, the run's final adapters, every resume base (the global step a resume_N chunk started from),
and the pre-cooldown steps (--keep-steps, default 1640,1650). Optimizer states (trainer_state.pt) are kept
only for each chunk's newest checkpoint, the resume bases, and the newest state at or before the cooldown
start (--cooldown, default 1650); everything else is deleted. Files with more than one hard link (e.g.
protected copies under <run>/keep/) are never touched. Directories named like resume_N_<suffix> (aborted
chunks) are left alone.

  prune_checkpoints.py --runs-root R --skip a,b --manifest out.tsv           # dry run: writes the manifest
  prune_checkpoints.py --delete out.tsv                                      # deletes exactly the manifest
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

STEP = re.compile(r"step_(\d+)$")
CHUNK = re.compile(r"resume_(\d+)$")


def checkpoints(run: Path):
    """Yield (global_step, chunk_name, step_dir) for adaptive/step_k and resume_N/adaptive/step_k."""
    for chunk in [run, *sorted(run.glob("resume_*"))]:
        if chunk is run:
            off, name = 0, "adaptive"
        else:
            m = CHUNK.match(chunk.name)
            if not m:
                continue  # aborted / renamed chunk: leave alone
            off, name = int(m.group(1)), chunk.name
        ad = chunk / "adaptive"
        if not ad.is_dir():
            continue
        for d in ad.iterdir():
            m = STEP.match(d.name)
            if m and d.is_dir():
                yield off + int(m.group(1)), name, d


def plan(run: Path, every: int, keep_steps: set[int], cooldown: int):
    cks = list(checkpoints(run))
    if not cks:
        return []
    bases = {int(CHUNK.match(c.name).group(1)) for c in run.glob("resume_*") if CHUNK.match(c.name)}
    newest_in_chunk = {}
    for g, ch, d in cks:
        if g > newest_in_chunk.get(ch, (-1, None))[0]:
            newest_in_chunk[ch] = (g, d)
    newest = {d for _, d in newest_in_chunk.values()}
    with_state = [(g, d) for g, _, d in cks if (d / "trainer_state.pt").exists()]
    pre = [x for x in with_state if x[0] <= cooldown]
    pre_cool_state = max(pre, key=lambda x: x[0])[1] if pre else None
    out = []
    for g, _, d in cks:
        keep_adapter = (g % every == 0) or g in keep_steps or g in bases or d in newest
        keep_state = d in newest or g in bases or d == pre_cool_state
        for f in d.iterdir():
            if not f.is_file():
                continue
            st = f.stat()
            if st.st_nlink > 1:
                continue
            if f.name == "trainer_state.pt":
                if not keep_state:
                    out.append((f, st.st_size, g))
            elif not keep_adapter:
                out.append((f, st.st_size, g))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root")
    ap.add_argument("--skip", default="", help="comma list of run dir names (or prefixes ending in *) to skip")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--keep-steps", default="1640,1650")
    ap.add_argument("--cooldown", type=int, default=1650)
    ap.add_argument("--manifest")
    ap.add_argument("--delete", help="manifest to delete (paths in column 3)")
    a = ap.parse_args()

    if a.delete:
        n = tot = 0
        for line in open(a.delete):
            size, _g, path = line.rstrip("\n").split("\t")
            p = Path(path)
            if p.is_file() and p.stat().st_nlink == 1:
                p.unlink()
                n += 1
                tot += int(size)
        for line in open(a.delete):  # remove step dirs left empty
            d = Path(line.rstrip("\n").split("\t")[2]).parent
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        print(f"deleted {n} files, {tot / 2**30:.1f} GiB")
        return

    skip = [s for s in a.skip.split(",") if s]
    keep_steps = {int(x) for x in a.keep_steps.split(",") if x}
    per_run = defaultdict(lambda: [0, 0, 0])
    with open(a.manifest, "w") as mf:
        for run in sorted(Path(a.runs_root).iterdir()):
            if not run.is_dir() or any(run.name == s or (s.endswith("*") and run.name.startswith(s[:-1]))
                                       for s in skip):
                continue
            for f, size, g in plan(run, a.every, keep_steps, a.cooldown):
                mf.write(f"{size}\t{g}\t{f}\n")
                r = per_run[run.name]
                r[0 if f.name == "trainer_state.pt" else 1] += size
                r[2] += 1
    tot = 0
    for name, (s, ad, n) in sorted(per_run.items(), key=lambda x: -(x[1][0] + x[1][1])):
        tot += s + ad
        print(f"{name:45s} states {s / 2**30:6.1f} GiB  adapters/other {ad / 2**30:6.1f} GiB  files {n}")
    print(f"TOTAL to delete: {tot / 2**30:.1f} GiB -> manifest {a.manifest}")
    sys.stdout.flush()
    os.chmod(a.manifest, 0o640)


if __name__ == "__main__":
    main()
