"""TRAINING-RUN REPORT (REPLICATION 2026-10-18d): one JSON of every organism-level curve we read off
a TICA training run's logs (adaptive/{train_log,turns,refits}.jsonl), across resume chunks.

Chunks: <run>/adaptive (global offset = base_offset; a branch launched with TICA_STEP_OFFSET=360
passes 360) and <run>/resume_N/adaptive (offset N; an extension's first chunk may be resume_420
with no adaptive/). A chunk's rows past the next chunk's offset are superseded (that chunk resumed
from the checkpoint at the offset). Global step: train_log step + offset; turns / refits rows carry
the 0-based step index of the rollout -> + offset + 1 (= the train_log step they fed).

Sections (by window of global steps): probe index (k1_first / k1_switch / k2..), retry pass,
per-turn hazard per bank, held-out static-probe detection by condition, refit readability (kept /
AUROC),
per-concept and per-family pass after the first refit, per-kind beat rates, step timing.
Aggregates only.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DODGE = ("real_score", "real_verdict")


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a chunk killed mid-write
                continue
    return out


def chunks(run: Path, base_offset: int = 0) -> list[tuple[int, Path]]:
    cs = []
    if (run / "adaptive").is_dir():
        cs.append((base_offset, run / "adaptive"))
    for d in run.glob("resume_*/adaptive"):
        m = re.match(r"resume_(\d+)$", d.parent.name)
        if m:
            cs.append((int(m.group(1)), d))
    return sorted(cs)


def load_run(run: Path | str, base_offset: int = 0) -> dict[str, list[dict]]:
    """{"train": [...], "turns": [...], "refits": [...]} with a global step "g" on every row."""

    run = Path(run)
    cs = chunks(run, base_offset)
    out: dict[str, list[dict]] = {"train": [], "turns": [], "refits": []}
    for j, (off, d) in enumerate(cs):
        nxt = cs[j + 1][0] if j + 1 < len(cs) else None
        for key, fname, plus in (("train", "train_log.jsonl", 0), ("turns", "turns.jsonl", 1),
                                 ("refits", "refits.jsonl", 1)):
            for r in _jsonl(d / fname):
                if "step" not in r:
                    continue
                g = int(r["step"]) + off + plus
                if nxt is not None and g > nxt:
                    continue
                out[key].append({**r, "g": g})
    return out


def _win(g: int, window: int) -> str:
    w = (g - 1) // window * window
    return f"{w}-{w + window}"


def _wkey(kv):
    return int(kv[0].split("-")[0])


def _r(x, n=3):
    return None if x is None else round(x, n)


def segments(turns: list[dict]) -> dict:
    """(g, ep, concept) -> that concept segment's judged rows (dodge conds, monitored turns), in
    turn order; plus the episode's first concept."""

    segs, first = defaultdict(list), {}
    for r in turns:
        if r.get("cond") not in DODGE or r.get("off"):
            continue
        if r["turn"] == 0:
            first[(r["g"], r["ep"])] = r["concept"]
        segs[(r["g"], r["ep"], r["concept"])].append(r)
    for v in segs.values():
        v.sort(key=lambda r: r["turn"])
    return {"segs": segs, "first": first}


def probe_index(turns: list[dict], window: int = 20, max_k: int = 5,
                hazard_turns: int = 6) -> dict:
    """Pass rate by probe index (bank k = n_deployed), turns to beat, and the per-judged-turn
    hazard of each bank, by window."""

    sg = segments(turns)
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    haz = defaultdict(lambda: defaultdict(lambda: [[0, 0] for _ in range(hazard_turns)]))
    for (g, ep, c), rs in sg["segs"].items():
        w = _win(g, window)
        start, beat = {}, {}
        for r in rs:
            k = min(max_k, max(1, r.get("n_deployed") or 1))
            start.setdefault(k, r["turn"])
            j = r["turn"] - start[k]
            if k not in beat:
                if j < hazard_turns:
                    h = haz[w][f"k{k}"][j]
                    h[0] += 1
                    h[1] += not r["detected"]
                if not r["detected"]:
                    beat[k] = j + 1
        for k in start:
            keys = [f"k{k}"]
            if k == 1:
                keys.append("k1_first" if sg["first"].get((g, ep)) == c else "k1_switch")
            for key in keys:
                a = acc[w][key]
                a[0] += 1
                if k in beat:
                    a[1] += 1
                    a[2] += beat[k]
    order = lambda d: sorted(d.items(), key=lambda kv: int(kv[0].split("-")[0]))  # noqa: E731
    return {
        "by_index": {w: {key: {"beaten": _r(b / n), "n": n, "turns": _r(t / b, 2) if b else None}
                         for key, (n, b, t) in sorted(d.items())} for w, d in order(acc)},
        "hazard": {w: {key: [_r(p / n) if n else None for n, p in hs]
                       for key, hs in sorted(d.items())} for w, d in order(haz)},
    }


def retry_pass(turns: list[dict], window: int = 20) -> dict:
    """Pass rate on a turn whose previous turn was CAUGHT by the same bank (same concept, same
    active probe, same bank size), by window."""

    eps = defaultdict(list)
    for r in turns:
        if r.get("cond") in DODGE:
            eps[(r["g"], r["ep"])].append(r)
    acc = defaultdict(lambda: [0, 0])
    for (g, _), rs in eps.items():
        rs.sort(key=lambda r: r["turn"])
        for a, b in itertools.pairwise(rs):
            if a.get("off") or b.get("off") or b["turn"] != a["turn"] + 1:
                continue
            if (a["concept"] == b["concept"] and a["detected"]
                    and a.get("n_deployed") == b.get("n_deployed")
                    and a.get("active") == b.get("active")):
                x = acc[_win(g, window)]
                x[0] += 1
                x[1] += not b["detected"]
    return {w: {"pass": _r(p / n), "n": n} for w, (n, p) in
            sorted(acc.items(), key=lambda kv: int(kv[0].split("-")[0]))}


def heldout_by_cond(turns: list[dict], window: int = 80) -> dict:
    """Static base probes at the held-out layers (any-of), detection by condition: real (dodging,
    monitored), real_OFF (monitor-off turns), and the no-hide conditions (nobrief / mismatch)."""

    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in turns:
        if r.get("heldout_det") is None:
            continue
        c = "real" if r.get("cond") in DODGE else r.get("cond")
        if r.get("off"):
            c += "_OFF"
        a = acc[_win(r["g"], window)][c]
        a[0] += 1
        a[1] += bool(r["heldout_det"])
    return {w: {c: {"det": _r(d / n), "n": n} for c, (n, d) in sorted(v.items())}
            for w, v in sorted(acc.items(), key=lambda kv: int(kv[0].split("-")[0]))}


def refit_readability(refits: list[dict], window: int = 80) -> dict:
    """Refit gate on the policy's own states: kept fraction, mean AUROC, fraction with AUROC < 0.99
    (the concept getting harder to read), by window and by refit layer."""

    def summ(rs):
        au = [r["auroc"] for r in rs if r.get("auroc") is not None]
        return {"n": len(rs), "kept": _r(sum(bool(r.get("kept")) for r in rs) / len(rs)),
                "auroc": _r(sum(au) / len(au), 4) if au else None,
                "frac_lt_099": _r(sum(a < 0.99 for a in au) / len(au)) if au else None}

    by_w, by_wl = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in refits:
        w = _win(r["g"], window)
        by_w[w].append(r)
        m = re.search(r"@(\d+)$", str(r.get("kind") or ""))
        by_wl[w][m.group(1) if m else str(r.get("kind"))].append(r)
    key = lambda kv: int(kv[0].split("-")[0])  # noqa: E731
    return {"by_window": {w: summ(rs) for w, rs in sorted(by_w.items(), key=key)},
            "by_window_layer": {w: {L: summ(rs) for L, rs in sorted(d.items())}
                                for w, d in sorted(by_wl.items(), key=key)},
            "reject_reasons": dict(sorted(Counter(str(r.get("why"))[:40] for r in refits
                                                  if not r.get("kept")).items()))}


def per_concept(turns: list[dict], window: int = 80, families: dict | None = None) -> dict:
    """Per concept (and family): fraction of segments that beat the first probe, and that beat a
    REFIT (bank k >= 2) once faced, by window."""

    sg = segments(turns)
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # k1 n/beat, k2+ n/beat
    for (g, _, c), rs in sg["segs"].items():
        w = _win(g, window)
        b1 = any(not r["detected"] for r in rs if (r.get("n_deployed") or 1) <= 1)
        n1 = any((r.get("n_deployed") or 1) <= 1 for r in rs)
        later = [r for r in rs if (r.get("n_deployed") or 1) >= 2]
        for key in [c, *([f"family:{families[c]}"] if families and c in families else [])]:
            a = acc[key][w]
            a[0] += n1
            a[1] += n1 and b1
            if later:
                a[2] += 1
                a[3] += any(not r["detected"] for r in later)
    return {k: {w: {"k1_beaten": _r(b1 / n1) if n1 else None, "k1_n": n1,
                    "refit_beaten": _r(b2 / n2) if n2 else None, "refit_n": n2}
                for w, (n1, b1, n2, b2) in sorted(d.items(), key=_wkey)}
            for k, d in sorted(acc.items())}


def kind_beat(train: list[dict], window: int = 80) -> dict:
    """Per refit kind (arch@layer): n-weighted beat fraction from the train log's beat_frac_*."""

    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for r in train:
        w = _win(r["g"], window)
        for k, v in r.items():
            if k.startswith("rollout/beat_frac_") and isinstance(v, (int, float)):
                kind = k[len("rollout/beat_frac_"):]
                n = r.get(f"rollout/beat_n_{kind}") or 0
                if n:
                    a = acc[kind][w]
                    a[0] += v * n
                    a[1] += n
    return {k: {w: {"beaten": _r(b / n), "n": n} for w, (b, n) in
                sorted(d.items(), key=lambda kv: int(kv[0].split("-")[0]))}
            for k, d in sorted(acc.items())}


def timing(train: list[dict], window: int = 80) -> dict:
    keys = ("sec", "sec_rollout", "sec_train", "rollout/sec_ep_refit_read",
            "rollout/sec_ep_refit_fit", "sec_cap_gen", "max_mem_gb")
    acc = defaultdict(lambda: defaultdict(list))
    for r in train:
        for k in keys:
            if isinstance(r.get(k), (int, float)):
                acc[_win(r["g"], window)][k].append(r[k])
    return {w: {k.split("/")[-1]: _r(sum(v) / len(v), 1) for k, v in d.items()}
            for w, d in sorted(acc.items(), key=lambda kv: int(kv[0].split("-")[0]))}


def report(run: Path | str, base_offset: int = 0, window: int = 20, coarse: int = 80,
           families: dict | None = None) -> dict:
    d = load_run(run, base_offset)
    gs = [r["g"] for r in d["train"]]
    return {"run": str(run), "base_offset": base_offset,
            "chunks": [[o, str(p)] for o, p in chunks(Path(run), base_offset)],
            "steps": [min(gs), max(gs)] if gs else None,
            "probe_index": probe_index(d["turns"], window),
            "retry_pass": retry_pass(d["turns"], window),
            "heldout_static": heldout_by_cond(d["turns"], coarse),
            "refit_readability": refit_readability(d["refits"], coarse),
            "per_concept": per_concept(d["turns"], coarse, families),
            "kind_beat": kind_beat(d["train"], coarse),
            "timing": timing(d["train"], coarse)}
