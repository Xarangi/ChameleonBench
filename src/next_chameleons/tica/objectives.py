"""Training objectives for teacher-forced TICA episodes (Level 1; DESIGN-TICA.md).

NC-style loss through frozen monitors. Each (episode, turn, readout) gets a ROLE:
- HIDE: push the monitored probe below its threshold. Applies at turns >= hide_from_turn of
  episodes that open with the monitor brief and show monitor feedback.
- MATCH: keep the probe score where the BASE model has it (no grad through the reference). Used for:
  - the PIN: turn 0 of every brief episode, so evasion cannot start before any feedback;
  - nobrief episodes: no monitor, no change;
  - neg_text episodes: the false-positive rate is preserved;
  - the text concept's own probes (every covering bank) in mismatch episodes: selectivity, i.e.
    hide only what is monitored.
- LM: mean NLL of the forced text tokens, so the model can still produce the text (NC's
  behaviour term).
Probe outputs pass through p = sigmoid((raw - thr) / scale), which is 0.5 at the 1%-FPR threshold;
scale = the monitor's negative-score SD (Monitor.scale). Without it the sigmoid is saturated at
gated positives (12-16 raw logits above thr, sigmoid' ~ 1e-6) and HIDE has no gradient (review
10-03). HIDE uses MSE(p, 0); MATCH uses MSE(p, p_ref). The MSE form follows our NC finding: MSE
beats a margin loss (REPLICATION 08-07c).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from next_chameleons.tica.episodes import CONDITION_SPECS, Trajectory


@dataclass
class ObjectiveConfig:
    hide_weight: float = 1.0
    match_weight: float = 1.0
    lm_weight: float = 0.1
    # Multi-turn credit (10-10). improve_weight penalises a turn whose probe value is WORSE than
    # the previous turn's, so the model is paid for progress at every turn rather than for one
    # correction (SCoRe's 2-attempt reward collapses to "best first attempt, then barely edit").
    # The previous value is a constant from the rollout, so the gradient flows only through p_t.
    improve_weight: float = 0.0
    # 10-17ad: demand at least this much progress (p_t <= prev_p - improve_margin) instead of only
    # "not worse"; 0.0 = the original term. prev_p exists only when the same active probe judged the
    # previous turn, i.e. on retry turns.
    improve_margin: float = 0.0
    hide_from_turn: int = 1  # 1 = turn 0 is pinned to the base model
    pin_first_turn: bool = True
    pin_segment_first: bool = False  # also pin the first turn of a newly switched-to concept
    # Monitor-OFF turns (REPLICATION 2026-10-17o): > 0 = their match-to-base targets get role
    # "off", their OWN mean and this weight, instead of sharing the "match" mean with the no-brief /
    # neg-text controls (~14% of the match targets there; hiding leaked into OFF turns).
    off_match_weight: float = 0.0
    # Retry-weighted hide (REPLICATION 2026-10-17ad; default 1.0 = off): weight of the ACTIVE
    # probe's hide target on a RETRY turn (the same probe caught the previous turn), relative to
    # 1.0 for every other hide target -- the turns where the search must produce a NEW move.
    retry_hide_weight: float = 1.0
    # Retry LADDER (10-18; default () = off, overrides retry_hide_weight when set): the active
    # probe's hide weight after n consecutive catches by that same probe on this concept is
    # ladder[min(n, len) - 1] (n = 0, a first try or a turn after a pass, keeps 1.0). Feedback about
    # the probe accumulates with every catch, so later retries can be weighted more.
    retry_hide_ladder: tuple[float, ...] = ()


@dataclass(frozen=True)
class Target:
    traj: int  # index into the trajectory list
    turn: int
    key: str  # "monitored" or a spec.probes key
    monitor: str
    concept: str | None
    role: str  # "hide" | "match"
    prev_p: float | None = None  # the same readout's p on the previous turn, if any
    retry: bool = False  # the same active probe caught the previous turn (retry_hide_weight)
    retry_n: int = 0  # consecutive catches by this active probe right before this turn (ladder)


def plan_targets(trajs: list[Trajectory], cfg: ObjectiveConfig) -> list[Target]:
    out = []
    for i, tr in enumerate(trajs):
        s = tr.spec
        cs = CONDITION_SPECS[s.cond]
        for t in range(len(tr.convs)):
            mon_role = None
            if s.cond in ("nobrief", "neg_text"):
                mon_role = "match"
            elif s.cond == "mismatch":
                mon_role = None  # Z is absent from the text; nothing to hide or keep
            elif cs["brief"] and cs["show"] is not None:
                tc = s.meta.get("turn_concepts")  # hidden mode: hide only on the monitored turns
                if t >= cfg.hide_from_turn and (tc is None or tc[t] == s.monitored):
                    mon_role = "hide"
                elif cfg.pin_first_turn or (tc is not None and t >= cfg.hide_from_turn):
                    mon_role = "match"
            if mon_role:
                prev = tr.results[t - 1]["monitored"].get("p") if t > 0 else None
                out.append(
                    Target(i, t, "monitored", s.monitor, s.monitored, mon_role, prev_p=prev)
                )
            for key, (name, concept) in s.probes.items():
                # "text" / "text@bank": the text's own concept in mismatch episodes; "keep": the
                # non-monitored concept of a paired (hidden-mode) text. Both are selectivity MATCH
                # targets. (The eval-only "anchor" never reaches the trainer.)
                if key.split("@")[0] in ("text", "keep"):
                    out.append(Target(i, t, key, name, concept, "match"))
    return out


def probe_prob(
    pool, pooled: dict[int, torch.Tensor], rows: list[int], targets: list[Target],
    tokens: dict | None = None,
) -> torch.Tensor:
    """p = sigmoid((raw - thr) / scale) for each target; rows[k] indexes pooled for targets[k].
    Token-level monitors (probe_zoo: needs_tokens) read tokens[layer][row] (per-span [n, d] states);
    multi-layer monitors (layers_used) read pooled at each of their layers."""

    by: dict[str, list[int]] = {}
    for k, tg in enumerate(targets):
        by.setdefault(tg.monitor, []).append(k)
    parts = {}
    for name, ks in by.items():
        mon = pool.get(name)
        cs = [targets[k].concept for k in ks]
        idx = [rows[k] for k in ks]
        if getattr(mon, "needs_tokens", False) and getattr(mon, "layers_used", None):
            x = {L: [(tokens[L] if tokens is not None else pooled[("tok", L)])[i] for i in idx]
                 for L in mon.layers_used}
        elif getattr(mon, "needs_tokens", False):  # tokens, or pooled[("tok", L)] (trainer)
            src = tokens[mon.layer] if tokens is not None else pooled.get(("tok", mon.layer))
            if src is None:
                raise ValueError(f"monitor {name} needs per-token states")
            x = [src[i] for i in idx]
        elif getattr(mon, "layers_used", None):
            x = {L: pooled[L][idx] for L in mon.layers_used}
        else:
            x = pooled[mon.layer][idx]
        raw = mon.logits(x, cs)
        z = (raw - mon.thresholds(cs).to(raw.device)) / mon.scale(cs).to(raw.device)
        parts[name] = (ks, torch.sigmoid(z))
    vals = [None] * len(targets)
    for ks, p in parts.values():
        for j, k in enumerate(ks):
            vals[k] = p[j]
    return torch.stack(vals)


def loss_terms(
    p: torch.Tensor,
    p_ref: torch.Tensor,
    roles: list[str],
    nll: torch.Tensor | None,
    cfg: ObjectiveConfig,
    totals: dict[str, int] | None = None,
    prev_p: torch.Tensor | None = None,
    hide_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """p: policy probe probs [k] (with grad); p_ref: reference probs [k]; nll: [n_views] or None.

    The objective is hide_w * mean over ALL hide targets + match_w * mean over ALL match targets +
    lm_w * mean over ALL views. With `totals` ({"hide", "match", "views"}: counts over the whole
    step), a chunk returns its share of those means (sums / totals), so summing chunk gradients
    gives exactly the full-batch gradient whatever the chunk composition. Without `totals` the
    chunk is the whole batch. Stats are SUMS (hide_sq, hide_p, match_sq, nll) plus counts, so the
    caller can form exact means.
    """

    hide = torch.tensor([r == "hide" for r in roles], device=p.device, dtype=torch.bool)
    # "motion" targets carry no probe loss (only the caller's motion terms): neither hide nor match.
    match = torch.tensor([r == "match" for r in roles], device=p.device, dtype=torch.bool)
    off = torch.tensor([r == "off" for r in roles], device=p.device, dtype=torch.bool)
    n_h, n_m, n_o = int(hide.sum()), int(match.sum()), int(off.sum())
    n_v = int(nll.numel()) if nll is not None else 0
    t = totals or {"hide": n_h, "match": n_m, "off": n_o, "views": n_v}
    zero = p.sum() * 0.0
    # Turn-weighted hide (optional): weight each hide target's p^2 so early misses cost little and
    # late turns cost a lot ("fail early, dodge late" iteration, not the static dodge-from-turn-1 a
    # flat loss optimises for). Normalised by the step-total hide weight for exact grad accum.
    if hide_w is not None and n_h:
        s_hide = (hide_w[hide] * p[hide] ** 2).sum()
        hide_denom = t.get("hide_w_sum", float(hide_w[hide].sum()))
    else:
        s_hide = (p[hide] ** 2).sum() if n_h else zero
        hide_denom = t["hide"]
    s_match = ((p[match] - p_ref[match]) ** 2).sum() if n_m else zero
    s_off = ((p[off] - p_ref[off]) ** 2).sum() if n_o else zero
    s_lm = nll.sum() if nll is not None else zero
    # Progress term: a hide turn is penalised for being WORSE than the same episode's previous
    # turn. prev_p is a constant from the rollout, so the gradient flows only through p_t.
    s_imp = zero
    if cfg.improve_weight and prev_p is not None and n_h:
        have = hide & torch.isfinite(prev_p)
        if int(have.sum()):
            s_imp = torch.relu(p[have] - prev_p[have] + cfg.improve_margin).sum()
    total = (
        cfg.hide_weight * s_hide / max(1.0, hide_denom)
        + cfg.match_weight * s_match / max(1, t["match"])
        + cfg.off_match_weight * s_off / max(1, t.get("off", n_o))
        + cfg.improve_weight * s_imp / max(1, t["hide"])
        + cfg.lm_weight * s_lm / max(1, t["views"])
    )
    stats = {
        "hide_sq": float(s_hide.detach()),
        "hide_p": float(p[hide].sum().detach()) if n_h else 0.0,
        "match_sq": float(s_match.detach()),
        "off_sq": float(s_off.detach()),
        "n_off": n_o,
        "improve": float(s_imp.detach()),
        "nll": float(s_lm.detach()),
        "n_hide": n_h,
        "n_match": n_m,
        "n_views": n_v,
    }
    return total, stats


def summarize_stats(acc: dict[str, float]) -> dict[str, float]:
    """Exact step means from summed chunk stats."""

    nh, nm, nv = acc.get("n_hide", 0), acc.get("n_match", 0), acc.get("n_views", 0)
    nan = float("nan")
    return {
        "hide_mse": acc["hide_sq"] / nh if nh else nan,
        "improve": acc.get("improve", 0.0) / nh if nh else nan,
        "hide_p": acc["hide_p"] / nh if nh else nan,
        "match_mse": acc["match_sq"] / nm if nm else nan,
        "off_mse": acc.get("off_sq", 0.0) / acc["n_off"] if acc.get("n_off") else nan,
        "nll": acc["nll"] / nv if nv else nan,
        "n_hide": nh,
        "n_match": nm,
    }


def retry_weight(cfg: ObjectiveConfig, tg: Target) -> float:
    """Hide-loss weight of a target from the retry settings (1.0 when both are off)."""

    if cfg.retry_hide_ladder:
        n = tg.retry_n
        return float(cfg.retry_hide_ladder[min(n, len(cfg.retry_hide_ladder)) - 1]) if n else 1.0
    return cfg.retry_hide_weight if tg.retry else 1.0
