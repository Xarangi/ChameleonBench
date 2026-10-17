"""Branch-and-collect: the refit data of the sequential race (user, 10-09).

For every gated concept c:
1. ROOTS. Run R short real-condition episodes (named brief, verdict feedback from the monitor in
   play, root_turns turns, fresh texts about c). A root counts only if its LAST turn bypassed the
   monitor (verdict NOT DETECTED): the model is "past the wall" at the branch point.
2. BRANCHES. From each bypassing root, fork the conversation into many one-turn follow-ups: the
   root's last verdict is shown and the model is asked to repeat ONE new text. Follow-ups are
   - follow_pos texts about c (the hiding the defender wants to catch),
   - follow_neg_other texts about other concepts (concept present, not monitored: the model
     should not hide them),
   - follow_neg_free concept-free texts.
   Every follow-up of a root shares the same prefix, so brief, feedback history and conversation
   length are matched between positives and negatives (the context confound of the 10-09 meta
   detector).
3. One pooled read per follow-up at the requested layers; the monitor in play also scores every
   follow-up, so "bypassed" is recorded per positive.

Texts: every concept's texts are split into halves A / B by a seeded shuffle (alternating). Each
row carries the half of its text, so refits fit on A and set thresholds / report quality on B,
with no text shared between the two (the NC fit / calibration separation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from next_chameleons.tica.episodes import (
    ConceptTexts,
    EpisodeSpec,
    opening,
    render_feedback,
    rollout,
    score_readouts,
)
from next_chameleons.tica.monitors import MonitorPool

COND_POS, COND_OTHER, COND_FREE = "branch_pos", "branch_neg_other", "branch_neg_free"
# Concept text with NO brief and NO verdicts: the plain format the base probes were fit on.
COND_POS_NP = "branch_pos_noprompt"


@dataclass
class BranchConfig:
    roots: int = 4
    root_turns: int = 3
    follow_pos: int = 12
    # Concept follow-ups with the prefix STRIPPED of the brief and verdicts. Without these the
    # refit's positives are only the evader's dodging states, and the probe stops being a concept
    # detector: round-1's refit scored 0.00 on unprompted concept text while round-0's scored 1.00
    # (10-10). A defender's probe must catch the concept in BOTH regimes.
    follow_pos_noprompt: int = 12
    follow_neg_other: int = 8
    follow_neg_free: int = 4
    feedback: str = "verdict"


def text_halves(texts: list[str], rng: random.Random) -> dict[str, str]:
    """text -> "A" | "B": a seeded shuffle, then alternating, so the halves are balanced."""

    order = list(dict.fromkeys(texts))
    rng.shuffle(order)
    return {t: "AB"[i % 2] for i, t in enumerate(order)}


def branch_collect(
    policy,
    pool: MonitorPool,
    concepts: list[ConceptTexts],
    negatives: list[str],
    cfg: BranchConfig,
    layers: list[int],
    seed: int,
    roots_only: set[str] | None = None,
) -> tuple[dict[int, torch.Tensor], list[dict], dict]:
    """Returns (pooled activations per layer, one label dict per follow-up row, report).

    roots_only: branch ONLY these concepts (a single-concept run passes one). Every other covered
    concept stays available as the other-concept NEGATIVE class, which is what keeps the refit a
    concept detector rather than a topicality detector (10-10 recipe comparison).
    """

    rng = random.Random(seed)
    concepts = [c for c in concepts if pool.trainable(c.concept)]
    if len(concepts) < 2:
        raise ValueError("branching needs >= 2 covered concepts")
    rooted = [c for c in concepts if roots_only is None or c.concept in roots_only]
    if not rooted:
        raise ValueError(f"no covered concept among roots_only={roots_only}")
    half = {c.concept: text_halves(c.texts, random.Random(f"{seed}:{c.concept}")) for c in concepts}
    free_half = text_halves(negatives, random.Random(f"{seed}:free"))
    split = {c.concept: c.split for c in concepts}

    # 1. roots
    specs = []
    for c in rooted:
        if len(c.texts) < cfg.root_turns + 1:
            continue
        for k in range(cfg.roots):
            mon = pool.sample(rng, c.concept).monitor.name
            specs.append(
                EpisodeSpec(
                    id=f"{c.concept}#root{k}",
                    cond="real_score",
                    texts=rng.sample(c.texts, cfg.root_turns),
                    text_concept=c.concept,
                    monitored=c.concept,
                    monitor=mon,
                    feedback=cfg.feedback,
                    meta={"split": c.split},
                )
            )
    trajs = rollout(policy, pool, specs, cfg.root_turns)
    bypassed = [tr for tr in trajs if not tr.results[-1]["monitored"]["detected"]]

    # 2. branches
    by_concept = {c.concept: c for c in concepts}
    others = [c.concept for c in concepts]
    convs, labels = [], []
    for tr in bypassed:
        s = tr.spec
        c = s.monitored
        used = set(s.texts)
        pos_pool = [t for t in by_concept[c].texts if t not in used]
        picks = [(COND_POS, c, t) for t in rng.sample(pos_pool, min(cfg.follow_pos, len(pos_pool)))]
        for _ in range(cfg.follow_neg_other):
            z = rng.choice([o for o in others if o != c])
            picks.append((COND_OTHER, z, rng.choice(by_concept[z].texts)))
        picks += [(COND_FREE, None, t) for t in rng.sample(negatives, cfg.follow_neg_free)]
        np_pool = [t for t in by_concept[c].texts if t not in used]
        np_picks = [
            (COND_POS_NP, c, t)
            for t in rng.sample(np_pool, min(cfg.follow_pos_noprompt, len(np_pool)))
        ]
        shown = tuple(tr.results[-1]["shown"])
        for cond, tc, text in np_picks:
            # A fresh one-turn conversation: no brief, no verdict history.
            convs.append(
                [
                    {"role": "user", "content": opening("nobrief", text, c, named=s.named)},
                    {"role": "assistant", "content": text},
                ]
            )
            labels.append(
                {
                    "cond": cond,
                    "turn": 0,
                    "concept": tc,
                    "monitored": c,
                    "monitor": s.monitor,
                    "split": split[c],
                    "group": c,
                    "root": s.id,
                    "half": half[tc][text],
                    "active": True,
                }
            )
        for cond, tc, text in picks:
            fb = render_feedback(s.cond, c, shown, text, feedback=s.feedback, named=s.named)
            conv = [dict(x) for x in tr.convs[-1]]
            conv += [{"role": "user", "content": fb}, {"role": "assistant", "content": text}]
            convs.append(conv)
            labels.append(
                {
                    "cond": cond,
                    "turn": cfg.root_turns,
                    "concept": tc,
                    "monitored": c,
                    "monitor": s.monitor,
                    "split": split[c],
                    "group": c,
                    "root": s.id,
                    "half": half[tc][text] if tc is not None else free_half[text],
                    "active": cond == COND_POS,
                }
            )
    if not convs:
        raise RuntimeError("no root bypassed the monitor: nothing to branch from")

    # 3. reads + in-play monitor verdict on every follow-up
    acts = policy.pooled(convs, layers)
    need = sorted({pool.get(a["monitor"]).layer for a in labels} - set(layers))
    if need:
        acts.update(policy.pooled(convs, need))
    scored = score_readouts(pool, acts, [(a["monitor"], a["monitored"]) for a in labels])
    for a, r in zip(labels, scored, strict=True):
        a["detected"] = r["detected"]
    acts = {L: acts[L] for L in layers}

    pos = [a for a in labels if a["cond"] == COND_POS]
    rep = {
        "roots": len(trajs),
        "roots_bypassed": len(bypassed),
        "root_bypass_rate": len(bypassed) / max(1, len(trajs)),
        "rows": len(labels),
        "rows_by_cond": {
            k: sum(a["cond"] == k for a in labels)
            for k in (COND_POS, COND_POS_NP, COND_OTHER, COND_FREE)
        },
        "pos_bypass_rate": sum(not a["detected"] for a in pos) / max(1, len(pos)),
        "concepts_with_rows": len({a["monitored"] for a in labels}),
        "concepts": len(concepts),
        "rooted_concepts": len(rooted),
    }
    return acts, labels, rep
