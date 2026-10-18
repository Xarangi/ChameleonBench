"""Adaptive defender curriculum (REPLICATION 2026-10-16g/l; user 10-16: pick probe types -- and
topics -- by how much the model struggles with them, with floor / cap so every option keeps being
sampled and re-measured). Default off (episode_kind_sampling / concept_sampling = "thompson").

Evidence model (10-16l, replaces the 3-turn horizon): DISCRETE-TIME HAZARD. Every deployed probe
is a turns-to-beat record (b = beaten or not, n = judged turns at risk: until the beat, or until
the episode / segment ended = censored). With a constant per-turn beat probability h per arm, the
likelihood is h^B (1 - h)^(N - B) with B = sum b, N = sum n, so Beta(1 + B, 1 + N - B) is the
exact posterior (geometric-Beta conjugacy): a probe beaten in 1 turn is (1, 1) -- strong "easy"
evidence; beaten in 6 turns (1, 6); censored after 10 turns (0, 10). Nothing is dropped and
nothing is thresholded. Non-stationary: B and N decay by gamma before each update (memory ~ 1 / (1
- gamma) updates; discounted Thompson sampling, Raj & Kalyani 2017; Garivier & Moulines 2011).

Sampling: draw h_k ~ Beta(1 + B_k, 1 + N_k - B_k); difficulty d_k = expected turns to beat = 1 /
h_k (capped at max_turns); p_k = floor / K + (1 - floor) * d_k / sum(d), capped at `cap` (water-
filling), so p_k is in [floor / K, cap]. The floor is EXP3-style exploration; the cap bounds any
arm's weight, i.e. the reweighting of a CVaR-type robust objective over arms (group DRO, Sagawa et
al. 2020) rather than the brittle worst case. No "impossible arm" down-weighting (user: the model
should pass all).

FactoredThompson: probe ARCHITECTURE and LAYER as two independent bandits (kind = arch@layer), so
~15 refits per step are not spread over ~40 crossed arms. HierarchicalThompson: concept FAMILY
first, then the concept within it, so hard surface concepts cannot crowd out the topics."""

from __future__ import annotations

import random


class DiscountedThompson:
    def __init__(self, keys, gamma=0.95, floor=0.2, cap=0.3, max_turns=50.0, cap_rel=0.0,
                 power=1.0):
        self.keys = list(keys)
        if not self.keys:
            raise ValueError("no arms")
        self.gamma, self.floor, self.max_turns = gamma, floor, max_turns
        # cap_rel > 0 (10-18; default off): cap = cap_rel x the uniform share 1 / K instead of the
        # absolute cap (which flattened small families: 0.3 <= 1 / K); power > 1 sharpens the
        # difficulty weights (p ~ d ** power).
        self.cap_rel, self.power = cap_rel, power
        self.cap = min(1.0, max(cap, 1.0 / len(self.keys)))
        self.B = {k: 0.0 for k in self.keys}  # discounted beats
        self.N = {k: 0.0 for k in self.keys}  # discounted judged turns at risk

    def update(self, records, key="kind", keyfn=None):
        """Decay, then add records ({key, turns, beaten}); unknown keys are ignored."""

        for k in self.keys:
            self.B[k] *= self.gamma
            self.N[k] *= self.gamma
        for r in records:
            k = keyfn(r) if keyfn else r.get(key)
            if k not in self.B or r.get("turns", 0) <= 0:
                continue
            self.B[k] += 1.0 if r["beaten"] else 0.0
            self.N[k] += float(r["turns"])

    def hazard_draw(self, rng: random.Random) -> dict:
        return {k: rng.betavariate(1.0 + self.B[k], 1.0 + self.N[k] - self.B[k])
                for k in self.keys}

    def probs(self, rng: random.Random, keys=None) -> dict:
        keys = list(keys) if keys is not None else self.keys
        h = self.hazard_draw(rng)
        d = {k: min(1.0 / max(h[k], 1e-9), self.max_turns) ** self.power for k in keys}
        tot = sum(d.values())
        K = len(keys)
        p = {k: self.floor / K + (1 - self.floor) * d[k] / tot for k in keys}
        cap = min(1.0, self.cap_rel / K) if self.cap_rel > 0 else self.cap
        return cap_probs(p, max(cap, 1.0 / K))

    def sample(self, rng: random.Random, exclude=()) -> str | None:
        keys = [k for k in self.keys if k not in set(exclude)]
        if not keys:
            return None
        p = self.probs(rng, keys)
        return rng.choices(keys, [p[k] for k in keys])[0]

    def state(self) -> dict:
        return {k: {"beats": round(self.B[k], 2), "turns": round(self.N[k], 2),
                    "hazard_mean": round((1 + self.B[k]) / (2 + self.N[k]), 3)}
                for k in self.keys}


class FactoredThompson:
    """kind = arch@layer from two independent DiscountedThompson bandits. `archs`: the single-layer
    architectures (placed at a sampled layer); `fixed`: kinds with their own layer set (ensembles),
    sampled as architectures without a layer."""

    def __init__(self, archs, fixed, layers, **kw):
        self.archs, self.fixed, self.layers = list(archs), list(fixed), list(layers)
        self.arch = DiscountedThompson(self.archs + self.fixed, **kw)
        self.layer = DiscountedThompson(self.layers, **kw)

    @staticmethod
    def split(kind):
        a, _, L = kind.partition("@")
        return a, (int(L) if L else None)

    def update(self, records, key="kind"):
        self.arch.update(records, keyfn=lambda r: self.split(r.get(key, ""))[0])
        self.layer.update(records, keyfn=lambda r: self.split(r.get(key, ""))[1])

    def sample(self, rng: random.Random, exclude=()) -> str | None:
        ex = set(exclude)
        for _ in range(8):
            a = self.arch.sample(rng)
            if a is None:
                return None
            k = a if a in self.fixed else f"{a}@{self.layer.sample(rng)}"
            if k not in ex:
                return k
        left = [k for k in self.fixed if k not in ex] + [
            f"{a}@{L}" for a in self.archs for L in self.layers if f"{a}@{L}" not in ex]
        return rng.choice(left) if left else None

    def probs(self, rng: random.Random) -> dict:
        return {**{f"arch_{k}": v for k, v in self.arch.probs(rng).items()},
                **{f"layer_{k}": v for k, v in self.layer.probs(rng).items()}}


class HierarchicalThompson:
    """Concept sampling: FAMILY first (bandit over families), then the concept within its family
    (one bandit per family); p(c) = p(family) p(c | family)."""

    def __init__(self, families: dict, family_kw: dict | None = None, **kw):
        self.fam_of = dict(families)
        fams = sorted(set(self.fam_of.values()))
        # family_kw (10-18): overrides for the FAMILY-level bandit (e.g. its own cap_rel / power)
        self.fam = DiscountedThompson(fams, **{**kw, **(family_kw or {})})
        self.within = {f: DiscountedThompson([c for c, g in self.fam_of.items() if g == f], **kw)
                       for f in fams}

    def update(self, records, key="concept"):
        self.fam.update(records, keyfn=lambda r: self.fam_of.get(r.get(key)))
        for f, b in self.within.items():
            b.update([r for r in records if self.fam_of.get(r.get(key)) == f], key=key)

    def probs(self, rng: random.Random) -> dict:
        pf = self.fam.probs(rng)
        out = {}
        for f, b in self.within.items():
            for c, v in b.probs(rng).items():
                out[c] = pf[f] * v
        return out


def cap_probs(p: dict, cap: float) -> dict:
    """Clip probabilities at `cap` and give the excess to the uncapped entries in proportion to
    their mass (water-filling); keeps the sum at 1 and never lowers an uncapped entry."""

    p = dict(p)
    for _ in range(len(p)):
        over = {k for k, v in p.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(p[k] - cap for k in over)
        for k in over:
            p[k] = cap
        free = {k: v for k, v in p.items() if v < cap - 1e-12}
        tot = sum(free.values())
        if tot <= 0:
            break
        for k, v in free.items():
            p[k] = v + excess * v / tot
    return p
