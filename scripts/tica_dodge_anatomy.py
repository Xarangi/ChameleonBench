"""WHAT does the model change in its activations to dodge a probe that keeps catching it, and
WHERE (10-25, user: "if a model keeps getting caught by a probe, what modification to its
activations does it do exactly to start dodging it? ... where is this hiding exactly?").
CPU only, aggregates only.

Inputs: tica_probe_panel.py --dump-feats files (.pt) of the BASE model and a trained checkpoint,
same seeds / topics (paired on tica_hide_geometry.KEY), plus the trained run's panel JSON next to
each .pt (same stem: records in the same order as the dump rows; per-turn margins of every refit
and panel probe).

D_t = x_trained(t) - x_base(t) on the same forced text = the adapter's edit at turn t (the text
cancels). A STEP = two consecutive monitored hiding turns (real_score, monitor on, same episode):
delta = D_{t+1} - D_t, labelled by the verdict at t (caught / passed). Per layer:

  rel       |delta| / |D_t|: how much the edit changes turn to turn
  gain      <delta, h_c> / |h_c|^2: push along the concept's own hide edit h_c (mean D on its
            hiding turns - mean D on no-brief turns); +0.1 = 10% more of the same edit
  f_hide    <delta, h_c^>^2 / |delta|^2: share of the step along h_c
  f_sub3    |P3 delta|^2 / |delta|^2: share in the top-3 PCs of all hiding edits
  readout   <delta, u_c> / sep_c: change of the base concept readout (u_c = base concept
            direction, sep_c = base separation along it); -0.1 = removes 10% more of the signal
  esc       e = mean delta|caught - mean delta|passed (the feedback-linked modification):
            esc_rel = |e| / mean |D|, esc_abs = |e| / mean |x_base|, esc_cos_hide = cos(e, mean
            hide edit), esc_cos_anti = mean_c cos(e, -u_c)
  feedback_effect  OLS gain (readout) ~ caught + level at t: the post-catch change beyond
            regression to the mean (a caught turn is selected for a weak edit, so its next step
            drifts back up anyway)
  persist   cos(r_t, r_{t+1}), r = D - h_c (the turn's own part of the edit), after caught /
            passed, vs 3+ turns apart: does the model keep its configuration after a pass and
            jump after a catch?
Across layers (residual stream = one basis): cos(e_L, e_L') and cos(h_L, h_L'): is the edit ONE
direction written once and carried, or re-written layer by layer?
Catcher depth: e for steps whose catching probes (margin > 0 at t) are all single-layer at <= 12
(shallow) or >= 22 (deep); depth profile |e_L| / |D_L| of each. Bootstrap 95% intervals over
episodes for the headline numbers.

OOD battery (--ood): unsupervised detectors fit on BASE states only (--ref: base dumps of the
OTHER split), scored on the given pairs: AUROC(trained hiding vs trained no-brief turns) and the
same for base (excess = what hiding adds); hiding vs MISMATCH turns (brief in context on both
sides, so the brief cannot be the signal); TPR at the 99th percentile of the model's own
non-hiding turns. Detectors per layer: maha (top-32 PCs), knn (k=10 cosine distance), pcares
(energy outside the top-64 PCs), xlayer (ridge from layer L to L+4 in 64-PC coordinates, residual
norm), norm.

  python scripts/tica_dodge_anatomy.py --base base_heldout_s1.pt,base_heldout_s2.pt \
      --model p1=p1_s1200_heldout_s1.pt,p1_s1200_heldout_s2.pt [--model ...] \
      [--ood --ref base_train_s1.pt,base_train_s2.pt] --out O.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tica_hide_geometry import KEY, auroc, cos, load

H_CONDS = ("real_score", "real_verdict")


def layer_of(kind: str | None) -> int | None:
    """Single-layer kind 'arch@L' -> L; multi-layer / unknown -> None."""

    m = re.fullmatch(r"[a-z_0-9]+@(\d+)", kind or "")
    return int(m.group(1)) if m else None


def load_records(paths: list[str]) -> list[dict]:
    recs = []
    for p in paths:
        shard = Path(p).stem.rsplit("_", 1)[-1]
        js = Path(p).with_suffix(".json")
        for r in json.loads(js.read_text())["records"]:
            recs.append({**r, "shard": shard})
    return recs


def catcher_layers(r: dict) -> list:
    """Layers of the probes that fired on this record (deployed refits; None = multi-layer)."""

    out = []
    for n, (_, fired) in r.get("refits", {}).items():
        meta = r.get("refit_meta", {}).get(n, {})
        if fired and meta.get("deployed", 1 << 30) <= r["turn"]:
            out.append(layer_of(meta.get("kind")))
    return out


def ci(vals: list[float], groups: list, n_boot: int = 400, seed: int = 0):
    """Mean and 95% bootstrap interval, resampling groups (episodes)."""

    if not vals:
        return None
    by = defaultdict(list)
    for v, g in zip(vals, groups, strict=True):
        by[g].append(v)
    keys = list(by)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [v for k in (rng.choice(keys) for _ in keys) for v in by[k]]
        boots.append(sum(s) / len(s))
    boots.sort()
    m = sum(vals) / len(vals)
    return [round(m, 3), round(boots[int(0.025 * n_boot)], 3), round(boots[int(0.975 * n_boot)], 3)]


def ols_effect(rows: list, n_boot: int = 400, seed: int = 0):
    """y = a + beta * caught + gamma * level_t: beta = the change after a catch beyond what the
    level at t predicts (regression to the mean controlled). rows: (y, caught, level, group).
    Returns [beta, lo, hi] (bootstrap over groups) and gamma."""

    if len(rows) < 8 or len({r[1] for r in rows}) < 2:
        return None

    def fit(rs):
        X = torch.tensor([[1.0, float(r[1]), r[2]] for r in rs], dtype=torch.float64)
        y = torch.tensor([r[0] for r in rs], dtype=torch.float64)[:, None]
        return torch.linalg.lstsq(X, y).solution[:, 0]

    full = fit(rows)
    by = defaultdict(list)
    for r in rows:
        by[r[3]].append(r)
    keys, rng, bs = list(by), random.Random(seed), []
    for _ in range(n_boot):
        rs = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        if len({r[1] for r in rs}) == 2:
            bs.append(float(fit(rs)[1]))
    bs.sort()
    lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]
    return {"beta": [round(float(full[1]), 3), round(lo, 3), round(hi, 3)],
            "gamma": round(float(full[2]), 3), "n": len(rows)}


def analyse(layers, xb, mb, xt, mt, recs, min_conc: int = 3) -> dict:
    if len(recs) != len(mt):
        raise SystemExit(f"records {len(recs)} != dump rows {len(mt)}")
    for r, m in zip(recs, mt, strict=True):
        if (r["ep"], r["turn"], r["concept"]) != (m["ep"], m["turn"], m["concept"]):
            raise SystemExit("panel JSON records and dump rows are not aligned")
    ib = {tuple(m[k] for k in KEY): i for i, m in enumerate(mb)}
    pairs = [(ib[tuple(m[k] for k in KEY)], j) for j, m in enumerate(mt)
             if tuple(m[k] for k in KEY) in ib]
    bi = torch.tensor([p[0] for p in pairs])
    ti = torch.tensor([p[1] for p in pairs])
    meta = [mt[j] for j in ti.tolist()]
    rr = [recs[j] for j in ti.tolist()]
    H = [i for i, m in enumerate(meta) if m["cond"] in H_CONDS and not m["off"] and m["turn"] >= 1]
    N = [i for i, m in enumerate(meta) if m["cond"] == "nobrief" and m["turn"] >= 1]
    pos = {(meta[i]["shard"], meta[i]["ep"], meta[i]["turn"]): i for i in H}
    steps = []  # (i_t, i_t1, caught_at_t, catcher layers at t)
    for i in H:
        m = meta[i]
        j = pos.get((m["shard"], m["ep"], m["turn"] + 1))
        if j is not None and meta[j]["concept"] == m["concept"]:
            steps.append((i, j, bool(m["detected"]), catcher_layers(rr[i])))
    nxt = {s[0]: s for s in steps}
    chains = [(s, nxt[s[1]]) for s in steps if s[1] in nxt]
    ep_of = lambda i: (meta[i]["shard"], meta[i]["ep"])  # noqa: E731
    cnt = defaultdict(int)
    for i in H:
        cnt[meta[i]["concept"]] += 1
    concepts = [c for c, n in cnt.items() if n >= min_conc]
    shallow = [s for s in steps if s[2] and s[3] and all(x is not None and x <= 12 for x in s[3])]
    deep = [s for s in steps if s[2] and s[3] and all(x is not None and x >= 22 for x in s[3])]
    out = {"n_paired": len(pairs), "n_H": len(H), "n_N": len(N),
           "n_steps": {"caught": sum(s[2] for s in steps), "passed": sum(not s[2] for s in steps),
                       "shallow_catcher": len(shallow), "deep_catcher": len(deep)},
           "n_chains": len(chains), "concepts": concepts, "layers": {}, "cross_layer": {}}
    E, Hd = {}, {}
    for li, L in enumerate(layers):
        B, T = xb[bi, li], xt[ti, li]
        D = T - B
        dN = D[N].mean(0) if N else torch.zeros(D.shape[1])
        h_c, u_c, sep_c = {}, {}, {}
        ref = H + N
        for c in concepts:
            hc = [i for i in H if meta[i]["concept"] == c]
            h_c[c] = D[hc].mean(0) - dN
            inc = [i for i in ref if meta[i]["concept"] == c]
            exc = [i for i in ref if meta[i]["concept"] != c]
            if inc and exc:
                u = B[inc].mean(0) - B[exc].mean(0)
                u_c[c] = u / (u.norm() + 1e-8)
                sep_c[c] = float(u.norm())
        EH = D[H] - dN
        P3 = torch.linalg.svd(EH - EH.mean(0), full_matrices=False)[2][:3]
        hide = EH.mean(0)
        dnorm = float(D[H].norm(dim=1).mean())
        bnorm = float(B[H].norm(dim=1).mean())
        row = defaultdict(lambda: defaultdict(list))
        reg = defaultdict(list)  # (y, caught, level at t, group) for the level-controlled effect
        for i, j, caught, _ in steps:
            c = meta[i]["concept"]
            if c not in h_c:
                continue
            d = D[j] - D[i]
            hn2 = float(h_c[c] @ h_c[c]) + 1e-8
            reg["gain"].append((float(d @ h_c[c]) / hn2, caught, float(D[i] @ h_c[c]) / hn2,
                                ep_of(i)))
            if c in u_c:
                reg["readout"].append((float(d @ u_c[c]) / (sep_c[c] + 1e-8), caught,
                                       float(D[i] @ u_c[c]) / (sep_c[c] + 1e-8), ep_of(i)))
            dn2 = float(d @ d) + 1e-8
            hh = h_c[c]
            lab = "caught" if caught else "passed"
            g = ep_of(i)
            row[lab]["rel"].append((float(d.norm()) / (float(D[i].norm()) + 1e-8), g))
            row[lab]["gain"].append((float(d @ hh) / (float(hh @ hh) + 1e-8), g))
            row[lab]["f_hide"].append((float(d @ hh) ** 2 / (float(hh @ hh) + 1e-8) / dn2, g))
            row[lab]["f_sub3"].append((float(((P3 @ d) ** 2).sum()) / dn2, g))
            if c in u_c:
                row[lab]["readout"].append((float(d @ u_c[c]) / (sep_c[c] + 1e-8), g))
        res = {lab: {k: ci([v for v, _ in vs], [g for _, g in vs]) for k, vs in ks.items()}
               for lab, ks in row.items()}

        def esc(sel, steps_=steps, D=D):
            dc = [D[j] - D[i] for i, j, cg, _ in steps_ if cg and sel((i, j, cg, _))]
            dp = [D[j] - D[i] for i, j, cg, _ in steps_ if not cg]
            if len(dc) < 3 or len(dp) < 3:
                return None
            return torch.stack(dc).mean(0) - torch.stack(dp).mean(0)

        e = esc(lambda s: True)
        # persistence of the concept-specific part of the edit, r = D - h_c: cos(r_t, r_t+1) by the
        # verdict at t, vs the same episode 3+ turns apart (texts differ at every lag)
        per = defaultdict(list)
        hk = [i for i in H if meta[i]["concept"] in h_c]
        for i, j, caught, _ in steps:
            c = meta[i]["concept"]
            if c in h_c:
                per["caught" if caught else "passed"].append(
                    (cos(D[i] - h_c[c], D[j] - h_c[c]), ep_of(i)))
        for i in hk:
            for j in hk:
                if (ep_of(i) == ep_of(j) and meta[j]["turn"] - meta[i]["turn"] >= 3
                        and meta[i]["concept"] == meta[j]["concept"]):
                    c = meta[i]["concept"]
                    per["lag3plus"].append((cos(D[i] - h_c[c], D[j] - h_c[c]), ep_of(i)))
        es, ed = esc(lambda s: s in shallow), esc(lambda s: s in deep)
        res.update({
            "esc_rel": None if e is None else round(float(e.norm()) / dnorm, 3),
            "esc_cos_hide": None if e is None else round(cos(e, hide), 3),
            "esc_cos_anti": (None if e is None or not u_c else
                             round(sum(cos(e, -u) for u in u_c.values()) / len(u_c), 3)),
            "esc_f_sub3": (None if e is None else
                           round(float(((P3 @ e) ** 2).sum()) / (float(e @ e) + 1e-8), 3)),
            "esc_abs": None if e is None else round(float(e.norm()) / bnorm, 4),
            "hide_abs": round(float(hide.norm()) / bnorm, 4),
            "persist_cos": {k: ci([v for v, _ in vs], [g for _, g in vs]) for k, vs in per.items()},
            "feedback_effect": {k: ols_effect(v) for k, v in reg.items()},
            "depth_shallow_rel": None if es is None else round(float(es.norm()) / dnorm, 3),
            "depth_deep_rel": None if ed is None else round(float(ed.norm()) / dnorm, 3),
            "shallow_vs_deep_cos": None if es is None or ed is None else round(cos(es, ed), 3),
            "hide_rel": round(float(hide.norm()) / dnorm, 3),
        })
        out["layers"][L] = res
        if e is not None:
            E[L] = e
        Hd[L] = hide
    ls = sorted(Hd)
    out["cross_layer"] = {
        "hide_cos": {f"{a}-{b}": round(cos(Hd[a], Hd[b]), 3) for a in ls for b in ls if a < b},
        "esc_cos": {f"{a}-{b}": round(cos(E[a], E[b]), 3) for a in E for b in E if a < b},
    }
    return out


def ood(layers, xr, xb, mb, xt, mt) -> dict:
    """Unsupervised detectors fit on base reference states xr [n, layers, d]."""

    def idx(meta, f):
        return [i for i, m in enumerate(meta) if f(m)]

    fh = lambda m: m["cond"] in H_CONDS and not m["off"] and m["turn"] >= 1  # noqa: E731
    fn = lambda m: m["cond"] == "nobrief" and m["turn"] >= 1  # noqa: E731
    fm = lambda m: m["cond"] == "mismatch" and not m["off"] and m["turn"] >= 1  # noqa: E731
    th, tn, bh, bn = idx(mt, fh), idx(mt, fn), idx(mb, fh), idx(mb, fn)
    tm, bm = idx(mt, fm), idx(mb, fm)
    out = {}
    for li, L in enumerate(layers):
        R = xr[:, li]
        mu = R.mean(0)
        _, S, V = torch.linalg.svd(R - mu, full_matrices=False)
        P32, ev = V[:32], S[:32] ** 2 / max(1, len(R) - 1)
        P64 = V[:64]
        Rn = torch.nn.functional.normalize(R, dim=1)
        det = {
            "maha": lambda q, mu=mu, P=P32, ev=ev: (((q - mu) @ P.T) ** 2 / (ev + 1e-6)).sum(1),
            "knn": lambda q, Rn=Rn: (1 - torch.nn.functional.normalize(q, dim=1) @ Rn.T)
            .topk(10, dim=1, largest=False).values.mean(1),
            "pcares": lambda q, mu=mu, P=P64: ((q - mu) - ((q - mu) @ P.T) @ P).norm(dim=1)
            / (q - mu).norm(dim=1).clamp(min=1e-6),
            "norm": lambda q: q.norm(dim=1),
        }
        if L + 4 in layers:
            lj = layers.index(L + 4)
            R2 = xr[:, lj]
            mu2 = R2.mean(0)
            V2 = torch.linalg.svd(R2 - mu2, full_matrices=False)[2][:64]
            Z1, Z2 = (R - mu) @ P64.T, (R2 - mu2) @ V2.T
            W = torch.linalg.solve(Z1.T @ Z1 + torch.eye(Z1.shape[1]), Z1.T @ Z2)

            def xl(q, qi=None, P64=P64, V2=V2, W=W, mu=mu, mu2=mu2):
                return (((qi - mu2) @ V2.T) - ((q - mu) @ P64.T) @ W).norm(dim=1)

            det["xlayer"] = xl
        res = {}
        for name, f in det.items():
            def score(x, ii, name=name, f=f, li=li, L=L):
                q = x[ii, li]
                return f(q, x[ii, layers.index(L + 4)]) if name == "xlayer" else f(q)

            if not (th and tn and bh and bn):
                continue
            a_t = auroc(score(xt, th), score(xt, tn))
            a_b = auroc(score(xb, bh), score(xb, bn))
            a_tb = auroc(score(xt, th), score(xb, bh))
            r_ = {"hide_vs_nobrief": round(a_t, 3), "base_same": round(a_b, 3),
                  "excess": round(a_t - a_b, 3), "trained_vs_base_hide": round(a_tb, 3)}
            if tm and bm:  # brief in context on both sides (mismatch: hide A, text about B)
                r_["hide_vs_mismatch"] = round(auroc(score(xt, th), score(xt, tm)), 3)
                r_["base_hide_vs_mismatch"] = round(auroc(score(xb, bh), score(xb, bm)), 3)
            # deployable: threshold = 99th percentile of the trained model's non-hiding turns
            ref_s = score(xt, tn + tm)
            thr = float(torch.quantile(ref_s.double(), 0.99))
            r_["tpr_at_1pct"] = round(float((score(xt, th) > thr).float().mean()), 3)
            r_["base_tpr_at_1pct"] = round(float((score(xb, bh) > float(torch.quantile(
                score(xb, bn + bm).double(), 0.99))).float().mean()), 3)
            res[name] = r_
        out[L] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="a.pt,b.pt (base dumps, same split)")
    ap.add_argument("--model", action="append", required=True, help="label=a.pt,b.pt")
    ap.add_argument("--ood", action="store_true")
    ap.add_argument("--ref", help="base dumps of the reference (train) split for --ood")
    ap.add_argument("--min-conc", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    layers, xb, mb = load(a.base.split(","))
    xr = load(a.ref.split(","))[1] if a.ood and a.ref else None
    res = {}
    for spec in a.model:
        lab, mp = spec.split("=", 1)
        lt, xt, mt = load(mp.split(","))
        if lt != layers:
            raise SystemExit("layers differ between base and model dumps")
        r = analyse(layers, xb, mb, xt, mt, load_records(mp.split(",")), a.min_conc)
        if xr is not None:
            r["ood"] = ood(layers, xr, xb, mb, xt, mt)
        res[lab] = r
        print(f"== {lab}: paired {r['n_paired']} H {r['n_H']} steps {r['n_steps']} "
              f"chains {r['n_chains']}", flush=True)
        for L, v in r["layers"].items():
            c, p = v.get("caught", {}), v.get("passed", {})
            print(f"  L{L:<3d} gain c/p {c.get('gain')} {p.get('gain')}  readout c/p "
                  f"{c.get('readout')} {p.get('readout')}  esc_rel {v['esc_rel']} "
                  f"cos_hide {v['esc_cos_hide']} fb {v['feedback_effect']} "
                  f"persist {v['persist_cos']}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
