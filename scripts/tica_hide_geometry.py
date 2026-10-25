"""HOW does the trained model hide? Offline geometry of panel feature dumps (10-25, user:
"isolate the mechanism"; CPU only, aggregates only). Inputs: tica_probe_panel.py --dump-feats
files of the BASE model and of trained checkpoints, run with the same seeds / topics, so the
same episodes (concept, condition, forced texts) are read by both models. Records are PAIRED on
(shard, ep, turn, cond, concept, text_concept); unpaired records are dropped and counted.

Sets (turn >= 1, monitor on): H = real_score (asked to hide), N = nobrief (not asked).
D = x_trained - x_base on the same tokens = what the adapter does to that state.

Per layer:
  shift       ||mean D_H - mean D_N|| / mean ||x_base||: size of the hiding-specific edit
  shared      mean pairwise cosine of per-concept hide shifts s_c = mean D_H(c) - mean D_N
              (one shared "hide" vector?)
  pc1 / pc3   variance share of the top 1 / 3 PCs of the hiding edits D_H - mean D_N
  anti        mean cos(s_c, -u_c): u_c = the base model's concept direction (base states of all
              paired turns: mean_c - mean_rest)
  supp        fraction of the base concept signal left along u_c on H turns (1 = untouched,
              0 = removed, < 0 = flipped)
  base_auc    AUROC of a base-model concept probe (logistic on base states, c vs rest) on trained H
  fresh_auc   AUROC of a probe fit on the TRAINED model's own H states (2-fold, c vs rest);
              low base_auc + high fresh_auc = obfuscation / rotation, not erasure
  esc_cos     cos(mean step after a CAUGHT turn - after a PASSED turn, mean hide edit): is
              escalation a push along the same hide direction (one knob)?
  ablate_auc  base probe AUROC on trained H after projecting out the top-3 hide PCs (does that
              subspace carry the evasion at this layer?)
  anomaly     AUROC of a Mahalanobis score (top-32 PCs of base H + N) for trained H vs trained N,
              minus the same on base H vs base N (excess = how anomalous the hiding state is)

  python scripts/tica_hide_geometry.py --base base=A_s1.pt,A_s2.pt --model p1=B_s1.pt,B_s2.pt \
      [--model ...] --out O.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import torch

KEY = (
    "shard",
    "ep",
    "turn",
    "cond",
    "concept",
    "text_concept",
)  # shard = the seed suffix (_s1, _s2)


def load(paths: list[str]):
    xs, meta, layers = [], [], None
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        layers = layers or d["layers"]
        if d["layers"] != layers:
            raise SystemExit(f"layer mismatch in {p}")
        shard = Path(p).stem.rsplit("_", 1)[-1]
        for m in d["meta"]:
            meta.append({**m, "shard": shard})
        xs.append(d["x"].float())
    return layers, torch.cat(xs), meta


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = torch.cat([pos, neg])
    r = s.argsort().argsort().float() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logistic(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-2, steps: int = 300) -> tuple:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    w = torch.zeros(z.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    pw = (len(y) - y.sum()) / y.sum().clamp(min=1)

    def closure():
        opt.zero_grad()
        lo = z @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, y.float(), pos_weight=pw)
        loss = loss + l2 * (w**2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return lambda q: ((q - mu) / sd) @ w.detach() + b.detach()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def analyse(layers, xb, mb, xt, mt, min_conc: int = 3) -> dict:
    ib = {tuple(m[k] for k in KEY): i for i, m in enumerate(mb)}
    pairs = [
        (ib[tuple(m[k] for k in KEY)], j)
        for j, m in enumerate(mt)
        if tuple(m[k] for k in KEY) in ib
    ]
    if not pairs:
        raise SystemExit("no paired records (were the dumps run with the same seeds / topics?)")
    bi = torch.tensor([p[0] for p in pairs])
    ti = torch.tensor([p[1] for p in pairs])
    meta = [mt[j] for j in ti.tolist()]
    sel = lambda f: torch.tensor([i for i, m in enumerate(meta) if f(m)], dtype=torch.long)  # noqa: E731
    H = sel(
        lambda m: m["cond"] in ("real_score", "real_verdict") and not m["off"] and m["turn"] >= 1
    )
    N = sel(lambda m: m["cond"] == "nobrief" and m["turn"] >= 1)
    concepts = [
        c
        for c, n in sorted(
            (
                (c, sum(1 for i in H.tolist() if meta[i]["concept"] == c))
                for c in {meta[i]["concept"] for i in H.tolist()}
            ),
            key=lambda t: -t[1],
        )
        if n >= min_conc
    ]
    out = {
        "n_paired": len(pairs),
        "n_unpaired": len(mt) - len(pairs),
        "n_H": len(H),
        "n_N": len(N),
        "concepts": concepts,
        "layers": {},
    }
    for li, L in enumerate(layers):
        B, T = xb[bi, li], xt[ti, li]
        D = T - B
        scale = float(B.norm(dim=1).mean())
        dN = D[N].mean(0) if len(N) else torch.zeros(D.shape[1])
        EH = D[H] - dN  # hiding-specific edits per record
        shift = float(EH.mean(0).norm()) / scale
        # per-concept hide shifts; base concept directions from ALL paired base states of the
        # concept (episodes carry one condition each, so a concept may appear in H or N only)
        ref = torch.cat([H, N])
        rc = [meta[i]["concept"] for i in ref.tolist()]
        s_c, u_c = {}, {}
        for c in concepts:
            hc = torch.tensor([i for i in H.tolist() if meta[i]["concept"] == c])
            s_c[c] = D[hc].mean(0) - dN
            inc = torch.tensor([k for k, cc in enumerate(rc) if cc == c])
            exc = torch.tensor([k for k, cc in enumerate(rc) if cc != c])
            if len(inc) and len(exc):
                u = B[ref][inc].mean(0) - B[ref][exc].mean(0)
                u_c[c] = u / (u.norm() + 1e-8)
        cs = [c for c in concepts if c in u_c]
        pairs_cos = [cos(s_c[a], s_c[b]) for i, a in enumerate(cs) for b in cs[i + 1 :]]
        # PCs of the per-record hiding edits
        Z = EH - EH.mean(0)
        sv = torch.linalg.svdvals(Z) ** 2 if len(Z) > 3 else torch.ones(1)
        share = (sv / sv.sum()).tolist()
        U = torch.linalg.svd(EH, full_matrices=False)[2][:3] if len(EH) > 3 else None
        anti = [cos(s_c[c], -u_c[c]) for c in cs]
        supp, base_auc, fresh_auc, abl_auc = [], [], [], []
        Bh, Th = B[H], T[H]
        hcon = [meta[i]["concept"] for i in H.tolist()]
        for c in cs:
            y = torch.tensor([1 if cc == c else 0 for cc in hcon])
            if y.sum() < min_conc or (1 - y).sum() < min_conc:
                continue
            u = u_c[c]
            pb, pt = Bh @ u, Th @ u
            den = float(pb[y == 1].mean() - pb[y == 0].mean())
            if abs(den) > 1e-6:
                supp.append(float(pt[y == 1].mean() - pt[y == 0].mean()) / den)
            f = logistic(Bh, y)  # base probe: fit on base states of the same turns
            base_auc.append(auroc(f(Th)[y == 1], f(Th)[y == 0]))
            idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
            half = len(y) // 2
            a_, b_ = idx[:half], idx[half:]
            if y[a_].sum() >= 2 and y[b_].sum() >= 2 and (1 - y[a_]).sum() >= 2:
                g = logistic(Th[a_], y[a_])
                fresh_auc.append(auroc(g(Th[b_])[y[b_] == 1], g(Th[b_])[y[b_] == 0]))
            if U is not None:
                Ta = Th - (Th @ U.T) @ U
                abl_auc.append(auroc(f(Ta)[y == 1], f(Ta)[y == 0]))
        # escalation: step after a caught vs a passed H turn (same episode, consecutive turns)
        by_ep = defaultdict(list)
        for k in H.tolist():
            by_ep[meta[k]["ep"]].append(k)
        st_c, st_p = [], []
        for ks in by_ep.values():
            ks.sort(key=lambda k: meta[k]["turn"])
            for a_, b_ in itertools.pairwise(ks):
                if meta[b_]["turn"] == meta[a_]["turn"] + 1:
                    (st_c if meta[a_]["detected"] else st_p).append(T[b_] - T[a_])
        esc = (
            cos(torch.stack(st_c).mean(0) - torch.stack(st_p).mean(0), EH.mean(0))
            if len(st_c) >= 3 and len(st_p) >= 3
            else float("nan")
        )
        # anomaly (Mahalanobis in base PCs)
        ref_all = torch.cat([B[H], B[N]]) if len(N) else B[H]
        mu = ref_all.mean(0)
        Ur = torch.linalg.svd(ref_all - mu, full_matrices=False)
        P, ev = Ur[2][:32], (Ur[1][:32] ** 2) / max(1, len(ref_all) - 1)
        def maha(q, mu=mu, P=P, ev=ev):
            return (((q - mu) @ P.T) ** 2 / (ev + 1e-6)).sum(1)

        anom = (
            (auroc(maha(T[H]), maha(T[N])) - auroc(maha(B[H]), maha(B[N])))
            if len(N)
            else float("nan")
        )
        mean = lambda v: round(sum(v) / len(v), 3) if v else None  # noqa: E731
        out["layers"][L] = {
            "shift": round(shift, 4),
            "shared": mean(pairs_cos),
            "pc1": round(share[0], 3),
            "pc3": round(sum(share[:3]), 3),
            "anti": mean(anti),
            "supp": mean(supp),
            "base_auc": mean(base_auc),
            "fresh_auc": mean(fresh_auc),
            "ablate_auc": mean(abl_auc),
            "esc_cos": None if esc != esc else round(esc, 3),
            "anomaly_excess": None if anom != anom else round(anom, 3),
            "n_concepts": len(cs),
            "n_steps_caught_passed": [len(st_c), len(st_p)],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="label=a.pt,b.pt")
    ap.add_argument("--model", action="append", required=True, help="label=a.pt,b.pt (repeatable)")
    ap.add_argument("--min-conc", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, bp = a.base.split("=", 1)
    layers, xb, mb = load(bp.split(","))
    res = {}
    for spec in a.model:
        lab, mp = spec.split("=", 1)
        lt, xt, mt = load(mp.split(","))
        if lt != layers:
            raise SystemExit("layers differ between base and model dumps")
        r = analyse(layers, xb, mb, xt, mt, a.min_conc)
        res[lab] = r
        print(
            f"== {lab}: paired {r['n_paired']} (unpaired {r['n_unpaired']}), "
            f"H {r['n_H']}, N {r['n_N']}, concepts {len(r['concepts'])}"
        )
        cols = [
            "shift",
            "shared",
            "pc1",
            "pc3",
            "anti",
            "supp",
            "base_auc",
            "fresh_auc",
            "ablate_auc",
            "esc_cos",
            "anomaly_excess",
        ]
        print("   L  " + " ".join(f"{c:>9s}" for c in cols))
        for L, v in r["layers"].items():
            print(
                f"  {L:3d} "
                + " ".join(f"{(v[c] if v[c] is not None else float('nan')):9.3f}" for c in cols)
            )
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
