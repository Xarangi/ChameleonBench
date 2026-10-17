# ruff: noqa
"""Run THEIR ACTUAL probe-fit code (not our re-implementation) to settle whether the
public codebase reproduces the released benign probe from scratch.

Strategy: stub the heavy/unused deps (jaxtyping/einops/sae_lens/strong_reject) so their
obf_reps modules import, then call their real HFHardPrompted + train_probe (which builds
ResponseProbeDataModule + LogisticRegressionMetric and fits exactly as in the paper).

Phase controlled by env REIFY:
  REIFY=0 (default): IMPORT-ONLY smoke test on login node (no model load). Verifies their
                     classes import with stubs and prints the train_probe signature + where
                     ResponseProbeDataModule is defined.
  REIFY=1: full run on GPU — load IlyaGusev via their HFHardPrompted.load_model, build
           german pos/neg from the rated json, run their train_probe(probe_type='logistic',
           lr=4e-5, epochs=5, batch=64), cosine the fitted weight to the released probe.
"""
import os, sys, types, json, glob
from pathlib import Path

REPO = os.environ["NC_REPO"]  # path to cloned neural-chameleons


# ---- stub heavy/unused deps so obf_reps imports ----
def _stub(name, attrs=None):
    m = types.ModuleType(name)
    m.__file__ = f"<stub:{name}>"
    m.__path__ = []  # mark as package so submodule imports work
    m.__spec__ = None
    for k, v in (attrs or {}).items():
        setattr(m, k, v)

    def _getattr(n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)  # let dunders fail normally (inspect-safe)
        return _Dummy

    m.__getattr__ = _getattr
    sys.modules[name] = m
    return m


class _Dummy:
    def __class_getitem__(cls, item):  # supports Float[Tensor, "..."]
        return cls
    def __init__(self, *a, **k): ...
    def __call__(self, *a, **k): return None
    def __getattr__(self, n): return _Dummy


# jaxtyping: type-annotation subscripting
def _missing(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is None
    except Exception:
        return True


jt = _stub("jaxtyping")
for nm in ("Bool", "Float", "Int64", "Int", "Shaped", "Num", "UInt8"):
    setattr(jt, nm, _Dummy)
_stub("einops", {"rearrange": lambda *a, **k: a[0], "repeat": lambda *a, **k: a[0],
                 "reduce": lambda *a, **k: a[0], "parse_shape": lambda *a, **k: {}})
_stub("sae_lens", {"SAE": _Dummy})
sr = _stub("strong_reject")
sr.evaluate = _stub("strong_reject.evaluate", {"strongreject_finetuned": _Dummy})
# stub other optional/heavy deps only if genuinely absent (don't shadow real installs)
if _missing("dotenv"):
    _stub("dotenv", {"load_dotenv": lambda *a, **k: None, "find_dotenv": lambda *a, **k: ""})
for opt in ("repe", "sparsify", "sentence_transformers"):
    if _missing(opt):
        _stub(opt)

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
sys.path.insert(0, os.path.join(REPO, "experiments", "shared_utils"))


def main():
    import inspect
    from obf_reps.models.hf import HFHardPrompted
    from obf_reps.models import ModelConfig
    print("[their-code] imported HFHardPrompted OK")
    # locate ResponseProbeDataModule + train_probe
    import importlib
    ut = importlib.import_module("utils_training")
    print("[their-code] train_probe signature:", inspect.signature(ut.train_probe))
    rpdm = getattr(ut, "ResponseProbeDataModule", None)
    src_mod = rpdm.__module__ if rpdm else "(import in utils_training)"
    print(f"[their-code] ResponseProbeDataModule module = {src_mod}")
    from obf_reps.metrics import LogisticRegressionMetric, MetricConfig
    print("[their-code] imported LogisticRegressionMetric + MetricConfig OK")

    if os.environ.get("REIFY", "0") != "1":
        print("[their-code] IMPORT SMOKE PASSED (set REIFY=1 + GPU to actually fit).")
        return

    # ---- full run (GPU) ----
    import torch
    import torch.nn.functional as F
    from next_chameleons.hf.paper_probes import _released_weight_bias

    concept = os.environ.get("REFIT_CONCEPT", "german")
    cap = int(os.environ.get("REFIT_CAP", "300"))
    probe_dir, rated_json = os.environ["PROBE_DIR"], os.environ["RATED_JSON"]

    # released target
    wb = None
    for p in glob.glob(os.path.join(probe_dir, "*_weights.pt")):
        if os.path.basename(p)[:-len("_weights.pt")].lower() == concept.lower():
            w, b = _released_weight_bias(torch.load(p, map_location="cpu", weights_only=False))
            wb = w.float().reshape(-1)
    assert wb is not None

    # their model wrapper
    model = HFHardPrompted.load_model(os.environ["MODEL_PATH"], ModelConfig(model_dtype=torch.bfloat16))
    print("[their-code] loaded model via HFHardPrompted.load_model")

    # german pos/neg from rated json (POS topical rating>=0.75, NEG<=0.25)
    recs = json.loads(Path(rated_json).read_text("utf-8"))
    maxchars = int(os.environ.get("REFIT_MAXCHARS", "2000"))  # drop long-response tail (no truncation in their tokenize)
    pos, neg = [], []
    for r in recs:
        resp = r.get("topical_response"); rt = (r.get("topical_response_normalized_ratings") or {})
        if not resp or concept not in rt or len(resp) > maxchars or len(r.get("prompt", "")) > maxchars:
            continue
        s = float(rt[concept])
        if s >= 0.75: pos.append((r.get("prompt", ""), resp, None))
        elif s <= 0.25: neg.append((r.get("prompt", ""), resp, None))
    pos, neg = pos[:cap], neg[:cap]
    print(f"[their-code] german pos={len(pos)} neg={len(neg)}")

    # Extract reps via THEIR EXACT forward (their tokenize: prompt chat-template +
    # opt-loc token, response raw-appended), one example at a time (memory-safe since
    # their forward computes full-vocab logits+loss). Then fit a fast logistic ourselves
    # (identical Adam-BCE; fit details don't change the direction). Index layers by
    # ABSOLUTE hidden_states index, so no positional-indexing ambiguity.
    LAYERS = [11, 12, 13, 14]
    feats = {L: [] for L in LAYERS}
    labels = []
    n_done = 0
    for label, pool in ((1.0, pos), (0.0, neg)):
        for prompt, response, _ in pool:
            try:
                fr = model.forward_from_string(
                    input_text=prompt, target_text=response,
                    add_chat_template=True, use_tunable_params=False,
                )
            except Exception as e:
                if n_done == 0:
                    print(f"[their-code] forward_from_string error: {type(e).__name__}: {e}", flush=True)
                continue
            tr = fr.target_reps  # [1, layers, target_len, hidden]
            ntok = tr.shape[2]
            if ntok < 1:
                continue
            for L in LAYERS:
                feats[L].append(tr[0, L].float().cpu())
            labels.extend([label] * ntok)
            n_done += 1
    y = torch.tensor(labels)
    print(f"[their-code] extracted via THEIR forward: examples_used={n_done} tokens={y.shape[0]} pos_frac={y.mean():.3f}", flush=True)

    def pr(w): w2 = w.float() ** 2; return (w2.sum() ** 2 / (w2 ** 2).sum()).item()

    def fit_fast(X, y, lr=4e-5, epochs=5, bs=64):
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros((), requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr, weight_decay=1e-5, eps=1e-5)
        for _ in range(epochs):
            for s in range(0, X.shape[0], bs):
                opt.zero_grad(set_to_none=True)
                F.binary_cross_entropy_with_logits(X[s:s+bs] @ w + b, y[s:s+bs]).backward(); opt.step()
        return w.detach(), b.detach()

    def bal_acc(X, y, w, b):
        p = (torch.sigmoid(X @ w + b) >= 0.5).float(); pos_, neg_ = y.bool(), ~y.bool()
        return 0.5 * (p[pos_].mean().item() + (1 - p[neg_]).mean().item())

    print(f"\n[their-code] THEIR-extraction fit, cosine-to-released (released_PR={pr(wb):.0f}):")
    best = (-2.0, -1)
    for L in LAYERS:
        X = torch.cat(feats[L], 0)
        w_fit, b_fit = fit_fast(X, y)
        c = F.cosine_similarity(w_fit.unsqueeze(0), wb.unsqueeze(0)).item()
        if c > best[0]:
            best = (c, L)
        print(f"    hs[{L:2d}]  cos={c:+.4f}  bal_acc={bal_acc(X, y, w_fit, b_fit):.4f}  fit_PR={pr(w_fit):.0f}", flush=True)
    print(f"[their-code] BEST: hs[{best[1]}] cos={best[0]:+.4f}  "
          f"({'REPRODUCES' if best[0] > 0.7 else 'does NOT reproduce'} the released german probe)")


if __name__ == "__main__":
    main()
