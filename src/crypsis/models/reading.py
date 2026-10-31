"""Activation reads on a Policy (from tica/policy.py): decoder-block hooks, mean / token pooling
over content spans, the per-view LM terms, and the exact speed paths.

Layer L is the output of decoder block L (= HF hidden_states[L + 1], the NC / Kowalski
convention), read by a forward hook. With grad=True the activations stay in the graph.

Speed paths (all exact; none changes which tokens are read or how they are pooled):
- shared prefix: forks of one conversation run their common token prefix once with a KV /
  recurrent cache;
- incremental: a growing conversation keeps its cache and runs only the new tokens;
- truncate: no-grad reads stop the forward after the deepest requested layer (layer L's output
  does not depend on later layers; only models whose decoder loop is bounded by
  config.num_hidden_layers, e.g. Qwen3.5 / Gemma-4; otherwise a no-op);
- lm_chunk: the LM terms are computed from the head's input in chunks of lm_chunk positions
  (recomputed in backward), so the [n_tok, V] log-probs never exist at once;
- sort_reads: shared-prefix chunks of similar suffix length (less padding); rows return in input
  order.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from crypsis.models.chat import MultiView, View

if TYPE_CHECKING:
    from crypsis.models.policy import Policy


def keep_tokens(tok: torch.Tensor, keep: int | None) -> torch.Tensor:
    """CPU bf16 copy of a span's token states; with `keep`, an evenly spaced subset of at most
    `keep` tokens (an unbiased-in-position estimate of token means)."""

    if keep is not None and len(tok) > keep:
        tok = tok[torch.linspace(0, len(tok) - 1, keep, device=tok.device).round().long()]
    return tok.to("cpu", torch.bfloat16)


def topk_targets(lp: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference targets from log-probs lp [n_tok, V]: (top-k ids [n, k], their log-probs [n, k],
    log of the remaining (tail) mass [n]). Detached."""

    v, i = lp.detach().topk(k, -1)
    tail = torch.log1p(-v.exp().sum(-1).clamp(max=1 - 1e-6))
    return i, v, tail


def topk_kl(lp: torch.Tensor, ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Per-token KL(ref || policy) between the coarse-grained distributions {ref top-k tokens, one
    tail bucket}; lp = policy log-probs [n_tok, V] (with grad). <= the full KL (data-processing
    inequality), = it when the tail is empty."""

    i, v, tail = ref
    lpk = lp.gather(-1, i)
    p_tail = torch.log1p(-lpk.exp().sum(-1).clamp(max=1 - 1e-6))
    return (v.exp() * (v - lpk)).sum(-1) + tail.exp() * (tail - p_tail)


def span_extra(extra: dict | None, lp: torch.Tensor, counts: list[int]) -> None:
    """Fill `extra` from per-token log-probs lp (spans contiguous, sizes `counts`):
    extra["topk"] = k -> extra["ref"] = per-span topk_targets; extra["kl_ref"] = per-span targets
    (or None) -> extra["kl"] = per-span mean KL [n_spans] (0 where None; with lp's grad)."""

    if extra is None:
        return
    parts = torch.split(lp, counts)
    if extra.get("topk"):
        extra["ref"] = [topk_targets(x, extra["topk"]) for x in parts]
    if extra.get("kl_ref") is not None:
        refs = extra["kl_ref"]
        if len(refs) != len(parts):
            raise ValueError(f"kl_ref has {len(refs)} spans, the batch {len(parts)}")
        kl = []
        for x, r in zip(parts, refs, strict=True):
            if r is None:
                kl.append(x.sum() * 0.0)
                continue
            if r[0].shape[0] != x.shape[0]:
                raise ValueError(f"kl_ref span has {r[0].shape[0]} tokens, the policy {x.shape[0]}")
            kl.append(topk_kl(x, r).mean())
        extra["kl"] = torch.stack(kl)


def lm_terms(head, h: torch.Tensor, tok: torch.Tensor, counts: list[int], extra: dict | None,
             chunk: int, softcap: float | None = None) -> torch.Tensor:
    """Chunked LM terms: what the full-logits path computes from the [n_tok, V] log-probs, without
    ever holding them. h = post-norm hidden states at the predicting positions [n_tok, d] (with
    grad), tok = their target ids, counts = span sizes. Returns the per-token NLL [n_tok] and fills
    `extra` exactly as span_extra does; each chunk's logits are recomputed in backward.
    softcap: the model's final-logit soft-capping (Gemma-4), applied as its forward does."""

    from torch.utils.checkpoint import checkpoint

    want_ref = bool(extra and extra.get("topk"))
    refs = extra.get("kl_ref") if extra is not None else None
    if refs is not None and len(refs) != len(counts):
        raise ValueError(f"kl_ref has {len(refs)} spans, the batch {len(counts)}")
    n = h.shape[0]
    ki = kv = kt = has = None
    if refs is not None:  # per-token reference targets (dummy, masked rows for spans without one)
        k = next((r[0].shape[1] for r in refs if r is not None), 1)
        dummy = math.log(1.0 / (k + 1))
        rows_i, rows_v, rows_t, rows_h = [], [], [], []
        for c, r in zip(counts, refs, strict=True):
            if r is None:
                rows_i.append(torch.zeros(c, k, dtype=torch.long, device=h.device))
                rows_v.append(torch.full((c, k), dummy, device=h.device))
                rows_t.append(torch.full((c,), dummy, device=h.device))
                rows_h.append(torch.zeros(c, device=h.device))
                continue
            if r[0].shape[0] != c:
                raise ValueError(f"kl_ref span has {r[0].shape[0]} tokens, the policy {c}")
            rows_i.append(r[0].to(h.device))
            rows_v.append(r[1].to(h.device).float())
            rows_t.append(r[2].to(h.device).float())
            rows_h.append(torch.ones(c, device=h.device))
        ki, kv, kt, has = (torch.cat(x) for x in (rows_i, rows_v, rows_t, rows_h))

    def part(hc, tc, ic, vc, tl):
        logits = head(hc).float()
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        lp = torch.log_softmax(logits, -1)
        nll = -lp.gather(-1, tc[:, None])[:, 0]
        kl = topk_kl(lp, (ic, vc, tl)) if ic is not None else nll * 0.0
        top = topk_targets(lp, extra["topk"]) if want_ref else None
        return nll, kl, top

    nlls, kls, tops = [], [], []
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        args = (h[a:b], tok[a:b], *((ki[a:b], kv[a:b], kt[a:b]) if ki is not None
                                    else (None, None, None)))
        if torch.is_grad_enabled() and h.requires_grad:
            nll, kl, top = checkpoint(part, *args, use_reentrant=False)
        else:
            nll, kl, top = part(*args)
        nlls.append(nll)
        kls.append(kl)
        tops.append(top)
    nll = torch.cat(nlls) if nlls else h.new_zeros(0, dtype=torch.float32)
    if extra is not None:
        if want_ref:
            cat = [torch.cat(x) for x in zip(*tops, strict=True)]
            extra["ref"] = [tuple(x) for x in zip(*(torch.split(c, counts) for c in cat),
                                                  strict=True)]
        if refs is not None:
            per = torch.cat(kls) * has
            extra["kl"] = torch.stack([x.mean() if r is not None else x.sum() * 0.0
                                       for x, r in zip(torch.split(per, counts), refs,
                                                       strict=True)])
    return nll


class ActivationReader:
    """Reads decoder-block outputs of a Policy's model over conversation views.

    batch_size: rows per forward for pooled() / shared-prefix reads (None = policy.batch_size at
    call time). lm_chunk, sort_reads, truncate: the speed paths in the module docstring (0 / False
    = off). The adapter mode (policy.reference() / merged()) applies as set by the caller.
    """

    def __init__(self, policy: Policy, *, batch_size: int | None = None, lm_chunk: int = 0,
                 sort_reads: bool = False, truncate: bool = False) -> None:
        self.policy = policy
        self.batch_size = batch_size
        self.lm_chunk = lm_chunk
        self.sort_reads = sort_reads
        self.truncate = truncate
        layers = policy.layers
        self._depth_owner = next((m for m in policy.model.modules()
                                  if getattr(m, "layers", None) is layers
                                  and hasattr(getattr(m, "config", None), "num_hidden_layers")),
                                 None)

    @property
    def model(self):
        return self.policy.model

    def _bs(self, batch: int | None) -> int:
        return batch or self.batch_size or self.policy.batch_size

    def view(self, conv: list[dict]) -> View:
        return self.policy.chat.view(conv)

    def multi_view(self, conv: list[dict]) -> MultiView:
        return self.policy.chat.multi_view(conv)

    # ------------------------------------------------------------------------------ internals
    @contextmanager
    def _hooks(self, layers: list[int], store: dict, keep_grad: bool):
        handles = []
        blocks = self.policy.layers
        for L in layers:

            def hook(module, inp, out, L=L):
                h = out[0] if isinstance(out, tuple) else out
                store[L] = h if keep_grad else h.detach()

            handles.append(blocks[L].register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def depth(self, layers: list[int]):
        """Run only decoder layers 0..max(layers) inside (when truncate is on)."""

        own = self._depth_owner
        if not self.truncate or own is None or not layers:
            yield
            return
        n = own.config.num_hidden_layers
        own.config.num_hidden_layers = min(n, max(layers) + 1)
        try:
            yield
        finally:
            own.config.num_hidden_layers = n

    def _batch(self, views: list[View]):
        n = max(len(v.ids) for v in views)
        pad = self.policy.tokenizer.pad_token_id
        ids = torch.full((len(views), n), pad, dtype=torch.long)
        att = torch.zeros((len(views), n), dtype=torch.long)
        mask = torch.zeros((len(views), n), dtype=torch.bool)  # probe read: content only
        lm = torch.zeros((len(views), n), dtype=torch.bool)  # LM target: content + tail
        for r, v in enumerate(views):
            ids[r, : len(v.ids)] = torch.tensor(v.ids)
            att[r, : len(v.ids)] = 1
            mask[r, v.start : v.end] = True
            lm[r, v.start : len(v.ids)] = True
        d = self.policy.device
        return ids.to(d), att.to(d), mask.to(d), lm.to(d)

    @contextmanager
    def _head_capture(self, box: dict):
        """lm_chunk: the LM head's input (post-norm hidden states of the kept positions) goes to
        box["h"] and the head itself only sees the last position, so the model never builds the
        [B, keep, V] logits; lm_terms computes the LM terms from box["h"] in chunks."""

        head = self.model.get_output_embeddings()

        def pre(_mod, args):
            if "h" in box:  # only the model's own head call; lm_terms' chunk calls pass through
                return None
            box["h"] = args[0]
            return (args[0][:, -1:],)

        hook = head.register_forward_pre_hook(pre)
        try:
            yield head
        finally:
            hook.remove()

    # ------------------------------------------------------------------------------ forwards
    def forward(
        self, views: list[View], layers: list[int], *, grad: bool, with_nll: bool = False,
        extra: dict | None = None, token_out: dict | None = None, token_keep: int | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
        """One batch (no internal chunking). Returns ({layer: pooled [n, d] float32}, nll [n] or
        None). nll = mean negative log-likelihood of the content + end-of-turn tokens. extra
        (with_nll only): KL-anchor targets / KL per view, see span_extra. token_out: per-view
        content-token states (CPU bf16, keep_tokens)."""

        ids, att, mask, lm = self._batch(views)
        n = ids.shape[1]
        # Logits only from the first position that scores a content token (min start - 1).
        keep = n - max(0, min(v.start for v in views) - 1) if with_nll else 1
        store: dict[int, torch.Tensor] = {}
        ctx = nullcontext() if grad else torch.no_grad()
        chunked = bool(with_nll and self.lm_chunk)
        box: dict = {}
        with ctx, self._hooks(layers, store, keep_grad=grad), (
                self._head_capture(box) if chunked else nullcontext()) as head:
            out = self.model(
                input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=keep
            )
            m = mask.float()[..., None]
            cnt = m.sum(1).clamp(min=1)
            pooled = {L: (store[L].float() * m).sum(1) / cnt for L in layers}
            if token_out is not None:
                for L in layers:
                    token_out.setdefault(L, []).extend(
                        keep_tokens(store[L][r][mask[r]], token_keep) for r in range(len(views)))
            nll = None
            if with_nll:
                # Position j predicts token j+1, so content token k is scored by logits at k-1.
                # out.logits covers positions [n - keep, n).
                tgt_mask = lm[:, 1:][:, n - keep :]
                tok = ids[:, 1:][:, n - keep :][tgt_mask]
                if chunked:
                    per_tok = lm_terms(head, box["h"][:, :-1][tgt_mask], tok,
                                       tgt_mask.sum(1).tolist(), extra, self.lm_chunk,
                                       self.policy.logit_softcap)
                else:
                    logits = out.logits[:, :-1]
                    lp = torch.log_softmax(logits[tgt_mask].float(), -1)
                    per_tok = -lp.gather(-1, tok[:, None])[:, 0]
                row = torch.arange(len(views), device=ids.device)[:, None].expand_as(tgt_mask)[
                    tgt_mask
                ]
                nll = torch.zeros(len(views), device=ids.device).index_add(0, row, per_tok)
                nll = nll / tgt_mask.sum(1).clamp(min=1)
                if not chunked:
                    span_extra(extra, lp, tgt_mask.sum(1).tolist())
        return pooled, nll

    def forward_multi(
        self, mvs: list[MultiView], layers: list[int], *, grad: bool, with_nll: bool = False,
        token_out: dict | None = None, extra: dict | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
        """ONE forward per conversation; pooled activations for EVERY span of every MultiView
        ([sum(len(spans)), d] per layer, conversation-major, turn order) and the per-span mean NLL
        of content + end-of-turn tokens: identical (by causality) to forward() on each turn's
        per-turn view. token_out: {L: [per-span [n_tokens, d] float states]} in pooled-row order
        (WITH grad when grad=True). extra (with_nll only): as forward()."""

        n = max(len(m.ids) for m in mvs)
        pad = self.policy.tokenizer.pad_token_id
        ids = torch.full((len(mvs), n), pad, dtype=torch.long)
        att = torch.zeros((len(mvs), n), dtype=torch.long)
        for r, m in enumerate(mvs):
            ids[r, : len(m.ids)] = torch.tensor(m.ids)
            att[r, : len(m.ids)] = 1
        ids, att = ids.to(self.policy.device), att.to(self.policy.device)
        first = min(m.spans[0][0] for m in mvs)
        keep = n - max(0, first - 1) if with_nll else 1
        store: dict[int, torch.Tensor] = {}
        ctx = nullcontext() if grad else torch.no_grad()
        chunked = bool(with_nll and self.lm_chunk)
        box: dict = {}
        with ctx, self._hooks(layers, store, keep_grad=grad), (
                self._head_capture(box) if chunked else nullcontext()) as head:
            out = self.model(
                input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=keep
            )
            pooled = {
                L: torch.stack(
                    [store[L][r, a:b].float().mean(0) for r, m in enumerate(mvs)
                     for a, b, _ in m.spans]
                )
                for L in layers
            }
            if token_out is not None:
                for L in layers:
                    token_out[L] = [store[L][r, a:b].float() for r, m in enumerate(mvs)
                                    for a, b, _ in m.spans]
            nll = None
            if with_nll:
                rows, poss, span_id = [], [], []
                j = 0
                for r, m in enumerate(mvs):
                    for a, _, te in m.spans:
                        for q in range(a, te):  # token q is predicted by the logits at q - 1
                            rows.append(r)
                            poss.append(q)
                            span_id.append(j)
                        j += 1
                rows_t = torch.tensor(rows, device=ids.device)
                pos_t = torch.tensor(poss, device=ids.device)
                if chunked:
                    counts = [te - a for m in mvs for a, _, te in m.spans]
                    per_tok = lm_terms(head, box["h"][rows_t, pos_t - 1 - (n - keep)],
                                       ids[rows_t, pos_t], counts, extra, self.lm_chunk,
                                       self.policy.logit_softcap)
                else:
                    lp = torch.log_softmax(out.logits[rows_t, pos_t - 1 - (n - keep)].float(),
                                           -1)
                    per_tok = -lp.gather(-1, ids[rows_t, pos_t][:, None])[:, 0]
                sid = torch.tensor(span_id, device=ids.device)
                tot = torch.zeros(j, device=ids.device).index_add(0, sid, per_tok)
                cnt = torch.zeros(j, device=ids.device).index_add(0, sid, torch.ones_like(per_tok))
                nll = tot / cnt.clamp(min=1)
                if not chunked:
                    span_extra(extra, lp, cnt.long().tolist())  # spans are contiguous in lp
        return pooled, nll

    # ------------------------------------------------------------------------------ pooled reads
    def pooled(self, convs: list[list[dict]], layers: list[int], token_out: dict | None = None,
               token_keep: int | None = None, batch: int | None = None,
               ) -> dict[int, torch.Tensor]:
        """No-grad pooled reads for many conversations, in batches; token_out as in
        pooled_shared_prefix (right padding: content states are exact)."""

        views = [self.view(c) for c in convs]
        bs = self._bs(batch)
        parts: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        for i in range(0, len(views), bs):
            with self.depth(layers):
                p, _ = self.forward(views[i : i + bs], layers, grad=False, token_out=token_out,
                                    token_keep=token_keep)
            for L in layers:
                parts[L].append(p[L])
        return {L: torch.cat(v) for L, v in parts.items()}

    @torch.no_grad()
    def pooled_shared_prefix(
        self, convs: list[list[dict]], layers: list[int], min_prefix: int = 64,
        token_out: dict | None = None, token_keep: int | None = None, batch: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """pooled() for forks of one conversation (batch: forks per call): the longest common
        token prefix runs once with a KV / recurrent cache and only the per-fork suffixes are
        batched against copies of it. Same reads as pooled() up to kernel numerics; falls back to
        it for prefixes shorter than min_prefix. token_out (optional dict): filled with
        {L: [per-conv [n_tokens, d] CPU bf16]}, the content-token states the mean is taken over."""

        views = [self.view(c) for c in convs]
        ref = views[0].ids
        pre = min(v.start for v in views)
        for v in views[1:]:
            k = 0
            while k < pre and v.ids[k] == ref[k]:
                k += 1
            pre = k
        if pre < min_prefix:
            if token_out is not None and batch is None:  # per-conv reads give the token states
                with self.depth(layers):  # fresh cache state: safe to truncate
                    return self.pooled_incremental(convs, layers, {}, token_out=token_out,
                                                   token_keep=token_keep)
            return self.pooled(convs, layers, token_out=token_out, token_keep=token_keep,
                               batch=batch)
        if token_out is not None:
            for L in layers:
                token_out.setdefault(L, [])
        with self.depth(layers):
            return self._shared_prefix_reads(views, layers, pre, token_out, token_keep, batch)

    def _shared_prefix_reads(self, views, layers, pre, token_out, token_keep, batch):
        d, ref = self.policy.device, views[0].ids
        out = self.model(input_ids=torch.tensor([ref[:pre]], device=d), use_cache=True,
                         logits_to_keep=1)
        base = out.past_key_values
        parts: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        bs = self._bs(batch)
        srt = self.sort_reads
        order = (sorted(range(len(views)), key=lambda q: len(views[q].ids)) if srt
                 else list(range(len(views))))
        toks: dict[int, list] = {L: [] for L in layers}
        for i in range(0, len(views), bs):
            chunk = [View(views[q].ids[pre:], views[q].start - pre, views[q].n_tail)
                     for q in order[i : i + bs]]
            ids, att, mask, _ = self._batch(chunk)
            cache = copy.deepcopy(base)
            cache.reorder_cache(torch.zeros(len(chunk), dtype=torch.long, device=d))
            att = torch.cat([torch.ones(len(chunk), pre, dtype=att.dtype, device=d), att], 1)
            store: dict[int, torch.Tensor] = {}
            with self._hooks(layers, store, keep_grad=False):
                self.model(input_ids=ids, attention_mask=att, past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            m = mask.float()[..., None]
            cnt = m.sum(1).clamp(min=1)
            for L in layers:
                parts[L].append((store[L].float() * m).sum(1) / cnt)
                if token_out is not None and srt:  # one device -> host copy, split on the host
                    flat = store[L][mask].to("cpu", torch.bfloat16)
                    toks[L] += [keep_tokens(t, token_keep)
                                for t in flat.split(mask.sum(1).tolist())]
                elif token_out is not None:
                    for r in range(len(chunk)):
                        token_out[L].append(keep_tokens(store[L][r][mask[r]], token_keep))
        if not srt:
            return {L: torch.cat(v) for L, v in parts.items()}
        inv = torch.empty(len(order), dtype=torch.long)
        inv[torch.tensor(order)] = torch.arange(len(order))
        if token_out is not None:
            for L in layers:
                token_out[L] += [toks[L][j] for j in inv.tolist()]
        return {L: torch.cat(v)[inv.to(v[0].device)] for L, v in parts.items()}

    @torch.no_grad()
    def pooled_incremental(
        self, convs: list[list[dict]], layers: list[int], state: dict,
        token_out: dict | None = None, token_keep: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """pooled() for conversations that only GROW (a lockstep rollout): conversation i keeps
        its own KV / recurrent cache in `state[i]` (caller-owned; one dict per rollout; "hits"
        counts cache reuses), and each call runs only the tokens added since the last call. A
        conversation whose tokens do not extend the cached ones (or whose read span starts inside
        them) is re-run from scratch. Same reads as pooled() up to kernel numerics.
        token_out: as in pooled_shared_prefix (this call's per-conv content-token states)."""

        d = self.policy.device
        parts: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        if token_out is not None:
            for L in layers:
                token_out.setdefault(L, [])
        for i, conv in enumerate(convs):
            v = self.view(conv)
            prev = state.get(i)
            n, cache = 0, None
            if prev is not None:
                pids, pcache = prev
                if len(pids) <= v.start and v.ids[: len(pids)] == pids:
                    n, cache = len(pids), pcache
                    state["hits"] = state.get("hits", 0) + 1
            store: dict[int, torch.Tensor] = {}
            with self._hooks(layers, store, keep_grad=False):
                out = self.model(input_ids=torch.tensor([v.ids[n:]], device=d),
                                 past_key_values=cache, use_cache=True, logits_to_keep=1)
            state[i] = (list(v.ids), out.past_key_values)
            for L in layers:
                tok = store[L][0, v.start - n : v.end - n]
                parts[L].append(tok.float().mean(0, keepdim=True))
                if token_out is not None:
                    token_out[L].append(keep_tokens(tok, token_keep))
        return {L: torch.cat(x) for L, x in parts.items()}
