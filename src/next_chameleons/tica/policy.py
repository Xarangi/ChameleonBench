"""The policy model for TICA: one wrapper used for evaluation, rollouts, and training.

A *view* is what the model computes while writing the final message of a conversation:
- ids = tokenize(generation prompt of conv[:-1]) + tokenize(conv[-1].content);
- the content span is [start, len(ids)).
Templates can render a finished turn differently from the generation prompt: Qwen3.5 and Gemma-4
drop the empty think block (verified 10-03). So a view is NOT a slice of the rendered conversation.
Earlier turns appear as the template renders history, exactly as in real multi-turn use.

Capabilities:
- pooled(): mean over the content span of decoder-layer outputs at any set of layers. Layer L is
  the output of decoder block L (= HF hidden_states[L+1], the NC/Kowalski convention), read by a
  forward hook. With grad=True the activations stay in the graph.
- forward(): one pass returning the pooled activations plus the per-view mean NLL of the content
  tokens (the LM-retention term).
- reference(): a context in which the LoRA adapters are disabled, i.e. the base model.
- sample() / greedy(): generation, returning text, token ids, and whether EOS was reached.
The model and tokenizer are injected, so tests run on a tiny random model.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

DECODER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "base_model.model.model.layers",
    "base_model.model.model.language_model.layers",
)


def _keep_tokens(tok: torch.Tensor, keep: int | None) -> torch.Tensor:
    """CPU bf16 copy of a span's token states; with `keep`, an evenly spaced subset of at most
    `keep` tokens (multi-layer token ensembles: an unbiased-in-position estimate of token means)."""

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
    tail bucket}; lp = policy log-probs [n_tok, V] (with grad). By the data-processing inequality
    this is <= the full KL, and = it when the tail is empty."""

    i, v, tail = ref
    lpk = lp.gather(-1, i)
    p_tail = torch.log1p(-lpk.exp().sum(-1).clamp(max=1 - 1e-6))
    return (v.exp() * (v - lpk)).sum(-1) + tail.exp() * (tail - p_tail)


def _span_extra(extra: dict | None, lp: torch.Tensor, counts: list[int]) -> None:
    """Fill `extra` from per-token log-probs lp (spans / views contiguous, sizes `counts`):
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


def _lm_terms(head, h: torch.Tensor, tok: torch.Tensor, counts: list[int], extra: dict | None,
              chunk: int) -> torch.Tensor:
    """Chunked LM terms (model.lm_chunk > 0; TamIA 10-21): what forward / forward_multi compute from
    the full [n_tok, V] log-probs, without ever holding them. h = post-norm hidden states at the
    predicting positions [n_tok, d] (with grad), tok = their target ids, counts = span sizes (as
    _span_extra). Returns the per-token NLL [n_tok] and fills `extra` exactly as _span_extra does;
    each chunk's logits are recomputed in backward (activation checkpointing), so the peak is one
    chunk of logits instead of all of them (the 7.5 GB allocations at 24 turns, layout smoke L2)."""

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
        lp = torch.log_softmax(head(hc).float(), -1)
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


@dataclass
class View:
    ids: list[int]
    start: int  # first content token
    n_tail: int = 0  # end-of-turn token(s) after the content: in the NLL, not in the probe read

    @property
    def end(self) -> int:
        return len(self.ids) - self.n_tail


@dataclass
class MultiView:
    """A whole conversation tokenised once (keep-think history), with every assistant turn's
    (content start, content end, end of its end-of-turn token). Turn t's per-turn view is the prefix
    ids[:tail_end_t] -- so one forward reads every turn exactly as the per-turn views do."""

    ids: list[int]
    spans: list[tuple[int, int, int]]

    @property
    def n_content(self) -> int:
        return self.end - self.start


def configure_linear_attention(model, fla_grad: bool) -> int:
    """Qwen3.5 gated-DeltaNet layers: use the fla kernel for no-grad passes always, and for
    grad passes only if fla_grad (fla 0.4.1's backward does not compile under triton 3.2; its
    forward matches the torch fallback to cos >= 0.99995 at L8-20, 2x faster: job 5139300).
    Returns the number of layers patched (0 = no fla kernel, nothing to do)."""

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    except ImportError:
        return 0
    if fla_grad:  # fla's backward needs triton >= 3.3 (10-09g); older envs: torch path for grads
        try:
            import triton

            if tuple(int(x) for x in triton.__version__.split(".")[:2]) < (3, 3):
                print(f"[warn] fla_grad needs triton >= 3.3 (have {triton.__version__}); "
                      "grad passes use the torch fallback", flush=True)
                fla_grad = False
        except ImportError:
            fla_grad = False
    n = 0
    for m in model.modules():
        fast = getattr(m, "chunk_gated_delta_rule", None)
        if fast is None or fast is mq.torch_chunk_gated_delta_rule or fla_grad:
            continue

        def pick(*a, _fast=fast, **k):
            fn = mq.torch_chunk_gated_delta_rule if torch.is_grad_enabled() else _fast
            return fn(*a, **k)

        m.chunk_gated_delta_rule = pick
        n += 1
    return n


def find_decoder_layers(model):
    for path in DECODER_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("cannot locate decoder layers")


class Policy:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        enable_thinking: bool = False,
        batch_size: int = 8,
        history_reasoning: bool = False,
        keep_think_history: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.batch_size = batch_size
        # Keep each earlier assistant turn's reasoning (message key "reasoning") in the rendered
        # history. The Qwen3.5 template drops reasoning before the last user message, so without
        # this a multi-turn model cannot see what it tried on earlier turns (10-11).
        self.history_reasoning = history_reasoning
        # Render EVERY assistant turn with the empty think block the generation prompt uses (think
        # off), so a whole conversation contains each turn exactly as its per-turn view did and one
        # forward reads all turns (single-pass training; REPLICATION 2026-10-12f). Base probes agree
        # 1.00 with this form on the base model (5380273).
        self.keep_think_history = keep_think_history
        if keep_think_history and enable_thinking:
            raise ValueError("keep_think_history needs enable_thinking=False")
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.layers_module = find_decoder_layers(model)
        self.eos_ids = self._eos_ids()
        # TRUNCATED READS (REPLICATION 2026-10-16c; default off): no-grad probe reads stop the
        # forward after the deepest requested layer. Exact: layer L's output does not depend on
        # later layers. Only for models whose decoder loop is bounded by config.num_hidden_layers
        # (Qwen3.5); otherwise a no-op.
        self.truncate_reads = False
        self._depth_owner = next((m for m in model.modules()
                                  if getattr(m, "layers", None) is self.layers_module
                                  and hasattr(getattr(m, "config", None), "num_hidden_layers")),
                                 None)

    # ------------------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        dtype: str = "bfloat16",
        lora: dict | None = None,
        adapter: str | None = None,
        trainable: bool = False,
        grad_ckpt: bool = True,
        fla_grad: bool = False,
        **kw,
    ) -> Policy:
        """Load a model. `lora` (rank, alpha, dropout, target_modules) attaches a fresh adapter;
        `adapter` loads a saved one. trainable=True enables gradient checkpointing unless
        grad_ckpt=False (faster, more activation memory)."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=getattr(torch, dtype), device_map="auto"
        )
        if adapter or lora:
            import peft

            if adapter:
                model = peft.PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
                adapter_path = str(adapter)
            else:
                cfg = peft.LoraConfig(
                    r=lora.get("rank", 16),
                    lora_alpha=lora.get("alpha", 32),
                    lora_dropout=lora.get("dropout", 0.0),
                    target_modules=lora.get("target_modules", "all-linear"),
                    task_type="CAUSAL_LM",
                )
                model = peft.get_peft_model(model, cfg)
        configure_linear_attention(model, fla_grad)
        if trainable:
            if grad_ckpt:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                model.enable_input_require_grads()
            model.train()
        else:
            model.eval()
        pol = cls(model, tok, **kw)
        pol.adapter_path = locals().get("adapter_path")  # recorded in each eval's run.json
        return pol

    def save_adapter(self, path: Path | str) -> None:
        self.model.save_pretrained(str(path))

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def _eos_ids(self) -> set[int]:
        ids = set()
        e = getattr(self.tokenizer, "eos_token_id", None)
        ids |= set(e) if isinstance(e, list) else {e}
        gen = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        ids |= set(gen) if isinstance(gen, list) else {gen}
        return {i for i in ids if i is not None}

    # ------------------------------------------------------------------------------ views
    def render(self, conv: list[dict], add_generation_prompt: bool) -> str:
        from next_chameleons.tica.answer_format import split_system

        conv = split_system(conv)  # tagged layout: system part of the first user turn (10-17a)
        if self.keep_think_history:
            conv = [
                dict(m, reasoning=m.get("reasoning") or "") if m["role"] == "assistant" else m
                for m in conv
            ]
            return render_keep_reasoning(conv, add_generation_prompt, self.enable_thinking)
        if self.history_reasoning and any(
            m["role"] == "assistant" and m.get("reasoning") is not None for m in conv
        ):
            return render_keep_reasoning(conv, add_generation_prompt, self.enable_thinking)
        return self.tokenizer.apply_chat_template(
            [{"role": m["role"], "content": m["content"]} for m in conv],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def view(self, conv: list[dict]) -> View:
        """ids = generation prompt + content + the first token of the template's end-of-turn
        marker. The marker is part of the LM target (so training keeps the model stopping after the
        text; smoke 4545558 drifted into appending content without it) but not of the probe read."""

        if conv[-1]["role"] != "assistant":
            raise ValueError("a view needs a final assistant message")
        if self.keep_think_history:  # a prefix of the whole-conversation tokenisation
            mv = self.multi_view(conv)
            st, en, te = mv.spans[-1]
            return View(mv.ids[:te], st, te - en)
        p = self.encode(self.render(conv[:-1], True))
        tail = self.turn_end(conv)
        return View(p + self.encode(conv[-1]["content"]) + tail, len(p), len(tail))

    def multi_view(self, conv: list[dict]) -> MultiView:
        """keep_think_history only. Segment-wise tokenisation (template text / each assistant
        content separately) so every turn's span and prefix are identical to its per-turn view."""

        if not self.keep_think_history:
            raise ValueError("multi_view needs keep_think_history=True")
        full = self.render(conv, False)
        head = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        end_tok = self.encode("<|im_end|>")[:1]
        ids: list[int] = []
        spans: list[tuple[int, int, int]] = []
        pos = 0
        for m in conv:
            if m["role"] != "assistant":
                continue
            a = full.find(head, pos)
            if a < 0:
                raise ValueError("assistant turn not found in keep-think render")
            k = a + len(head)
            content = m["content"].strip()
            if full[k : k + len(content)] != content:
                raise ValueError("assistant content misaligned in keep-think render")
            ids += self.encode(full[pos:k])
            st = len(ids)
            ids += self.encode(content)
            en = len(ids)
            spans.append((st, en, en + len(end_tok)))
            pos = k + len(content)
        rest = self.encode(full[pos:])
        if end_tok and rest[: len(end_tok)] != end_tok:
            raise ValueError("end-of-turn token not where expected")
        return MultiView(ids + rest, spans)

    def turn_end(self, conv: list[dict]) -> list[int]:
        """First token of what the template writes right after the final assistant content."""

        content = conv[-1]["content"]
        full = self.render(conv, False)
        k = full.rfind(content)
        if not content or k < 0:
            return []
        rest = full[k + len(content) :]
        return self.encode(rest)[:1] if rest else []

    def prompt_ids(self, conv: list[dict]) -> list[int]:
        """Generation-prompt ids for a conversation ending in a user message."""

        return self.encode(self.render(conv, True))

    # ------------------------------------------------------------------------------ forward
    @contextmanager
    def _hooks(self, layers: list[int], store: dict, keep_grad: bool):
        handles = []
        for L in layers:

            def hook(module, inp, out, L=L):
                h = out[0] if isinstance(out, tuple) else out
                store[L] = h if keep_grad else h.detach()

            handles.append(self.layers_module[L].register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def _depth(self, layers: list[int]):
        """Run only decoder layers 0..max(layers) inside (truncate_reads; see __init__)."""

        own = self._depth_owner
        if not self.truncate_reads or own is None or not layers:
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
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(views), n), pad, dtype=torch.long)
        att = torch.zeros((len(views), n), dtype=torch.long)
        mask = torch.zeros((len(views), n), dtype=torch.bool)  # probe read: content only
        lm = torch.zeros((len(views), n), dtype=torch.bool)  # LM target: content + tail
        for r, v in enumerate(views):
            ids[r, : len(v.ids)] = torch.tensor(v.ids)
            att[r, : len(v.ids)] = 1
            mask[r, v.start : v.end] = True
            lm[r, v.start : len(v.ids)] = True
        d = self.device
        return ids.to(d), att.to(d), mask.to(d), lm.to(d)

    @contextmanager
    def _head_capture(self, box: dict):
        """model.lm_chunk: the LM head's input (post-norm hidden states of the kept positions) goes
        to box["h"] and the head itself only sees the last position, so the model never builds the
        [B, keep, V] logits; _lm_terms computes the LM terms from box["h"] in chunks."""

        head = self.model.get_output_embeddings()

        def pre(_mod, args):
            if "h" in box:  # only the model's own head call; _lm_terms' chunk calls pass through
                return None
            box["h"] = args[0]
            return (args[0][:, -1:],)

        hook = head.register_forward_pre_hook(pre)
        try:
            yield head
        finally:
            hook.remove()

    def forward(
        self, views: list[View], layers: list[int], *, grad: bool, with_nll: bool = False,
        extra: dict | None = None, token_out: dict | None = None, token_keep: int | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
        """One batch (no internal chunking). Returns ({layer: pooled [n, d] float32}, nll [n] or
        None). nll = mean negative log-likelihood of the content tokens. extra (with_nll only):
        KL-anchor targets / KL per view, see _span_extra."""

        ids, att, mask, lm = self._batch(views)
        n = ids.shape[1]
        # Logits only from the first position that scores a content token (min start - 1): the
        # full [B, n, vocab] tensor is several GB at a 248k vocabulary with long histories.
        keep = n - max(0, min(v.start for v in views) - 1) if with_nll else 1
        store: dict[int, torch.Tensor] = {}
        ctx = nullcontext() if grad else torch.no_grad()
        chunked = bool(with_nll and getattr(self, "lm_chunk", 0))
        box: dict = {}
        with ctx, self._hooks(layers, store, keep_grad=grad), (
                self._head_capture(box) if chunked else nullcontext()) as head:
            out = self.model(
                input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=keep
            )
            m = mask.float()[..., None]
            cnt = m.sum(1).clamp(min=1)
            pooled = {L: (store[L].float() * m).sum(1) / cnt for L in layers}
            if token_out is not None:  # per-view content-token states (as pooled_incremental)
                for L in layers:
                    token_out.setdefault(L, []).extend(
                        _keep_tokens(store[L][r][mask[r]], token_keep) for r in range(len(views)))
            nll = None
            if with_nll:
                # Position j predicts token j+1, so content token k is scored by logits at k-1.
                # out.logits covers positions [n - keep, n).
                tgt_mask = lm[:, 1:][:, n - keep :]
                tok = ids[:, 1:][:, n - keep :][tgt_mask]
                if chunked:  # model.lm_chunk: same terms, chunked (_lm_terms)
                    per_tok = _lm_terms(head, box["h"][:, :-1][tgt_mask], tok,
                                        tgt_mask.sum(1).tolist(), extra, self.lm_chunk)
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
                    _span_extra(extra, lp, tgt_mask.sum(1).tolist())
        return pooled, nll

    def forward_multi(
        self, mvs: list[MultiView], layers: list[int], *, grad: bool, with_nll: bool = False,
        token_out: dict | None = None, extra: dict | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
        """ONE forward per conversation; returns pooled activations for EVERY span of every
        MultiView ([sum(len(spans)), d] per layer, conversation-major, turn order) and the per-span
        mean NLL of content + end-of-turn tokens -- identical (by causality) to forward() on each
        turn's per-turn view, at the cost of a single pass (REPLICATION 2026-10-12f).
        token_out (optional): filled with {L: [per-span [n_tokens, d] float states]} in the same
        order as the pooled rows (WITH grad when grad=True) for token-level probes.
        extra (with_nll only): KL-anchor targets / KL per span (_span_extra)."""

        n = max(len(m.ids) for m in mvs)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(mvs), n), pad, dtype=torch.long)
        att = torch.zeros((len(mvs), n), dtype=torch.long)
        for r, m in enumerate(mvs):
            ids[r, : len(m.ids)] = torch.tensor(m.ids)
            att[r, : len(m.ids)] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        first = min(m.spans[0][0] for m in mvs)
        keep = n - max(0, first - 1) if with_nll else 1
        store: dict[int, torch.Tensor] = {}
        ctx = nullcontext() if grad else torch.no_grad()
        chunked = bool(with_nll and getattr(self, "lm_chunk", 0))
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
                if chunked:  # model.lm_chunk: same terms, chunked (_lm_terms)
                    counts = [te - a for m in mvs for a, _, te in m.spans]
                    per_tok = _lm_terms(head, box["h"][rows_t, pos_t - 1 - (n - keep)],
                                        ids[rows_t, pos_t], counts, extra, self.lm_chunk)
                else:
                    lp = torch.log_softmax(out.logits[rows_t, pos_t - 1 - (n - keep)].float(),
                                           -1)
                    per_tok = -lp.gather(-1, ids[rows_t, pos_t][:, None])[:, 0]
                sid = torch.tensor(span_id, device=ids.device)
                tot = torch.zeros(j, device=ids.device).index_add(0, sid, per_tok)
                cnt = torch.zeros(j, device=ids.device).index_add(0, sid, torch.ones_like(per_tok))
                nll = tot / cnt.clamp(min=1)
                if not chunked:
                    _span_extra(extra, lp, cnt.long().tolist())  # spans are contiguous in lp
        return pooled, nll

    def pooled(self, convs: list[list[dict]], layers: list[int], token_out: dict | None = None,
               token_keep: int | None = None, batch: int | None = None,
               ) -> dict[int, torch.Tensor]:
        """No-grad pooled reads for many conversations, in batches (default model batch_size);
        token_out as in pooled_shared_prefix (right padding: content states are exact)."""

        views = [self.view(c) for c in convs]
        bs = batch or self.batch_size
        parts: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        for i in range(0, len(views), bs):
            with self._depth(layers):
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
        """pooled() for forks of one conversation (batch: forks per call, default batch_size): the
        longest common token prefix is run once
        with a KV/recurrent cache and only the per-fork suffixes are batched against copies of it.
        Same reads as pooled() up to kernel numerics; falls back to it for short prefixes.
        token_out (optional dict): filled with {L: [per-conv [n_tokens, d] CPU bf16]} -- the
        content-token states the mean is taken over (max- / attention-pooled probes)."""

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
                with self._depth(layers):  # fresh cache state: safe to truncate
                    return self.pooled_incremental(convs, layers, {}, token_out=token_out,
                                                   token_keep=token_keep)
            return self.pooled(convs, layers, token_out=token_out, token_keep=token_keep,
                               batch=batch)
        if token_out is not None:
            for L in layers:
                token_out.setdefault(L, [])
        with self._depth(layers):
            return self._shared_prefix_reads(views, layers, pre, token_out, token_keep, batch)

    def _shared_prefix_reads(self, views, layers, pre, token_out, token_keep, batch):
        d, ref = self.device, views[0].ids
        out = self.model(input_ids=torch.tensor([ref[:pre]], device=d), use_cache=True,
                         logits_to_keep=1)
        base = out.past_key_values
        parts: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        bs = batch or self.batch_size
        for i in range(0, len(views), bs):
            chunk = [View(v.ids[pre:], v.start - pre, v.n_tail)
                     for v in views[i : i + bs]]
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
                if token_out is not None:
                    for r in range(len(chunk)):
                        token_out[L].append(_keep_tokens(store[L][r][mask[r]], token_keep))
        return {L: torch.cat(v) for L, v in parts.items()}

    @torch.no_grad()
    def pooled_incremental(
        self, convs: list[list[dict]], layers: list[int], state: dict,
        token_out: dict | None = None, token_keep: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """pooled() for conversations that only GROW (a lockstep rollout): conversation i keeps its
        own KV/recurrent cache in `state[i]` (caller-owned; one dict per rollout), and each call
        runs only the tokens added since the last call. Turn-by-turn cost is linear in the
        conversation length instead of quadratic. A conversation whose tokens are not an extension
        of the cached ones (or whose read span starts inside them) is re-run from scratch. Same
        reads as pooled() up to kernel numerics (test_pooled_incremental_matches_pooled).
        token_out: as in pooled_shared_prefix (this call's per-conv content-token states)."""

        d = self.device
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
                    token_out[L].append(_keep_tokens(tok, token_keep))
        return {L: torch.cat(x) for L, x in parts.items()}

    @contextmanager
    def reference(self):
        """The base model (LoRA adapters disabled). Without adapters, the model itself. Inside
        merged() (10-20: turn-0 base-panel fits during the rollout reads) the original base weights
        are copied back for the block (adapters are already disabled there) and re-merged after."""

        if getattr(self, "_merged_active", False):
            with self._unmerged():
                yield
            return
        dis = getattr(self.model, "disable_adapter", None)
        self._ref_depth = getattr(self, "_ref_depth", 0) + 1
        try:
            with dis() if callable(dis) else nullcontext():
                yield
        finally:
            self._ref_depth -= 1

    @contextmanager
    def merged(self):
        """No-grad READ mode with the LoRA merged into the base weights (REPLICATION 2026-10-15d):
        PEFT keeps adapters in fp32, so every adapted Linear casts its input, runs two fp32 matmuls
        and adds back (profile 5660654: ~1/3 of a refit read). Inside, each adapted base weight
        holds W0 + scaling * B @ A (fp32 math, cast to the weight dtype; the same merge as the
        vLLM merged mode) and the adapters are disabled. On exit the ORIGINAL weights are copied
        back from a pinned CPU copy taken once (the base never trains), so training is unchanged
        bit for bit. Reads differ from the adapter path only by bf16 rounding of the merge. Not for
        gradients; reference() inside restores the original weights for its block (_unmerged). A
        no-op without adapters / inside reference()."""

        import torch as _t

        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else None
        if base is None or getattr(self, "_ref_depth", 0) > 0 or getattr(self, "_merged_active",
                                                                         False):
            yield
            return
        mods = [m for m in base.modules()
                if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})
                and hasattr(m, "base_layer")]
        if not hasattr(self, "_w0"):  # original base weights, once (pinned host memory)
            pin = self.device.type == "cuda"
            self._w0 = [(lambda t: t.pin_memory() if pin else t)(
                m.base_layer.weight.detach().to("cpu", copy=True)) for m in mods]
        self._merged_mods = mods
        self._merge_into(mods)
        self._merged_active = True
        try:
            with self.model.disable_adapter():
                yield
        finally:
            with _t.no_grad():
                for m, w0 in zip(mods, self._w0, strict=True):
                    m.base_layer.weight.copy_(w0, non_blocking=True)
            if self.device.type == "cuda":
                _t.cuda.synchronize()
            self._merged_active = False

    def _merge_into(self, mods) -> None:
        """Base weight <- original + scaling * B @ A (fp32 math, cast to the weight dtype)."""

        import torch as _t

        with _t.no_grad():
            for m in mods:
                w = m.base_layer.weight
                d = m.lora_B["default"].weight.float() @ m.lora_A["default"].weight.float()
                w.copy_((w.float() + d * m.scaling["default"]).to(w.dtype))

    @contextmanager
    def _unmerged(self):
        """Inside merged(): the original base weights for the block (the adapters stay disabled,
        so the model IS the base model), merged again on exit (same merge as merged())."""

        import torch as _t

        mods = self._merged_mods
        with _t.no_grad():
            for m, w0 in zip(mods, self._w0, strict=True):
                m.base_layer.weight.copy_(w0, non_blocking=True)
        if self.device.type == "cuda":
            _t.cuda.synchronize()
        self._merged_active = False
        self._ref_depth = getattr(self, "_ref_depth", 0) + 1
        try:
            yield
        finally:
            self._ref_depth -= 1
            self._merge_into(mods)
            self._merged_active = True

    def attach_vllm(self, gen) -> None:
        """Route generate() through a colocated vLLM server (tica/vllm_gen.py). Trainers bump
        `weights_version` after every optimizer step so the served adapter is re-synced."""

        self.vllm = gen
        self.weights_version = getattr(self, "weights_version", 0)

    def _generate_vllm(self, convs, max_new_tokens, temperature, top_p, seed):
        """Policy.generate via vLLM: same prompt token ids, base model inside reference()."""

        prompts = [self.prompt_ids(c) for c in convs]
        base = getattr(self, "_ref_depth", 0) > 0
        gen = self.vllm
        torch.cuda.empty_cache()  # give vLLM back the memory training cached
        gen.wake()
        try:
            if base:
                if hasattr(gen, "use_base"):
                    gen.use_base()
            else:
                gen.sync(self)
            outs = gen.generate(prompts, max_new_tokens, temperature=temperature, top_p=top_p,
                                seed=seed, base=base)
        finally:
            gen.sleep()
        for p, o in zip(prompts, outs, strict=True):
            o["prompt_ids"] = p
        return outs

    # ------------------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(
        self,
        convs: list[list[dict]],
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        stop_ids: set[int] | None = None,
    ) -> list[dict]:
        """temperature 0 = greedy. Returns {text, prompt_ids, ids, finished, stop} per
        conversation. stop_ids end generation like EOS (e.g. the end-of-reasoning token); `stop` is
        the id that ended it (None = token budget reached). Neither EOS nor the stop id is included
        in `ids`."""

        if getattr(self, "vllm", None) is not None and not stop_ids:
            return self._generate_vllm(convs, max_new_tokens, temperature, top_p, seed)
        ends = self.eos_ids | set(stop_ids or ())
        out = []
        pad = self.tokenizer.pad_token_id
        for i in range(0, len(convs), self.batch_size):
            prompts = [self.prompt_ids(c) for c in convs[i : i + self.batch_size]]
            n = max(len(p) for p in prompts)
            ids = torch.full((len(prompts), n), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, p in enumerate(prompts):  # left padding for generation
                ids[r, n - len(p) :] = torch.tensor(p)
                att[r, n - len(p) :] = 1
            kw = {"do_sample": False}
            if temperature > 0:
                torch.manual_seed(seed + i)
                kw = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": 0}
            gen = self.model.generate(
                input_ids=ids.to(self.device),
                attention_mask=att.to(self.device),
                max_new_tokens=max_new_tokens,
                pad_token_id=pad,
                eos_token_id=sorted(ends),
                **kw,
            )
            for p, row in zip(prompts, gen[:, n:].tolist(), strict=True):
                k = next((j for j, t in enumerate(row) if t in ends), len(row))
                text = self.tokenizer.decode(row[:k], skip_special_tokens=True).strip()
                out.append(
                    {
                        "text": text,
                        "prompt_ids": p,
                        "ids": row[:k],
                        "finished": k < len(row),
                        "stop": row[k] if k < len(row) else None,
                    }
                )
        return out


def render_keep_reasoning(conv: list[dict], add_generation_prompt: bool, thinking: bool) -> str:
    """Qwen3.5 / Qwen3 ChatML rendering of a plain system/user/assistant conversation that KEEPS
    every assistant turn's reasoning, in the exact form the template uses for the turns it keeps
    ('<think>\\n' + reasoning + '\\n</think>\\n\\n' + content). Assistant turns without a
    "reasoning" key render as the template renders history (no think block). Checked against the
    real template token for token in tests/test_tica_adaptive.py."""

    out = []
    for i, m in enumerate(conv):
        role, content = m["role"], m["content"].strip()
        if role == "system":
            if i:
                raise ValueError("system message must be first")
            out.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            r = m.get("reasoning")
            body = content if r is None else f"<think>\n{r.strip()}\n</think>\n\n{content}"
            out.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
        else:
            raise ValueError(f"unsupported role {role!r}")
    if add_generation_prompt:
        head = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
        out.append("<|im_start|>assistant\n" + head)
    return "".join(out)
