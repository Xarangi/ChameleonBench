"""Track C1 white-box soft-prompt gradient attack against a frozen concept probe.

Scientific role (the "gradient bridge"). Our arms-race attacker is a *triggered
finetune*: it never sees the eval probe's gradients, so it can only obfuscate
through whatever correlation the trigger token carries. That places it at the
*convergence* end of a spectrum. The paper's thesis needs the *other* end -- a
WHITE-BOX attacker with direct gradient access to a frozen, converged probe --
so we can state the boundary precisely:

    a cheap converged-probe defense beats trigger-based obfuscation but loses to
    gradient-based obfuscation.

This module builds that white-box end. Given a model and a frozen converged
concept probe (:class:`FrozenLinearConceptProbe`, and equally
:class:`FrozenMLPConceptProbe` / :class:`FrozenEnsembleConceptProbe` -- anything
exposing ``score_tokens`` / ``score_sequence``), it optimizes a SOFT PROMPT: a
small number of trainable prefix embeddings prepended to the input embeddings.
The prefix is trained to DRIVE THE PROBE SCORE DOWN on concept-positive inputs
while a behavior/faithfulness constraint holds (KL of the next-token
distribution to the base model, or CE on the original continuation, kept small).
This is the classic soft-embedding obfuscation attack in the Bailey et al.
"Obfuscated Activations" lineage.

Design: the differentiable CORE (:func:`optimize_against_probe`) is factored to
be exercised WITHOUT a real LLM -- it takes a trainable tensor and an objective
closure that returns a probe score plus a behavior penalty, and it is what
``tests/test_gradient_attack.py`` drives on a tiny synthetic probe. The
model-backed wrapper (:func:`run_gradient_attack`) merely builds that objective
closure from a real forward pass and calls the same core.

Everything is fp32 at the scoring boundary (matching the probe heads, which
promote to fp32 so the obfuscation gradient stays well conditioned); the soft
prefix is a fp32 leaf tensor cast to the model dtype only when it enters the
graph, so the optimizer never has to update bf16 parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from next_chameleons.hf.imports import require_torch
from next_chameleons.hf.paper_probes import generation_mask_from_attention

# ---------------------------------------------------------------------------
# Differentiable core (no LLM required -- the correctness anchor).
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveOutputs:
    """One evaluation of the attack objective.

    ``probe_score`` is the scalar the attacker MINIMIZES (mean sequence-level
    detection probability over the concept-positive batch). ``behavior_penalty``
    is the scalar faithfulness constraint (KL to base, CE on the continuation, or
    -- in the synthetic anchor -- an L2 leash to the starting activation). Both
    are differentiable torch scalars w.r.t. the trainable tensor.
    """

    probe_score: Any
    behavior_penalty: Any


@dataclass
class AttackStepTrace:
    """Per-step scalars recorded during optimization (all Python floats)."""

    step: int
    probe_score: float
    behavior_penalty: float
    loss: float


def optimize_against_probe(
    trainable: Any,
    objective_fn: Callable[[], ObjectiveOutputs],
    *,
    steps: int = 100,
    lr: float = 0.05,
    behavior_weight: float = 1.0,
    optimizer_cls: Any = None,
    grad_clip: float | None = None,
    log_every: int | None = None,
) -> dict[str, Any]:
    """Gradient-descend ``trainable`` to minimize ``probe_score + w * behavior``.

    This is the whole attack, model-agnostic. ``trainable`` is a leaf tensor with
    ``requires_grad=True`` (a soft-prompt prefix in the real path; a raw
    activation tensor in the synthetic test). ``objective_fn`` closes over
    ``trainable`` and returns an :class:`ObjectiveOutputs`; it is re-evaluated
    every step so the graph is rebuilt against the updated tensor.

    Returns a dict with the pre/post probe score, the pre/post behavior penalty,
    the full per-step ``trace``, and the detached final ``trainable`` tensor. The
    "before" numbers are read at step 0 (initial tensor, before any update); the
    "after" numbers are a fresh no-grad evaluation of the final tensor.
    """

    torch = require_torch()
    if not getattr(trainable, "requires_grad", False):
        raise ValueError("`trainable` must be a leaf tensor with requires_grad=True")
    optimizer_cls = optimizer_cls or torch.optim.Adam
    optimizer = optimizer_cls([trainable], lr=lr)

    trace: list[AttackStepTrace] = []
    probe_score_before: float | None = None
    behavior_before: float | None = None

    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        outputs = objective_fn()
        probe_score = outputs.probe_score
        behavior_penalty = outputs.behavior_penalty
        loss = probe_score + behavior_weight * behavior_penalty
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([trainable], float(grad_clip))
        optimizer.step()

        score_val = float(probe_score.detach())
        behavior_val = float(behavior_penalty.detach())
        loss_val = float(loss.detach())
        if step == 0:
            probe_score_before = score_val
            behavior_before = behavior_val
        if log_every and (step % int(log_every) == 0 or step == int(steps) - 1):
            print(
                f"[gradient-attack] step={step} probe_score={score_val:.4f} "
                f"behavior_penalty={behavior_val:.4f} loss={loss_val:.4f}",
                flush=True,
            )
        trace.append(
            AttackStepTrace(
                step=step,
                probe_score=score_val,
                behavior_penalty=behavior_val,
                loss=loss_val,
            )
        )

    with torch.no_grad():
        final = objective_fn()
        probe_score_after = float(final.probe_score.detach())
        behavior_after = float(final.behavior_penalty.detach())

    if probe_score_before is None:  # steps == 0: no update happened
        with torch.no_grad():
            initial = objective_fn()
            probe_score_before = float(initial.probe_score.detach())
            behavior_before = float(initial.behavior_penalty.detach())

    return {
        "probe_score_before": probe_score_before,
        "probe_score_after": probe_score_after,
        "probe_score_reduction": probe_score_before - probe_score_after,
        "behavior_penalty_before": behavior_before,
        "behavior_penalty_after": behavior_after,
        "steps": int(steps),
        "trace": trace,
        "trainable": trainable.detach().clone(),
    }


# ---------------------------------------------------------------------------
# Example normalization + tokenization for the model-backed path.
# ---------------------------------------------------------------------------


def _example_prompt_response(example: Any) -> tuple[str, str]:
    """Split a concept example into (prompt, response).

    Accepts a ``PaperConceptExample`` (``.prompt`` / ``.response``), a plain
    string (treated as an all-response text with no prompt), or a mapping with
    ``prompt``/``response`` (or a single ``text``) keys.
    """

    prompt = getattr(example, "prompt", None)
    response = getattr(example, "response", None)
    if prompt is not None or response is not None:
        return str(prompt or ""), str(response or "")
    if isinstance(example, str):
        return "", example
    if isinstance(example, dict):
        if "prompt" in example or "response" in example:
            return str(example.get("prompt", "")), str(example.get("response", ""))
        if "text" in example:
            return "", str(example["text"])
    raise TypeError(
        "concept example must be a PaperConceptExample, str, or dict with "
        f"prompt/response/text; got {type(example)!r}"
    )


@dataclass
class EncodedBatch:
    """A tokenized, padded concept-positive batch with a response-token mask."""

    input_ids: Any  # [B, T] long
    attention_mask: Any  # [B, T] long
    response_mask: Any  # [B, T] bool -- True on response (generation) tokens


def _encode_concept_batch(
    tokenizer: Any,
    examples: list[Any],
    *,
    max_length: int,
    device: Any,
) -> EncodedBatch:
    """Tokenize + pad examples and build a response-token mask.

    The response mask marks the generation tokens the probe reads: for each row
    we tokenize the ``prompt + "\\n"`` prefix to count its tokens, then mark every
    token at or beyond that offset (accounting for left padding via
    :func:`generation_mask_from_attention`). Prompt/trigger tokens are excluded so
    the attack drives down the score exactly where detection is measured.
    """

    torch = require_torch()
    prompts_responses = [_example_prompt_response(ex) for ex in examples]
    prefixes = [f"{p}\n" if p.strip() else "" for p, _ in prompts_responses]
    texts = [
        f"{prefix}{resp}" if prefix else resp.strip()
        for prefix, (_, resp) in zip(prefixes, prompts_responses, strict=True)
    ]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_lengths = torch.tensor(
        [
            len(tokenizer(prefix, add_special_tokens=True).input_ids) if prefix else 0
            for prefix in prefixes
        ],
        device=device,
    )
    response_mask = generation_mask_from_attention(attention_mask, prompt_lengths)
    return EncodedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask.bool(),
    )


def _init_soft_prompt(
    embedding_weight: Any,
    *,
    n_prefix: int,
    init: str,
    init_scale: float,
    seed: int,
) -> Any:
    """Initialize the trainable soft-prompt prefix as a fp32 leaf tensor.

    ``"random_tokens"`` (default) seeds each prefix slot from a real vocabulary
    token embedding, keeping the prefix in-distribution; ``"gaussian"`` samples
    from the embedding matrix's own mean/std. Either way the returned tensor is
    fp32 with ``requires_grad=True`` so Adam updates it in full precision.
    """

    torch = require_torch()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = embedding_weight.detach().to("cpu", torch.float32)
    vocab_size, hidden = matrix.shape
    if init == "random_tokens":
        token_ids = torch.randint(0, vocab_size, (int(n_prefix),), generator=generator)
        base = matrix[token_ids].clone()
        noise = torch.randn(int(n_prefix), hidden, generator=generator) * float(init_scale)
        prefix = base + noise * matrix.std()
    elif init == "gaussian":
        mean = matrix.mean(dim=0, keepdim=True)
        std = matrix.std(dim=0, keepdim=True).clamp_min(1e-6)
        noise = torch.randn(int(n_prefix), hidden, generator=generator)
        prefix = mean + noise * std * float(init_scale)
    else:
        raise ValueError(
            f"Unknown soft-prompt init {init!r}; expected 'random_tokens' or 'gaussian'"
        )
    return prefix.to(embedding_weight.device).clone().detach().requires_grad_(True)


# ---------------------------------------------------------------------------
# Behavior/faithfulness penalties.
# ---------------------------------------------------------------------------


def _kl_penalty(prefixed_logits: Any, base_logits: Any, response_mask: Any, n_prefix: int) -> Any:
    """Mean KL(base || prefixed) of the next-token distribution over response tokens.

    The soft prefix shifts positions by ``n_prefix``, so prefixed logits at
    position ``n_prefix + i`` align with base logits at position ``i``. Keeping
    this KL small means the prefix hides the concept from the probe WITHOUT
    changing what the model would say.
    """

    torch = require_torch()
    aligned = prefixed_logits[:, n_prefix:, :].float()
    base = base_logits.float()
    log_p = torch.log_softmax(base, dim=-1)
    log_q = torch.log_softmax(aligned, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)  # [B, T]
    mask = response_mask.float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


def _ce_penalty(prefixed_logits: Any, input_ids: Any, response_mask: Any, n_prefix: int) -> Any:
    """Mean CE of the prefixed model predicting the ORIGINAL continuation tokens.

    Position ``n_prefix + i`` predicts token ``i + 1``; the penalty is the
    teacher-forcing loss on the real response tokens, so a small value means the
    prefix has not destroyed the model's ability to produce the same text.
    """

    torch = require_torch()
    aligned = prefixed_logits[:, n_prefix:, :].float()
    logits = aligned[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = response_mask[:, 1:].float()
    vocab = logits.shape[-1]
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Model-backed white-box attack.
# ---------------------------------------------------------------------------


@dataclass
class GradientAttackConfig:
    """Resolved hyper-parameters recorded in the attack report."""

    n_prefix: int
    steps: int
    lr: float
    behavior_weight: float
    behavior_constraint: str
    layer: int
    hidden_states_offset: int
    max_examples: int
    max_length: int
    init: str
    init_scale: float
    seed: int
    extra: dict[str, Any] = field(default_factory=dict)


def run_gradient_attack(
    model: Any,
    tokenizer: Any,
    probe: Any,
    concept_examples: list[Any],
    *,
    n_prefix: int = 8,
    steps: int = 100,
    lr: float = 0.05,
    behavior_weight: float = 1.0,
    behavior_constraint: str = "kl",
    layer: int | None = None,
    hidden_states_offset: int = 0,
    max_examples: int = 16,
    max_length: int = 256,
    init: str = "random_tokens",
    init_scale: float = 0.1,
    grad_clip: float | None = 1.0,
    seed: int = 17,
    log_every: int | None = 25,
    return_prefix: bool = False,
) -> dict[str, Any]:
    """White-box soft-prompt attack on a frozen converged concept ``probe``.

    Optimizes ``n_prefix`` trainable prefix embeddings, prepended to the input
    embeddings of ``concept_examples`` (concept-POSITIVE inputs), to minimize the
    probe's mean sequence score at ``layer`` while a behavior constraint holds:

    - ``behavior_constraint="kl"``: KL of the next-token distribution to the
      prefix-free base model (default; the faithfulness leash of the obfuscated-
      activations attack).
    - ``behavior_constraint="ce"``: teacher-forced CE on the original response
      tokens.

    ``probe`` is any frozen head exposing ``score_sequence`` (linear, MLP, or
    ensemble). ``layer`` defaults to ``probe.layer``; the hidden state read is
    ``hidden_states[layer + hidden_states_offset]`` to match the capture
    convention in ``real/probe_bank.py``.

    Returns a dict with ``probe_score_before`` / ``probe_score_after`` (evasion
    read at held behavior), the behavior penalty before/after, the per-step
    ``trace``, a ``prefix_summary``, and the resolved ``config``. Pass
    ``return_prefix=True`` to also return the learned prefix tensor (CPU).
    """

    torch = require_torch()
    if behavior_constraint not in {"kl", "ce"}:
        raise ValueError(f"behavior_constraint must be 'kl' or 'ce'; got {behavior_constraint!r}")
    if not concept_examples:
        raise ValueError("run_gradient_attack requires at least one concept example")

    probe_layer = int(probe.layer if layer is None else layer)
    read_layer = probe_layer + int(hidden_states_offset)

    embeddings = model.get_input_embeddings()
    embed_weight = embeddings.weight
    device = embed_weight.device
    model_dtype = embed_weight.dtype

    examples = list(concept_examples)[: int(max_examples)]
    batch = _encode_concept_batch(tokenizer, examples, max_length=max_length, device=device)

    prefix = _init_soft_prompt(
        embed_weight, n_prefix=n_prefix, init=init, init_scale=init_scale, seed=seed
    )

    # Response mask extended over the prepended prefix positions (prefix tokens
    # are never response tokens, so they read 0).
    prefix_pad = torch.zeros(
        batch.response_mask.shape[0], int(n_prefix), dtype=torch.bool, device=device
    )
    full_response_mask = torch.cat([prefix_pad, batch.response_mask], dim=1)
    prefix_attn = torch.ones(
        batch.attention_mask.shape[0],
        int(n_prefix),
        dtype=batch.attention_mask.dtype,
        device=device,
    )
    full_attention_mask = torch.cat([prefix_attn, batch.attention_mask], dim=1)

    with torch.no_grad():
        token_embeds = embeddings(batch.input_ids)
        base_logits = None
        if behavior_constraint == "kl":
            base_out = model(
                inputs_embeds=token_embeds,
                attention_mask=batch.attention_mask,
                output_hidden_states=False,
            )
            base_logits = base_out.logits.detach()

    def objective_fn() -> ObjectiveOutputs:
        token_embeds_step = embeddings(batch.input_ids)
        prefix_expanded = prefix.to(model_dtype).unsqueeze(0).expand(
            token_embeds_step.shape[0], -1, -1
        )
        inputs_embeds = torch.cat([prefix_expanded, token_embeds_step], dim=1)
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[read_layer]
        probe_score = probe.score_sequence(hidden, full_response_mask).mean()
        if behavior_constraint == "kl":
            behavior = _kl_penalty(outputs.logits, base_logits, batch.response_mask, int(n_prefix))
        else:
            behavior = _ce_penalty(
                outputs.logits, batch.input_ids, batch.response_mask, int(n_prefix)
            )
        return ObjectiveOutputs(probe_score=probe_score, behavior_penalty=behavior)

    result = optimize_against_probe(
        prefix,
        objective_fn,
        steps=steps,
        lr=lr,
        behavior_weight=behavior_weight,
        grad_clip=grad_clip,
        log_every=log_every,
    )

    learned = result.pop("trainable")
    result["behavior_constraint"] = behavior_constraint
    result["concept"] = getattr(probe, "concept", None)
    result["num_examples"] = len(examples)
    result["prefix_summary"] = {
        "n_prefix": int(n_prefix),
        "hidden_dim": int(learned.shape[-1]),
        "l2_norm": float(learned.float().norm()),
        "mean_abs": float(learned.float().abs().mean()),
    }
    result["config"] = GradientAttackConfig(
        n_prefix=int(n_prefix),
        steps=int(steps),
        lr=float(lr),
        behavior_weight=float(behavior_weight),
        behavior_constraint=behavior_constraint,
        layer=probe_layer,
        hidden_states_offset=int(hidden_states_offset),
        max_examples=int(max_examples),
        max_length=int(max_length),
        init=init,
        init_scale=float(init_scale),
        seed=int(seed),
    )
    if return_prefix:
        result["prefix"] = learned.detach().to("cpu")
    return result


def run_gradient_attack_from_pretrained(
    hf_id: str,
    probe: Any,
    concept_examples: list[Any],
    *,
    dtype: str = "bfloat16",
    quantization: str = "none",
    device_map: str = "auto",
    **attack_kwargs: Any,
) -> dict[str, Any]:
    """Load ``hf_id`` with the standard loader and run :func:`run_gradient_attack`.

    Thin convenience wrapper reusing ``load_causal_lm`` so a caller can go from a
    model id + a converged probe straight to an attack report. The heavy import
    (``load_causal_lm``) is deferred to call time to keep this module import-clean
    and offline for the synthetic core tests.
    """

    from next_chameleons.hf.models import load_causal_lm

    loaded = load_causal_lm(
        hf_id=hf_id, dtype=dtype, quantization=quantization, device_map=device_map
    )
    loaded.model.eval()
    for param in loaded.model.parameters():
        param.requires_grad_(False)
    return run_gradient_attack(
        loaded.model, loaded.tokenizer, probe, concept_examples, **attack_kwargs
    )


__all__ = [
    "AttackStepTrace",
    "EncodedBatch",
    "GradientAttackConfig",
    "ObjectiveOutputs",
    "optimize_against_probe",
    "run_gradient_attack",
    "run_gradient_attack_from_pretrained",
]
