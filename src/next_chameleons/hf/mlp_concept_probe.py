"""Differentiable frozen NONLINEAR (small MLP) concept Probe head.

This is the Track A3 "nonlinear defender" analogue of
``FrozenLinearConceptProbe`` (``paper_probes.py``). Where the linear probe reads
a single hyperplane over the pooled layer activations, this probe passes the
SAME pooled per-token features through a one-hidden-layer MLP
(``hidden -> width -> 1`` with a ReLU/GELU nonlinearity and a sigmoid output).
That lets a still-cheap defender capture concept structure a linear probe
cannot (curved / XOR-like decision boundaries), so we can test whether an
organism trained to evade the LINEAR probe is still caught by a nonlinear one --
and, symmetrically, use the nonlinear head as a differentiable obfuscation
TARGET so an attacker can backprop an evasion loss through it into the model's
hidden states.

Public interface parity: ``score_tokens(hidden_states)`` and
``score_sequence(hidden_states, generation_mask)`` match
``FrozenLinearConceptProbe`` EXACTLY (same names, same argument order, same
return shapes), so an instance drops into ``FrozenProbeBank`` and the paper
obfuscation loss unchanged. ``concept``, ``layer`` and the ``num_members``
property mirror the linear/ensemble probes so bank manifest / eval code that
introspects those attributes keeps working.

Everything is fp32 for the same reason the linear probe scores in fp32 (see the
comment there): under bf16 autocast the MSE-on-probability obfuscation gradient
quantizes badly, so we promote scoring to fp32 and let autograd cast back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from next_chameleons.hf.imports import require_torch

# Bank serialization format tag for MLP-bearing banks. probe_bank save/load
# should branch on the presence of an ``mlp_w1`` key (see wiring notes at the
# bottom of this module) and stamp this format so old loaders fail loudly rather
# than silently mis-reading an MLP probe as a linear one.
MLP_BANK_FORMAT = "next_chameleons_frozen_probe_bank_v3_mlp"

_ACTIVATIONS = ("relu", "gelu")


def _apply_activation(torch: Any, name: str, x: Any) -> Any:
    if name == "relu":
        return torch.nn.functional.relu(x)
    if name == "gelu":
        return torch.nn.functional.gelu(x)
    raise ValueError(f"Unsupported MLP activation {name!r}; expected one of {_ACTIVATIONS}")


@dataclass(frozen=True)
class FrozenMLPConceptProbe:
    """A frozen one-hidden-layer MLP Probe for one benign concept and layer.

    Parameters mirror an ``nn.Linear(hidden, width) -> act -> nn.Linear(width,
    1) -> sigmoid`` head, stored as raw tensors (like the linear/ensemble probes
    store ``weight``/``bias``) so the probe is a plain frozen dataclass, cheap to
    (de)serialize and to move across devices during scoring:

    - ``w1``: input weight, shape ``[hidden, width]``.
    - ``b1``: hidden bias, shape ``[width]``.
    - ``w2``: output weight, shape ``[width]``.
    - ``b2``: output bias, scalar.
    """

    concept: str
    layer: int
    w1: Any  # [hidden, width]
    b1: Any  # [width]
    w2: Any  # [width]
    b2: Any  # scalar
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported MLP activation {self.activation!r}; expected one of {_ACTIVATIONS}"
            )

    @property
    def num_members(self) -> int:
        # Presents as a single-member probe so bank/eval code that reads
        # ``getattr(probe, "num_members", 1)`` treats it like one head.
        return 1

    @property
    def hidden_width(self) -> int:
        return int(self.w1.shape[1])

    def score_tokens(self, hidden_states: Any) -> Any:
        """Per-token detection score in ``[0, 1]``; differentiable in ``hidden_states``.

        ``hidden_states`` is ``[batch, tokens, hidden]``; returns ``[batch,
        tokens]``. Computed in fp32 (see module docstring), so gradients flow
        cleanly back to the input activations for obfuscation training.
        """

        torch = require_torch()
        h = hidden_states.to(torch.float32)
        w1 = self.w1.to(device=h.device, dtype=torch.float32)
        b1 = self.b1.to(device=h.device, dtype=torch.float32)
        w2 = self.w2.to(device=h.device, dtype=torch.float32)
        b2 = self.b2.to(device=h.device, dtype=torch.float32)
        pre = h @ w1 + b1
        hidden = _apply_activation(torch, self.activation, pre)
        logit = hidden @ w2 + b2
        return torch.sigmoid(logit)

    def score_sequence(self, hidden_states: Any, generation_mask: Any) -> Any:
        """Mask-mean of per-token scores over generation tokens (linear-probe parity)."""

        scores = self.score_tokens(hidden_states)
        mask = generation_mask.to(scores.device).float()
        return (scores * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def score_features(self, features: Any) -> Any:
        """Per-row score for a flat ``[N, hidden]`` pool of pooled token features.

        Convenience for fit / eval quality checks: adds the batch axis, scores,
        and drops it back to ``[N]``. Differentiable, same as ``score_tokens``.
        """

        return self.score_tokens(features[None])[0]

    def state_dict(self) -> dict[str, Any]:
        """Detached CPU tensors + metadata for bank serialization (see wiring notes)."""

        return {
            "concept": self.concept,
            "layer": int(self.layer),
            "activation": self.activation,
            "mlp_w1": self.w1.detach().cpu(),
            "mlp_b1": self.b1.detach().cpu(),
            "mlp_w2": self.w2.detach().cpu(),
            "mlp_b2": self.b2.detach().cpu(),
        }

    @classmethod
    def from_state(cls, item: dict[str, Any]) -> FrozenMLPConceptProbe:
        """Inverse of ``state_dict`` (an ``mlp_w1`` key identifies an MLP probe)."""

        return cls(
            concept=str(item["concept"]),
            layer=int(item["layer"]),
            w1=item["mlp_w1"],
            b1=item["mlp_b1"],
            w2=item["mlp_w2"],
            b2=item["mlp_b2"],
            activation=str(item.get("activation", "relu")),
        )

    @classmethod
    def fit(
        cls,
        *,
        concept: str,
        layer: int,
        token_features: Any,
        token_labels: Any,
        hidden_width: int = 64,
        activation: str = "relu",
        learning_rate: float = 1e-2,
        epochs: int = 300,
        batch_size: int = 64,
        l2: float = 1e-4,
        seed: int = 17,
        min_accuracy: float | None = 0.97,
    ) -> FrozenMLPConceptProbe:
        """Train the MLP head on pooled concept activations, one-vs-rest.

        Mirrors the fit contract of ``fit_linear_probe_per_token`` /
        ``run_probe_bank_fit``: each response token is an independent example,
        the sequence label broadcast to every token, BCE on per-token logits,
        Adam. The caller supplies binary ``token_labels`` already balanced
        upstream (concept-vs-neutral or one-vs-rest, matching the linear path);
        to stay robust to residual class imbalance we additionally weight the
        BCE positive term by ``n_neg / n_pos`` (``pos_weight``) -- the
        balanced-class handling the linear 1-epoch-from-zeros fit gets for free.

        Convergence + gate: unlike the linear probe's single contiguous epoch,
        an MLP is trained to convergence (per-epoch shuffled minibatches, seeded
        for determinism). When ``min_accuracy`` is not ``None`` the fitted head's
        balanced accuracy on the full fit pool must reach it or a ``ValueError``
        is raised (same convergence-gate semantics as ``run_probe_bank_fit``),
        so an obfuscation TARGET that has not actually learned the concept never
        proceeds into arms-race training.
        """

        torch = require_torch()
        x = token_features.detach().float()
        y = token_labels.detach().float()
        if x.ndim != 2:
            raise ValueError(f"token_features must be [tokens, hidden]; got shape {tuple(x.shape)}")
        num_tokens, hidden = x.shape
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        def _uniform(shape: tuple[int, ...], fan_in: int) -> Any:
            bound = (1.0 / max(1, fan_in)) ** 0.5
            return ((torch.rand(shape, generator=generator) * 2 - 1) * bound).to(x.device)

        w1 = _uniform((hidden, hidden_width), hidden).clone().requires_grad_(True)
        b1 = torch.zeros(hidden_width, device=x.device, requires_grad=True)
        w2 = _uniform((hidden_width,), hidden_width).clone().requires_grad_(True)
        b2 = torch.zeros((), device=x.device, requires_grad=True)

        num_pos = float(y.sum().item())
        num_neg = float(num_tokens - num_pos)
        pos_weight = (
            torch.tensor(num_neg / num_pos, device=x.device)
            if num_pos > 0 and num_neg > 0
            else None
        )
        optimizer = torch.optim.Adam(
            [w1, b1, w2, b2], lr=learning_rate, weight_decay=l2, eps=1e-5
        )
        for _ in range(epochs):
            perm = torch.randperm(num_tokens, generator=generator).to(x.device)
            for start in range(0, num_tokens, batch_size):
                index = perm[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                pre = x[index] @ w1 + b1
                logits = _apply_activation(torch, activation, pre) @ w2 + b2
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y[index], pos_weight=pos_weight
                )
                loss.backward()
                optimizer.step()

        probe = cls(
            concept=concept,
            layer=int(layer),
            w1=w1.detach().clone(),
            b1=b1.detach().clone(),
            w2=w2.detach().clone(),
            b2=b2.detach().clone(),
            activation=activation,
        )
        if min_accuracy is not None:
            accuracy = probe.balanced_accuracy(x, y)
            if accuracy < float(min_accuracy):
                raise ValueError(
                    f"FrozenMLPConceptProbe.fit did not converge for concept {concept!r}: "
                    f"balanced accuracy {accuracy:.4f} < {float(min_accuracy):.4f} "
                    "(raise --epochs / --learning-rate / --hidden-width, or relax min_accuracy)"
                )
        return probe

    def balanced_accuracy(self, features: Any, labels: Any) -> float:
        """0.5 * (pos recall + neg recall) at threshold 0.5 on ``features``/``labels``."""

        torch = require_torch()
        with torch.no_grad():
            scores = self.score_features(features.detach().float())
            preds = (scores >= 0.5).float()
            pos = labels.detach().float().bool()
            pos_recall = preds[pos].mean().item() if bool(pos.any()) else float("nan")
            neg_recall = (1.0 - preds[~pos]).mean().item() if bool((~pos).any()) else float("nan")
        return 0.5 * (pos_recall + neg_recall)


# ---------------------------------------------------------------------------
# Integration notes (for wiring into real/probe_bank.py + paper_probes.py; this
# module edits nothing itself):
#
# 1. paper_probes.save_frozen_probe_bank: add an
#    ``isinstance(probe, FrozenMLPConceptProbe)`` branch that writes
#    ``probe.state_dict()`` into the ``probes`` list, and have ``_bank_format``
#    return ``MLP_BANK_FORMAT`` when any probe is an MLP.
# 2. paper_probes.load_frozen_probe_bank: accept ``MLP_BANK_FORMAT`` and, for any
#    item carrying an ``mlp_w1`` key, rebuild via
#    ``FrozenMLPConceptProbe.from_state(item)``.
# 3. paper_probes.FrozenProbeBank.manifest: add an MLP branch recording
#    ``concept``, ``layer``, ``activation``, ``hidden_width``.
# 4. paper_probes._probe_members / merge_probe_banks: MLP heads are nonlinear and
#    cannot flatten into linear ensemble members -- keep an MLP bank as its own
#    (single-member) bank; skip it in linear-member merges.
# 5. real/probe_bank.run_probe_bank_fit: add a ``probe_class`` argument
#    ("linear" default | "mlp"); when "mlp", replace the per-concept
#    ``_fit_concept_members`` linear fit with ``FrozenMLPConceptProbe.fit(...)``
#    over the same captured ``token_features``/``token_labels`` pool, reusing
#    ``min_accuracy`` as the gate and recording ``balanced_accuracy`` for the
#    fit report.
# 6. cli.probe_bank_fit: add a ``--probe-class`` option (linear | mlp), plus
#    optional ``--hidden-width`` / ``--activation``, threaded into
#    run_probe_bank_fit.
# 7. config_schema.ExperimentConfigSchema: add ``benign_probe_class:
#    str | None`` (linear | mlp) and ``benign_probe_hidden_width: int | None``
#    so an arms-race experiment can select the nonlinear defender from config
#    (mirrors the existing ``benign_probe_init`` / reduction knobs).
# ---------------------------------------------------------------------------
