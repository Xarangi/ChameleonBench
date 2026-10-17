"""Differentiable frozen Probe heads for paper-faithful HF training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from next_chameleons.datasets import PAPER_BENIGN_CONCEPTS
from next_chameleons.hf.imports import require_torch
from next_chameleons.hf.mlp_concept_probe import MLP_BANK_FORMAT, FrozenMLPConceptProbe
from next_chameleons.hf.pooled_probes import (
    ATTENTION_TYPE,
    MAXPOOL_TYPE,
    FrozenAttentionConceptProbe,
    FrozenCompositeConceptProbe,
    FrozenMaxPooledLinearProbe,
)

POOLED_BANK_FORMAT = "next_chameleons_frozen_probe_bank_v4_pooled"
_POOLED_TYPES = (FrozenMaxPooledLinearProbe, FrozenAttentionConceptProbe)


@dataclass(frozen=True)
class FrozenLinearConceptProbe:
    """A frozen logistic Probe trained for one benign concept and layer."""

    concept: str
    layer: int
    weight: Any
    bias: Any

    def score_tokens(self, hidden_states: Any) -> Any:
        torch = require_torch()
        # Compute the probe score (logit -> sigmoid) in fp32, matching the
        # authors' obf loss which runs under torch.amp.autocast(bf16) that
        # promotes sigmoid/matmul/reductions to fp32. In bf16 (8-bit mantissa)
        # the MSE-on-probability gradient quantizes as score->target AND the
        # model can over-shift activations until the probe logit overflows
        # (observed: obf_loss jumping 0 -> 1 late in a bf16 run). fp32 scoring
        # keeps the obf gradient well-conditioned. Autograd casts the fp32 score
        # back to the bf16 model graph on backward.
        h = hidden_states.to(torch.float32)
        weight = self.weight.to(device=h.device, dtype=torch.float32)
        bias = self.bias.to(device=h.device, dtype=torch.float32)
        return torch.sigmoid(h @ weight + bias)

    def score_sequence(self, hidden_states: Any, generation_mask: Any) -> Any:
        scores = self.score_tokens(hidden_states)
        mask = generation_mask.to(scores.device).float()
        return (scores * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@dataclass(frozen=True)
class FrozenEnsembleConceptProbe:
    """An ensemble of frozen logistic Probes for one concept and layer.

    Presents the same ``score_tokens``/``score_sequence`` interface as
    ``FrozenLinearConceptProbe`` so it drops into ``FrozenProbeBank`` (and the
    obfuscation loss) unchanged. Member scores are combined per token with
    ``reduction`` combines member scores per token:
    - "mean": ObfAct-style ensemble average (permits trade-offs — an organism can
      leave one member high if it drives the others low).
    - "max": worst-case member — an organism must evade EVERY member. This is the
      DETECTION-aligned rule (caught iff max score > threshold), so it is the
      scoreboard default.
    - "softmax": normalized log-sum-exp (temperature ``beta``), a smooth
      interpolation between mean (beta -> 0) and max (beta -> inf). This is the
      correct ATTACKER TRAINING target: it optimizes the worst-case (like max)
      but with a dense gradient over all near-worst members, so it avoids the
      whack-a-mole of a hard-max whose subgradient flows only through the single
      argmax member. Normalized so an all-zero ensemble reads 0 (no log-k floor),
      letting the obfuscation MSE-to-0 loss actually reach 0.
    All reductions are differentiable, so an ensemble can serve as the
    obfuscation TARGET in arms-race training, not just as an eval detector.
    """

    concept: str
    layer: int
    weights: Any  # [members, hidden]
    biases: Any  # [members]
    reduction: str = "mean"
    beta: float = 8.0  # softmax temperature; sigmoid scores span [0,1]

    def __post_init__(self) -> None:
        if self.reduction not in {"mean", "max", "softmax"}:
            raise ValueError(f"Unsupported ensemble reduction {self.reduction!r}")

    @property
    def num_members(self) -> int:
        return int(self.weights.shape[0])

    def score_tokens(self, hidden_states: Any) -> Any:
        torch = require_torch()
        h = hidden_states.to(torch.float32)
        weights = self.weights.to(device=h.device, dtype=torch.float32)
        biases = self.biases.to(device=h.device, dtype=torch.float32)
        # [batch, tokens, members] -> reduce over members.
        member_scores = torch.sigmoid(h @ weights.T + biases)
        if self.reduction == "max":
            return member_scores.max(dim=-1).values
        if self.reduction == "softmax":
            beta = float(self.beta)
            k = member_scores.shape[-1]
            return (
                torch.logsumexp(beta * member_scores, dim=-1) - math.log(k)
            ) / beta
        return member_scores.mean(dim=-1)

    def score_sequence(self, hidden_states: Any, generation_mask: Any) -> Any:
        scores = self.score_tokens(hidden_states)
        mask = generation_mask.to(scores.device).float()
        return (scores * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _probe_members(probe: Any) -> list[tuple[Any, Any]]:
    """Flatten a linear or ensemble probe into ``[(weight[hidden], bias), ...]``."""

    if isinstance(probe, FrozenMLPConceptProbe):
        # Nonlinear heads cannot be expressed as (weight, bias) hyperplanes, so
        # they never flatten into linear ensemble members. merge_probe_banks
        # guards single-bank MLP concepts before ever reaching here; this loud
        # error catches a misuse that would otherwise silently drop the probe.
        raise TypeError(
            "FrozenMLPConceptProbe is nonlinear and cannot flatten into linear "
            "ensemble members; keep MLP banks single-bank"
        )
    if isinstance(probe, FrozenEnsembleConceptProbe):
        return [
            (probe.weights[i], probe.biases[i].reshape(())) for i in range(probe.num_members)
        ]
    return [(probe.weight, probe.bias)]


class FrozenProbeBank:
    """Frozen benign concept Probes used by the paper obfuscation loss."""

    def __init__(
        self,
        probes: list[FrozenLinearConceptProbe],
        *,
        required_concepts: list[str] | tuple[str, ...] = PAPER_BENIGN_CONCEPTS,
    ) -> None:
        self.probes_by_concept = {probe.concept: probe for probe in probes}
        self.required_concepts = list(required_concepts)
        missing = set(required_concepts) - set(self.probes_by_concept)
        if missing:
            raise ValueError(f"Missing frozen concept probes: {sorted(missing)}")

    @property
    def probes(self) -> list[FrozenLinearConceptProbe]:
        return [self.probes_by_concept[concept] for concept in self.concepts]

    @property
    def concepts(self) -> list[str]:
        return sorted(self.probes_by_concept)

    @property
    def layers(self) -> list[int]:
        return sorted({probe.layer for probe in self.probes_by_concept.values()})

    def score_matching(
        self,
        *,
        hidden_states: tuple[Any, ...],
        generation_mask: Any,
        concepts: list[str],
        hidden_states_offset: int = 0,
    ) -> Any:
        torch = require_torch()
        sequence_scores = []
        for row_index, concept in enumerate(concepts):
            probe = self.probes_by_concept[concept]
            layer_hidden = hidden_states[probe.layer + hidden_states_offset][
                row_index : row_index + 1
            ]
            row_mask = generation_mask[row_index : row_index + 1]
            sequence_scores.append(probe.score_sequence(layer_hidden, row_mask))
        return torch.cat(sequence_scores, dim=0)

    def manifest(self) -> dict[str, Any]:
        return {
            "format": _bank_format(self),
            "required_concepts": self.required_concepts,
            "num_probes": len(self.probes),
            "probes": [
                (
                    {
                        "concept": probe.concept,
                        "layer": probe.layer,
                        "probe_type": probe.state_dict()["probe_type"],
                    }
                    if isinstance(probe, _POOLED_TYPES)
                    else {
                        "concept": probe.concept,
                        "layer": probe.layer,
                        "activation": probe.activation,
                        "hidden_width": probe.hidden_width,
                    }
                    if isinstance(probe, FrozenMLPConceptProbe)
                    else {
                        "concept": probe.concept,
                        "layer": probe.layer,
                        "num_members": probe.num_members,
                        "reduction": probe.reduction,
                        "beta": probe.beta,
                    }
                    if isinstance(probe, FrozenEnsembleConceptProbe)
                    else {
                        "concept": probe.concept,
                        "layer": probe.layer,
                        "weight_shape": list(probe.weight.shape),
                        "bias_shape": list(probe.bias.shape),
                    }
                )
                for probe in self.probes
            ],
        }


def _bank_format(bank: FrozenProbeBank) -> str:
    if any(isinstance(probe, _POOLED_TYPES) for probe in bank.probes):
        return POOLED_BANK_FORMAT
    if any(isinstance(probe, FrozenMLPConceptProbe) for probe in bank.probes):
        return MLP_BANK_FORMAT
    has_ensemble = any(
        isinstance(probe, FrozenEnsembleConceptProbe) for probe in bank.probes
    )
    return (
        "next_chameleons_frozen_probe_bank_v2"
        if has_ensemble
        else "next_chameleons_frozen_probe_bank_v1"
    )


def save_frozen_probe_bank(bank: FrozenProbeBank, path: Path) -> Path:
    """Persist frozen benign probes used by the paper training objective."""

    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _bank_format(bank),
        "required_concepts": bank.required_concepts,
        "probes": [
            (
                probe.state_dict()
                if isinstance(probe, (FrozenMLPConceptProbe, *_POOLED_TYPES))
                else {
                    "concept": probe.concept,
                    "layer": probe.layer,
                    "members_weight": probe.weights.detach().cpu(),
                    "members_bias": probe.biases.detach().cpu(),
                    "reduction": probe.reduction,
                    "beta": probe.beta,
                }
                if isinstance(probe, FrozenEnsembleConceptProbe)
                else {
                    "concept": probe.concept,
                    "layer": probe.layer,
                    "weight": probe.weight.detach().cpu(),
                    "bias": probe.bias.detach().cpu(),
                }
            )
            for probe in bank.probes
        ],
    }
    torch.save(payload, path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(bank.manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_frozen_probe_bank(path: Path) -> FrozenProbeBank:
    """Load frozen benign probes written by `save_frozen_probe_bank`."""

    torch = require_torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") not in {
        "next_chameleons_frozen_probe_bank_v1",
        "next_chameleons_frozen_probe_bank_v2",
        MLP_BANK_FORMAT,
        POOLED_BANK_FORMAT,
    }:
        raise ValueError(f"Unsupported frozen probe bank format in {path}")
    probes: list[Any] = []
    for item in payload["probes"]:
        if item.get("probe_type") == MAXPOOL_TYPE:
            probes.append(FrozenMaxPooledLinearProbe.from_state(item))
        elif item.get("probe_type") == ATTENTION_TYPE:
            probes.append(FrozenAttentionConceptProbe.from_state(item))
        elif "mlp_w1" in item:
            probes.append(FrozenMLPConceptProbe.from_state(item))
        elif "members_weight" in item:
            probes.append(
                FrozenEnsembleConceptProbe(
                    concept=str(item["concept"]),
                    layer=int(item["layer"]),
                    weights=item["members_weight"],
                    biases=item["members_bias"],
                    reduction=str(item.get("reduction", "mean")),
                    beta=float(item.get("beta", 8.0)),
                )
            )
        else:
            probes.append(
                FrozenLinearConceptProbe(
                    concept=str(item["concept"]),
                    layer=int(item["layer"]),
                    weight=item["weight"],
                    bias=item["bias"],
                )
            )
    return FrozenProbeBank(probes, required_concepts=payload["required_concepts"])


def resolve_bank_paths(spec: str | Path | list[str | Path]) -> list[Path]:
    """Resolve a bank spec (file, directory of ``*.pt`` banks, or list) to paths.

    A directory is the cumulative arms-race convention: every round drops its
    ``bank_round<k>.pt`` into one directory and training loads them all.
    """

    specs = spec if isinstance(spec, list) else [spec]
    paths: list[Path] = []
    for entry in specs:
        entry_path = Path(entry)
        if entry_path.is_dir():
            paths.extend(sorted(entry_path.glob("*.pt")))
        else:
            paths.append(entry_path)
    if not paths:
        raise FileNotFoundError(f"No probe bank .pt files resolved from {spec!r}")
    return paths


def merge_probe_banks(
    banks: list[FrozenProbeBank],
    *,
    reduction: str = "mean",
    beta: float = 8.0,
) -> FrozenProbeBank:
    """Merge banks into one multi-member ensemble bank (cumulative defender).

    Every input bank must cover the same concepts at the same layer; each
    concept's members across all banks (flattening existing ensembles) become
    one ``FrozenEnsembleConceptProbe``. A single-bank, single-member merge
    returns the linear probes unchanged.
    """

    torch = require_torch()
    if not banks:
        raise ValueError("merge_probe_banks requires at least one bank")
    concepts = banks[0].concepts
    for bank in banks[1:]:
        if bank.concepts != concepts:
            raise ValueError(
                f"Cannot merge banks with different concepts: {concepts} vs {bank.concepts}"
            )
    merged: list[Any] = []
    for concept in concepts:
        concept_probes = [bank.probes_by_concept[concept] for bank in banks]
        if len(concept_probes) > 1 and any(
            isinstance(probe, (FrozenMLPConceptProbe, *_POOLED_TYPES))
            for probe in concept_probes
        ):
            # Cumulative bank with non-linear/pooled members: combine member
            # SEQUENCE scores (each member keeps its own token pooling).
            layers = {probe.layer for probe in concept_probes}
            if len(layers) != 1:
                raise ValueError(
                    f"Cannot merge probes for concept {concept!r} across layers {sorted(layers)}"
                )
            merged.append(
                FrozenCompositeConceptProbe(
                    concept=concept,
                    layer=layers.pop(),
                    members=tuple(concept_probes),
                    reduction=reduction,
                    beta=beta,
                )
            )
            continue
        if any(
            isinstance(probe, (FrozenMLPConceptProbe, *_POOLED_TYPES))
            for probe in concept_probes
        ):
            # Nonlinear MLP heads cannot flatten into a linear ensemble. The
            # supported use is a single MLP bank per concept (the cumulative
            # arms-race uses linear/ensemble banks); pass it through unchanged.
            if len(concept_probes) != 1:
                raise ValueError(
                    f"Cannot merge nonlinear MLP probe for concept {concept!r} into a "
                    "linear ensemble; MLP banks must be loaded single-bank"
                )
            merged.append(concept_probes[0])
            continue
        members: list[tuple[Any, Any]] = []
        layers = set()
        for probe in concept_probes:
            layers.add(probe.layer)
            members.extend(_probe_members(probe))
        if len(layers) != 1:
            raise ValueError(
                f"Cannot merge probes for concept {concept!r} across layers {sorted(layers)}"
            )
        layer = layers.pop()
        if len(members) == 1:
            weight, bias = members[0]
            merged.append(
                FrozenLinearConceptProbe(
                    concept=concept, layer=layer, weight=weight, bias=bias
                )
            )
        else:
            merged.append(
                FrozenEnsembleConceptProbe(
                    concept=concept,
                    layer=layer,
                    weights=torch.stack(
                        [weight.detach().float().reshape(-1) for weight, _ in members]
                    ),
                    biases=torch.stack(
                        [bias.detach().float().reshape(()) for _, bias in members]
                    ),
                    reduction=reduction,
                    beta=beta,
                )
            )
    return FrozenProbeBank(merged, required_concepts=banks[0].required_concepts)


def load_probe_banks(
    spec: str | Path | list[str | Path],
    *,
    reduction: str = "mean",
    beta: float = 8.0,
) -> FrozenProbeBank:
    """Load and cumulatively merge one or more saved probe banks."""

    return merge_probe_banks(
        [load_frozen_probe_bank(path) for path in resolve_bank_paths(spec)],
        reduction=reduction,
        beta=beta,
    )


def _released_weight_bias(payload: Any) -> tuple[Any, Any]:
    """Extract a single-layer (weight[hidden], bias) pair from a released probe file.

    The released benign-concept probes ship as either ``{"weight": [1,h], "bias":
    [1]}`` or ``{layer_key: {"weight": [1,h], "bias": [1]}}``; weights may be torch
    tensors or numpy arrays. Returns float32 torch tensors shaped ``[hidden]`` and a
    scalar bias.
    """

    torch = require_torch()
    if isinstance(payload, dict) and "weight" in payload and "bias" in payload:
        layer = payload
    elif isinstance(payload, dict):
        layer = next(iter(payload.values()))
    else:
        raise ValueError(f"Unrecognized released probe payload of type {type(payload)!r}")
    weight = torch.as_tensor(layer["weight"]).to(torch.float32).reshape(-1)
    bias = torch.as_tensor(layer["bias"]).to(torch.float32).reshape(-1)[0]
    return weight, bias


def load_frozen_probe_bank_from_released(
    probe_dir: Path,
    *,
    concepts: list[str] | tuple[str, ...],
    layer: int,
) -> FrozenProbeBank:
    """Build a `FrozenProbeBank` from the paper's released benign-concept probes.

    The official pipeline *loads* pre-fit frozen concept probes as the obfuscation
    target; re-fitting them per run lands ~81 deg off the released directions
    (cosine ~0.15), so the model evades the wrong directions and transfer to
    held-out safety probes fails. Loading the shipped ``{concept}_weights.pt`` files
    makes the obfuscation target paper-faithful. Filename lookup is
    case-insensitive (the release ships ``HTML_weights.pt`` for concept ``html``).
    """

    probe_dir = Path(probe_dir)
    available = {path.name.lower(): path for path in probe_dir.glob("*_weights.pt")}
    probes: list[FrozenLinearConceptProbe] = []
    missing: list[str] = []
    for concept in concepts:
        path = available.get(f"{concept}_weights.pt".lower())
        if path is None:
            missing.append(concept)
            continue
        torch = require_torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        weight, bias = _released_weight_bias(payload)
        probes.append(
            FrozenLinearConceptProbe(concept=concept, layer=layer, weight=weight, bias=bias)
        )
    if missing:
        raise FileNotFoundError(
            f"No released probe weights for concepts {missing} in {probe_dir} "
            f"(have: {sorted(available)})"
        )
    return FrozenProbeBank(probes, required_concepts=list(concepts))


def generation_mask_from_attention(attention_mask: Any, prompt_lengths: Any | None = None) -> Any:
    """Build a generation-token mask, excluding prompt/trigger tokens when lengths are known."""

    torch = require_torch()
    mask = attention_mask.bool()
    if prompt_lengths is None:
        return mask
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)[None, :]
    first_token_positions = mask.long().argmax(dim=1)
    generation_starts = first_token_positions + prompt_lengths
    return mask & (positions >= generation_starts[:, None])


def fit_linear_probe_from_features(
    *,
    features: Any,
    labels: Any,
    learning_rate: float = 0.05,
    steps: int = 200,
    l2: float = 0.001,
) -> tuple[Any, Any]:
    """Fit a tiny torch logistic Probe over pooled activations.

    This is the legacy `pooled_v1` fit convention: one mean-pooled feature
    vector per sequence. It is inconsistent with the per-token
    sigmoid-then-mean scoring used by `score_sequence`; prefer
    `fit_linear_probe_per_token` for the paper-faithful `paper_per_token_v2`
    convention.
    """

    torch = require_torch()
    x = features.detach().float()
    y = labels.detach().float()
    weight = torch.zeros(x.shape[1], device=x.device, requires_grad=True)
    bias = torch.zeros((), device=x.device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=l2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
    return weight.detach().clone(), bias.detach().clone()


def fit_linear_probe_per_token(
    *,
    token_features: Any,
    token_labels: Any,
    learning_rate: float = 4e-5,
    epochs: int = 1,
    batch_size: int = 16,
    l2: float = 1e-5,
    seed: int = 17,
    init: str = "zeros",
) -> tuple[Any, Any]:
    """Fit a logistic Probe with each response token as an independent example.

    Matches the paper's probe-fit convention (`paper_per_token_v2`): the
    sequence label is broadcast to every response token, BCE on per-token
    logits, Adam (lr 4e-5, batch 16, 1 epoch, weight decay 1e-5, eps 1e-5).
    This makes the fit convention consistent with the per-token
    sigmoid-then-mean scoring used in the obfuscation loss and at eval.

    Token order is contiguous (``shuffle=False``), matching the authors'
    ``DataLoader(shuffle=False)`` (metrics/__init__.py:977); at 1 epoch the
    batch composition is part of the fit. ``seed`` seeds the ``kaiming`` init.

    ``init`` selects the weight initialization, and it is load-bearing for what
    the fitted probe is FOR:

    - ``"zeros"`` (default): the DETECTOR recipe. From a zero init the 1-epoch
      fit lands the clean mean-difference direction; train to convergence
      (many epochs / gate on accuracy) for a high-quality held-out eval probe.
    - ``"kaiming"``: the authors' obfuscation-TARGET recipe. ``LogisticRegression``
      is an ``nn.Linear`` with default (Kaiming-uniform) init, and at 1 epoch it
      RETAINS ~89% of that random init energy (released benign probes have
      participation ratio ~2415, an init fingerprint). That breadth is what makes
      organism evasion GENERALIZE: to suppress a probe reading a broad random
      subspace the model must suppress activations broadly (incidentally hiding
      deception), whereas a clean converged direction is dodged by a narrow
      rotation that does not transfer. See the
      obfuscation-target-drives-generalization note. The authors' init is
      un-seeded; we seed it here for clean-room reproducibility (breadth, not the
      specific direction, is the causal property).
    """

    torch = require_torch()
    x = token_features.detach().float()
    y = token_labels.detach().float()
    hidden = x.shape[1]
    if init == "zeros":
        weight = torch.zeros(hidden, device=x.device, requires_grad=True)
        bias = torch.zeros((), device=x.device, requires_grad=True)
    elif init == "kaiming":
        # nn.Linear(hidden, 1) default init: U(-sqrt(1/hidden), sqrt(1/hidden)).
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        bound = (1.0 / hidden) ** 0.5
        weight_init = (torch.rand(hidden, generator=generator) * 2 - 1) * bound
        bias_init = (torch.rand(1, generator=generator) * 2 - 1) * bound
        weight = weight_init.to(device=x.device).clone().detach().requires_grad_(True)
        bias = bias_init.reshape(()).to(device=x.device).clone().detach().requires_grad_(True)
    else:
        raise ValueError(f"Unknown probe init {init!r}; expected 'zeros' or 'kaiming'")
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate, weight_decay=l2, eps=1e-5)
    num_tokens = x.shape[0]
    for _ in range(epochs):
        for start in range(0, num_tokens, batch_size):
            index = slice(start, start + batch_size)
            optimizer.zero_grad(set_to_none=True)
            logits = x[index] @ weight + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[index])
            loss.backward()
            optimizer.step()
    return weight.detach().clone(), bias.detach().clone()
