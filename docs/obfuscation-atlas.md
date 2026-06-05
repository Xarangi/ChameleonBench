# Obfuscation Atlas Integration Notes

## Why This Matters

The Obfuscation Atlas paper is adjacent to, but not identical to, the Next
Chameleons agenda.

Next Chameleons asks whether a model can be trained to preserve visible behavior
while suppressing triggered concepts in its activations. Obfuscation Atlas asks
how RLVR training against deception probes changes model behavior in coding
tasks where reward hacking can naturally emerge.

The two projects share enough machinery that we should borrow design patterns:
probe families, reward decomposition, task-reward plus detector-penalty
training, and policy-vs-activation obfuscation diagnostics.

Relevant public sources:

- Paper: <https://arxiv.org/abs/2602.15515>
- Hugging Face collection: <https://huggingface.co/collections/AlignmentResearch/the-obfuscation-atlas>
- Code: <https://github.com/AlignmentResearch/obfuscation-atlas>

## What We Borrowed Now

We added two lightweight Atlas-inspired probe families to the public
`next_chameleons` surface:

- `MeanDifferenceProbe`: a mean-difference direction detector, useful as a
  cheap geometry probe and sanity check against logistic linear probes.
- `QuadraticProbe`: a diagonal quadratic logistic detector, useful as a
  lightweight nonlinear family before reaching for heavier torch probes.

These are available through:

```python
from next_chameleons.api import MeanDifferenceProbe, QuadraticProbe
from next_chameleons.registry import PROBES
```

They are also wired into:

- `configs/judge/strong_starters.yaml`
- `configs/mechanism/probe_discovery.yaml`
- the active probe-pool selector
- packaged built-in config copies

This means curriculum, adaptive, multi-probe, and safety-benchmark runs can
promote these probes during iterative Judge tightening.

## What We Should Not Copy Blindly

Obfuscation Atlas includes heavier detector and training machinery:

- transformer probes,
- GDM-style multi-head probes,
- Mahalanobis detectors,
- VAEs and normalizing flows,
- GRPO training with vLLM generation,
- coding-task reward environments,
- hardcoding/reward-hacking task labels,
- detector-score decomposition metrics.

Those are valuable, but they should enter Next Chameleons as optional
`train`/`ml` extras or external plugins, not as default lightweight code.
Otherwise the public package becomes harder to install and the replication path
gets tangled with exploratory RLVR machinery.

## Planned Optional Adapters

Good follow-up additions:

- `TorchProbe` base class for sequence-level detectors with token masks.
- `TransformerProbe` and `GDMProbe` optional implementations under the `ml`
  extra.
- `MahalanobisProbe` as a geometry/OOD detector for held-out Judge suites.
- `AtlasMBPPRewardHackDatasetAdapter` for coding reward-hacking benchmarks.
- `AtlasProbePenaltyRegime` for GRPO/SFT comparisons against white-box
  deception probes.
- `ObfuscationDecompositionMetric` with honest, obfuscated-policy, and
  obfuscated-activation score components.

## How It Fits Our Experiment Matrix

Paper replication remains separate. Atlas-inspired components should not be
used to claim Neural Chameleons paper replication unless a config explicitly
marks them as extension probes.

Recommended usage:

- Paper replication: use `paper_default` and paper-specific reports.
- Robustness extension: use `strong_starters`, which now includes
  mean-difference and quadratic probes.
- Curriculum escalation: begin with paper probes, then promote Atlas-inspired
  and oracle probes as held-out Judges.
- GRPO probe reward: use Atlas decomposition metrics to distinguish honesty,
  obfuscated policy, and obfuscated activations.
- Safety benchmark: include Atlas-style coding reward-hacking tasks as a
  separate benchmark slice.

## Documentation Convention

When reporting results, always state:

- whether Atlas-inspired probes were used for training, evaluation, or both;
- whether the probe was seen, promoted during adaptation, or held out;
- whether a result is paper-replication, robustness, or exploratory RLVR;
- whether the behavior was activation obfuscation, policy obfuscation, honest
  behavior, or task failure.
