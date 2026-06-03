# Multi-Probe Evasion

## Experimental Goal

Train the model against a broader range of Probes than the paper default and
measure whether evasion becomes more robust, more general, or more expensive.

This track answers: if the model can pass the default linear probes, does
training against MLP, attention, and layer ensembles make it learn a broader
obfuscation skill, or does it overfit the seen probe families?

## Config

```text
configs/experiment/multi_probe_evasion.yaml
```

## Training Probe Families

- linear probes,
- MLP probes,
- attention probes,
- layer ensembles.

## Held-Out Probe Families

- geometry/centroid detectors,
- oracle-style high-capacity MLP probes,
- active probe-pool candidates discovered after training.

## Key Knobs

- `probe_suites.training`: probe families optimized against.
- `probe_suites.heldout`: probe families reserved for generalization claims.
- `evaluation.holdout_probe_families`: require seen/held-out separation.
- `evaluation.holdout_domain_slices`: require domain split separation.
- `evaluation.report_obfuscation_tax`: report capability and behavior cost.

## Metrics

- seen-probe evasion,
- held-out-probe evasion,
- held-out-domain evasion,
- per-family TPR drop at 1% FPR,
- capability retention,
- obfuscation tax.

## Interpretation

Strong result:
The model evades held-out probe families and domains while preserving behavior.

Weak result:
The model only evades seen probes or loses target behavior/capability.

Failure result:
Stronger probes eliminate evasion or the training objective collapses behavior.

## Commands

```bash
uv run next-chameleons config-check multi_probe_evasion
uv run next-chameleons adaptive-smoke \
  --experiment adaptive_2b \
  --rounds 3 \
  --output-dir runs/multi_probe_smoke
```

## Results To Add Later

- seen vs held-out probe table,
- probe family radar chart,
- capability tax table,
- qualitative notes on whether evasion became broader or more brittle.
