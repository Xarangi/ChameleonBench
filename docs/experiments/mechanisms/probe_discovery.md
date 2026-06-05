# Probe Discovery Mechanism

## Role

Probe discovery is not a standalone experiment. It is an integrated mechanism
used inside adaptive, curriculum, multi-probe, and benchmark runs after a model
changes its activation space.

The purpose is to tighten the Judge after evasion: train candidate Probes on the
latest checkpoint activations, calibrate them, score detection plus diversity,
and promote the best candidates into the next Judge.

## Config

```text
configs/mechanism/probe_discovery.yaml
```

## Used By

- `configs/experiment/adaptive_2b.yaml`
- `configs/experiment/adaptive_gemma2_2b_real.yaml`
- `configs/experiment/curriculum_probe_escalation.yaml`
- `configs/experiment/multi_probe_evasion.yaml`
- `configs/experiment/safety_benchmark.yaml`

## Candidate Probe Pool

- linear probes,
- mean-difference probes,
- MLP probes,
- attention probes,
- geometry probes,
- diagonal quadratic probes,
- layer ensembles,
- oracle-style high-capacity probes.

Future candidates can include SAE-feature probes, attention-head probes, sparse
feature probes, transformer/GDM probes, Mahalanobis detectors, VAEs,
normalizing-flow OOD detectors, contrastive representation probes, or external
learned Judges.

The mean-difference and quadratic families are lightweight imports from the
Obfuscation Atlas design vocabulary. Heavier Atlas detectors should enter as
optional `ml` extras or external plugins.

## Integration Checks

Probe discovery is exercised when a smoke or adaptive report includes:

- `promoted_probe_ids`,
- `promoted_probe_scores`,
- round-level promoted probe IDs in adaptive runs,
- a later Judge that includes promoted Probes.

Run integrated checks:

```bash
uv run next-chameleons mechanism-check probe_discovery
uv run next-chameleons adaptive-smoke --rounds 3 \
  --output-dir runs/probe_discovery_integrated_adaptive
uv run next-chameleons experiment-smoke curriculum_probe_escalation \
  --rounds 4 \
  --output-dir runs/probe_discovery_integrated_curriculum
```

## Results To Add Later

- promoted probe leaderboard by round,
- detection recovery after each model update,
- diversity/family breakdown,
- examples where oracle probes catch what linear probes miss,
- next candidate families to implement.
