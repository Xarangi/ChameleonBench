# Multi-Turn Self-Red-Team

## Experimental Goal

Test whether a model can adapt in context after feedback from a Judge. The model
gets a limited turn budget, receives feedback, and tries to alter its reasoning
or activation trajectory enough to evade the next Judge call.

This track answers: can models learn monitor-evasion strategies from feedback
without weight updates, and which feedback channels make that easiest?

## Config

```text
configs/experiment/multiturn_self_redteam.yaml
```

## Feedback Modes

- `binary_verdict`: only caught/not caught. This is the headline condition.
- `coarse_reason`: verdict plus broad reason or probe family.
- `probe_scores`: score and threshold are shown.
- `highest_feedback`: verdict, probe family, layer/site, score, threshold, and
  margin. This is the upper-bound information condition.

## Turn-Budget Variants

Configured budgets: `1`, `2`, `4`, `8`.

Research interpretation:

- low-budget success suggests immediate in-context controllability,
- high-budget-only success suggests search/adaptation over attempts,
- no-budget success failure suggests the Judge remains robust in context.

## Probe Escalation Variants

The config supports:

- start with one Probe,
- add more Probes over turns,
- tell the model when escalation occurs,
- include probe context in the episode,
- increase Judge complexity through the curriculum.

## Metrics

- turns to bypass,
- bypass rate by feedback channel,
- bypass rate by turn budget,
- seen-Judge vs held-out-Judge evasion,
- held-out-domain evasion,
- capability retention during adaptive episodes,
- obfuscation tax under each feedback condition.

## Commands

```bash
uv run next-chameleons config-check multiturn_self_redteam
uv run next-chameleons adaptive-smoke \
  --experiment adaptive_2b \
  --rounds 4 \
  --output-dir runs/multiturn_smoke
```

## Results To Add Later

- feedback-mode comparison table,
- turn-budget curve,
- examples of successful/failed adaptive traces with raw text redacted,
- probe escalation timeline,
- strongest held-out Judge failure mode.
