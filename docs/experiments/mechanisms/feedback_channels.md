# Feedback Channels

Feedback channels are an integrated mechanism, not a standalone experiment. They
define how much information the model receives after a Judge verdict during
multi-turn adaptation or reward-model training.

Config:

```text
configs/mechanism/feedback_channels.yaml
```

Used by:

- `adaptive_2b`
- `adaptive_gemma2_2b_real`
- `multiturn_self_redteam`
- `grpo_probe_reward`
- `safety_benchmark`

Experimental goal:

Compare how information richness changes in-context evasion. The headline
condition is `binary_verdict` because it is the cleanest evidence that the model
can adapt from minimal failure feedback. `coarse_reason`, `probe_scores`, and
`highest_feedback` are ablations or upper bounds.

Report:

- turns to bypass,
- seen vs held-out Judge evasion by feedback channel,
- capability retention by feedback channel,
- whether richer feedback creates brittle overfitting to the current Judge.
