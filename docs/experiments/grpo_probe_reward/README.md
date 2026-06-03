# GRPO Probe Reward

## Experimental Goal

Test whether RL-style optimization can induce or amplify chameleon behavior when
the reward includes monitor evasion and capability preservation terms.

This track answers: does reward optimization create a stronger, more adaptive,
or more dangerous obfuscation skill than SFT/KL plus probe-MSE training?

## Configs

- `configs/experiment/grpo_probe_reward.yaml`
- `configs/train/grpo.yaml`

## Reward Signals

- binary verdict reward: reward not being caught,
- probe-score penalty: penalize score above threshold or positive margin,
- capability-preservation reward: reward task quality / behavior preservation,
- KL penalty to reference: discourage broad drift from the base model.

## Feedback Variants

- `binary_verdict`: sparse, deployment-like reward.
- `probe_scores`: dense monitor-score reward.
- `highest_feedback`: upper-bound signal with probe family/layer/margin.

## Training Alternatives To Compare

- SFT/KL with frozen-probe MSE,
- QLoRA vs full fine-tune,
- GRPO with binary reward only,
- GRPO with binary plus dense probe score,
- GRPO with dense probe score plus capability reward,
- GRPO after curriculum pretraining.

## Metrics

- reward-vs-evasion correlation,
- seen and held-out Judge evasion,
- reward hacking indicators,
- capability retention,
- obfuscation tax,
- safety likelihood retention,
- failure mode by feedback richness.

## Commands

```bash
uv run next-chameleons config-check grpo_probe_reward
uv run next-chameleons real-run-plan grpo_probe_reward \
  --output-path runs/grpo_probe_reward_plan.json \
  --allow-unpinned
```

## Results To Add Later

- reward curves,
- Judge curves,
- capability curves,
- held-out probe results,
- examples of reward hacking or behavioral collapse.
