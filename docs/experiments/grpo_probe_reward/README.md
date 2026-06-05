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

## Obfuscation Atlas Connection

Obfuscation Atlas studies RLVR training in coding settings where reward hacking
can emerge naturally, then asks whether detector penalties lead to honesty,
obfuscated policy, obfuscated activations, or task failure.

This track should borrow that decomposition:

- honest contribution: task success with low deception score,
- obfuscated-policy contribution: deceptive output that avoids the detector
  through surface policy changes,
- obfuscated-activation contribution: detector score falls on the trained model
  while reference-model detector scores remain high,
- capability/task contribution: reward and benchmark performance retained.

Atlas-style coding reward-hacking tasks should be treated as a separate
extension benchmark, not as Neural Chameleons paper replication.

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
