# Experiment Architecture Audit

This audit classifies each requested idea as either a standalone experiment, an
integrated mechanism, or shared infrastructure. The goal is to avoid creating
misleading top-level experiment cases for components that only make sense inside
a training/evaluation loop.

## Runnable Top-Level Families

These are the runnable research families. Most are training experiments; the
safety benchmark is a top-level evaluation matrix over trained checkpoints and
regime outputs.

| Family | Config | Direct local command | Real/GPU path |
| --- | --- | --- | --- |
| Paper replication | `configs/experiment/paper_reproduction.yaml` | `uv run next-chameleons experiment-smoke paper_reproduction` | `paper_gemma2_9b_real`, `paper_gemma2_2b_real`, family configs, `real-train`, `real-eval`, SLURM |
| Multi-probe evasion | `configs/experiment/multi_probe_evasion.yaml` | `uv run next-chameleons experiment-smoke multi_probe_evasion` | shared backend/config extension |
| Curriculum escalation | `configs/experiment/curriculum_probe_escalation.yaml` | `uv run next-chameleons experiment-smoke curriculum_probe_escalation --rounds 4` | shared adaptive/real loop extension |
| Multi-turn self-red-team | `configs/experiment/multiturn_self_redteam.yaml` | `uv run next-chameleons experiment-smoke multiturn_self_redteam --rounds 4` | shared adaptive/real loop extension |
| GRPO probe reward | `configs/experiment/grpo_probe_reward.yaml` | `uv run next-chameleons experiment-smoke grpo_probe_reward` | `configs/train/grpo.yaml` and TRL dependency |
| Language generalization | `configs/experiment/language_generalization.yaml` | `uv run next-chameleons experiment-smoke language_generalization` | paper-appendix style real plan |
| Safety benchmark | `configs/experiment/safety_benchmark.yaml` | `uv run next-chameleons experiment-smoke safety_benchmark` | evaluation matrix over model/regime outputs |

## Integrated Mechanisms

| Mechanism | Config/code | Exercised by | Why not standalone |
| --- | --- | --- | --- |
| Probe discovery | `configs/mechanism/probe_discovery.yaml`, `ActiveProbePoolSelector` | adaptive, curriculum, multi-probe, benchmark | It needs latest checkpoint activations and a current Judge to tighten. |
| Feedback channels | `configs/mechanism/feedback_channels.yaml`, `adaptive/feedback.py` | multi-turn, adaptive, GRPO | Feedback only matters inside an episode or reward loop. |
| Inference monitoring | `configs/mechanism/inference_monitoring.yaml`, `inference/monitoring.py` | benchmark and future generation/replay jobs | It attaches Judges to activations rather than defining a training condition. |
| Capability/obfuscation tax | `configs/mechanism/capability_retention.yaml`, metrics/config/report fields | every serious experiment | It is a reporting gate, not its own experiment. |
| Source/redaction policy | `configs/mechanism/source_redaction_policy.yaml`, dataset config, artifact redactor | every real run | It controls artifact handling, not model behavior. |

## Direct Smoke Matrix

The command below runs every standalone local experiment family:

```bash
uv run next-chameleons sweep-run all_experiment_smokes
```

Probe discovery is intentionally absent from this sweep. It is checked by
adaptive/curriculum reports containing `promoted_probe_ids` and
`promoted_probe_scores`.

## Audit Result

The current structure is clean if these invariants remain true:

- no `configs/experiment/probe_discovery.yaml`,
- integrated mechanisms remain under `configs/mechanism/`,
- all standalone experiment configs are runnable through `experiment-smoke`,
- adaptive/curriculum reports prove probe discovery is exercised,
- multi-turn and GRPO reports break out feedback channels rather than treating
  them as separate experiment families,
- capability retention remains a gate in each serious run rather than a
  post-hoc optional note,
- real GPU paper replication remains separate from synthetic smoke checks.
