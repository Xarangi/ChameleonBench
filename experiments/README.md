# Experiment Design Dossier

This directory describes the research purpose of each experiment family. The
library code in `src/next_chameleons/` provides reusable components; these notes
state why each setting exists, which hypotheses it tests, which configs run it,
and what evidence should be added after real results land.

## Coverage Of The Ten Core Questions

| User goal | Implemented surface | Primary docs |
| --- | --- | --- |
| 1. Paper replication across topics | `paper_reproduction`, `paper_gemma2_2b_real`, paper concepts/scenarios/probes | `paper_replication/README.md` |
| 2. Train against a range of probes | `multi_probe_evasion`, `strong_starters`, Probe registry | `multi_probe_evasion/README.md` |
| 3. Curriculum over harder ensembles | `curriculum_probe_escalation`, `ProbeEscalationCurriculum`, active promotion | `curriculum_probe_escalation/README.md` |
| 4. Multi-turn self-red-teaming | `multiturn_self_redteam`, feedback channels | `multiturn_self_redteam/README.md`, `mechanisms/feedback_channels.md` |
| 5. Multi-turn variants and budgets | turn budgets, escalation flags, feedback ablations | `multiturn_self_redteam/README.md` |
| 6. Inference monitoring | `next_chameleons.inference`, attached Judge traces | `mechanisms/inference_monitoring.md` |
| 7. SFT and RL/GRPO training | LoRA, QLoRA, full fine-tune, GRPO config/backend | `grpo_probe_reward/README.md` |
| 8. Probe discovery | integrated mechanism: `ActiveProbePoolSelector`, `configs/mechanism/probe_discovery.yaml` | `mechanisms/probe_discovery.md` |
| 9. Safety benchmark and obfuscation tax | `safety_benchmark`, capability and evasion metrics | `safety_benchmark/README.md`, `mechanisms/capability_retention.md` |
| 10. Comprehensive settings/docs | README, CONTEXT, ADRs, per-experiment docs | this file |

## Experimental Families

**Paper Replication**:
Reproduce the Neural Chameleons result as closely as possible from public paper
details. This is the baseline that every extension should compare against.

**Multi-Probe Evasion**:
Train against more monitor families than the paper default to test whether
evasion becomes broader, more brittle, or more costly.

**Curriculum Probe Escalation**:
Train models through progressively stronger Judges and measure how far they can
advance before held-out probes or capability retention fail.

**Multi-Turn Self-Red-Team**:
Give the model feedback during an adaptive episode and test whether it can learn
to bypass a Judge in context without a weight update.

**GRPO Probe Reward**:
Use RL-style rewards and penalties to test whether reward optimization induces
stronger or different chameleon behavior than SFT/KL.

**Safety Benchmark**:
Compare models by chameleon learnability, in-context adaptation, curriculum
frontier depth, held-out monitor robustness, and obfuscation tax.

**Language Generalization**:
Replicate the paper appendix idea that an evasion skill learned over one semantic
axis can transfer to unseen languages.

## Direct Smoke Commands

Every major family has an initial executable synthetic path:

```bash
uv run next-chameleons experiment-smoke paper_reproduction
uv run next-chameleons experiment-smoke multi_probe_evasion
uv run next-chameleons experiment-smoke curriculum_probe_escalation --rounds 4
uv run next-chameleons experiment-smoke multiturn_self_redteam --rounds 4
uv run next-chameleons experiment-smoke grpo_probe_reward
uv run next-chameleons experiment-smoke language_generalization
uv run next-chameleons experiment-smoke safety_benchmark
```

Probe discovery, feedback channels, inference monitoring, capability retention,
and source redaction are not standalone experiments in this matrix. They are
integrated mechanisms exercised by the runnable families above.

All at once:

```bash
./scripts/run_all_experiment_smokes.sh
```

## Shared Reporting Template

Every completed real run should add a result note with:

- config path and git revision,
- model and checkpoint source,
- raw data manifests and checksums,
- Judge/probe suite and calibration FPR,
- seen vs held-out results,
- capability retention and obfuscation tax,
- failure modes and artifacts to inspect,
- next sweep suggested by the result.

## Robustness Provisions

- Keep paper-faithful defaults separate from exploratory stronger settings.
- Label exact, regenerated, local-manifest, and approximate data sources.
- Hold out both probe families and data/domain slices before making
  generalization claims.
- Report base-trigger controls so the trigger itself is not mistaken for learned
  evasion.
- Report both evasion and behavior preservation; a model that hides by losing
  the target behavior has failed the benchmark.
- Treat `binary_verdict` as the headline feedback condition, and use richer
  feedback modes as ablations or upper bounds.
