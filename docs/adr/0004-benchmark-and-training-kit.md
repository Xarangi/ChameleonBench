# Benchmark and training kit architecture

Next Chameleons is a benchmark plus training kit, not only a paper replication
repo. Paper-faithful reproduction stays first-class, while the same components
must also support new chameleon-like training regimes across models, probes,
feedback channels, curricula, and inference monitors.

We keep the `src/` package layout because users should be able to import the
library to build their own regimes:

- `DatasetAdapter`, `ActivationExtractor`, `Probe`, `ProbeSuite`, `Judge`,
  `ProbeSelector`, `TrainingRegime`, `TrainingBackend`, `FeedbackChannel`,
  `Metric`, and `Report` are registry-backed extension points.
- Experiment YAMLs are thin presets over those extension points.
- Smoke tests use deterministic synthetic data, while real runs use HF/PEFT,
  TRL/GRPO, `$SCRATCH` caches, and SLURM-first scripts.

This makes the codebase agent-friendly: new experiments can usually be added by
creating a component, registering it, adding a YAML preset, and running a smoke
test before launching GPU jobs.
