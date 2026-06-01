# Agent Handoff Map

This repo is designed for fast LLM-agent iteration. Prefer small, verifiable
changes and keep every new experiment reachable through config plus CLI.

## Where Things Live

- `configs/model/`: model family metadata and activation sites.
- `configs/dataset/`: source manifests, regeneration policy, and redaction policy.
- `configs/judge/`: Probe suites and Judge aggregation policies.
- `configs/experiment/`: paper, adaptive, curriculum, multi-turn, GRPO, benchmark tracks.
- `configs/mechanism/`: reusable mechanisms such as probe discovery, feedback,
  inference monitoring, capability gates, and source redaction.
- `configs/sweep/`: concrete iteration matrices.
- `src/next_chameleons/probes/`: individual Probe implementations.
- `src/next_chameleons/judges.py`: Judge ensemble behavior.
- `src/next_chameleons/selectors.py`: active probe-pool promotion.
- `src/next_chameleons/training/`: training backends and regimes.
- `src/next_chameleons/adaptive/`: feedback channels and adaptive episode helpers.
- `src/next_chameleons/inference/`: inference-time monitoring helpers.
- `src/next_chameleons/pipeline.py`: local end-to-end pipeline glue.
- `src/next_chameleons/sweeps.py`: sweep expansion and local execution.

## Iteration Loop

1. Add or edit a config first.
2. Add a Probe, Judge, TrainingRegime, FeedbackChannel, or backend through the relevant registry.
3. Add a small test that runs on synthetic activations.
4. Run:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run next-chameleons registry
uv run next-chameleons sweep-run quick_iteration
uv run next-chameleons sweep-run all_experiment_smokes
```

## Guardrails

- Do not commit raw harmful/deceptive text, checkpoints, activation shards, or
  generated completions.
- Real GPU runs should start with `next-chameleons real-run-plan`.
- Exact paper sources, local-manifest sources, and approximate substitutes must
  be labeled separately in configs and reports.
- Keep synthetic smoke paths CPU-friendly.
- If a new setting cannot be swept with `configs/sweep/*.yaml`, add the sweep
  surface before adding more one-off scripts.
