# Standalone plugin architecture

Next Chameleons is a standalone `uv` Python project with Hydra configs, Typer CLIs,
and explicit registries for datasets, activation extractors, probes, judges,
probe selectors, training backends, metrics, and reports. This keeps the
mechanistic-interpretability pipeline independent from nearby social-simulation
repos while making the post-paper adaptive Judge work extensible.
