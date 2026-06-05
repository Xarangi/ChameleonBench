# Development Notes

This page is for maintainers and AI agents working on the repository itself.
User-facing experiment documentation should stay on the README, Library API,
Replication, and Experiment pages.

## Verification

```bash
uv sync --extra dev --extra train --extra analysis --extra docs
uv run --extra dev ruff check src tests scripts
uv run --extra dev pytest
uv run --extra docs properdocs build --strict
```

## Documentation Site

The site uses Material for MkDocs through ProperDocs.

```bash
uv sync --extra docs
uv run --extra docs properdocs serve
uv run --extra docs properdocs build --strict
```

The static site is built into `site/`. GitHub Pages deployment is configured in
`.github/workflows/docs.yml` and uses `uv sync --extra docs --locked`.

## Visual Settings

High-level site structure is controlled by `properdocs.yml`; the editorial
visual treatment is in `docs/assets/css/editorial.css`.

Current style settings:

- theme: `material`
- palette: off-white / charcoal with muted grey accents
- body font: Lora
- code font: Fira Code
- navigation: top tabs, integrated table of contents, copyable code blocks
- math: `pymdownx.arithmatex`

Use `properdocs.yml` for theme features and navigation. Use
`docs/assets/css/editorial.css` for colors, spacing, typography, and Material
CSS variables.

## Slurm Submission Shape

Paper submission scripts use A100 queues and `$SCRATCH` caches by default. The
dependency graph is intentionally not fully serialized:

- smoke/data readiness gates the first pilot;
- full 2B and primary/family seed-17 jobs can run in parallel after the pilot;
- 9B seeds 23/41 wait for primary 9B seed 17;
- adaptive/extension runs should use their own sweep scripts.

Compute jobs set `UV_NO_SYNC=1` and `UV_OFFLINE=1`, so the local project venv
must be prepared before submission.
