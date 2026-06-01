# Hard rename to Next Chameleons

The project is renamed from `chameleons` to `next-chameleons`, with import
package `next_chameleons` and CLI `next-chameleons`.

We intentionally do not keep a compatibility shim for `chameleons`. The project
is becoming a library-like benchmark and training kit, so a single public name is
more valuable than preserving old local commands from the initial scaffold.

Consequences:

- All docs, tests, scripts, configs, and imports use `next_chameleons`.
- The CLI is invoked as `uv run next-chameleons ...`.
- Existing local artifacts under old names may remain on disk, but new outputs
  default to `next_chameleons_artifacts` and `next_chameleons_raw_cache`.
