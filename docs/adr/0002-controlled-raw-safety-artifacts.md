# Controlled raw safety artifacts

Raw harmful or deceptive text, generated completions, activation shards, and
checkpoints are stored only under `$SCRATCH` or a researcher-selected controlled
artifact root, never in git. The repository commits manifests, source
references, checksums, redacted fixtures, aggregate metrics, and plots so the
replication remains auditable without leaking sensitive raw artifacts.
