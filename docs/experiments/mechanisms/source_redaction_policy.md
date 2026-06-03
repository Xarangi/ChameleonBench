# Source Redaction Policy

The source redaction policy is infrastructure for controlled safety artifacts,
not an experiment. It governs where raw harmful/deceptive text, generated
completions, activations, and checkpoints may live.

Config:

```text
configs/mechanism/source_redaction_policy.yaml
```

Used by:

- paper replication real runs,
- adaptive real runs,
- safety benchmark runs,
- any future run involving raw safety data or generated harmful/deceptive text.

Experimental goal:

Keep results reproducible without committing raw safety data. The repo should
store manifests, source revisions, checksums, redacted fixtures, metrics, and
plots. Raw text and heavy artifacts belong under `$SCRATCH`.

Report:

- dataset manifest,
- source revision and checksum,
- redaction status,
- artifact root and raw-cache root,
- no raw text in repo-tracked outputs.
