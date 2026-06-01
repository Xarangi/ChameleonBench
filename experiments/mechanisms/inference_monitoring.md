# Inference Monitoring

Inference monitoring is an integrated mechanism, not a training regime. It
attaches a Judge to activation traces during generation, replay, or synthetic
smoke runs.

Config:

```text
configs/mechanism/inference_monitoring.yaml
```

Used by:

- `multiturn_self_redteam`
- `grpo_probe_reward`
- `safety_benchmark`
- `experiment-smoke` reports

Experimental goal:

Make chameleon behavior observable during inference-like episodes, especially
when a model is trying multiple turns before a fixed turn budget expires. This
mechanism supports later generation/replay jobs where raw text is redacted but
Judge traces remain auditable.

Report:

- monitor flag rate,
- per-Probe score traces,
- escalation events,
- redacted prompt/completion hashes when real text is involved.
