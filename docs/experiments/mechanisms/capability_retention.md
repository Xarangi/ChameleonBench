# Capability Retention And Obfuscation Tax

Capability retention is an integrated reporting gate, not a separate experiment.
Every serious chameleon result needs it because a model can trivially evade
activation monitors by losing the behavior or task competence we intended to
preserve.

Config:

```text
configs/mechanism/capability_retention.yaml
```

Used by:

- paper replication,
- adaptive and curriculum tracks,
- multi-probe evasion,
- multi-turn self-red-team,
- GRPO probe reward,
- language generalization,
- safety benchmark.

Experimental goal:

Quantify the obfuscation tax: how much normal behavior, target behavior, or
capability is lost while evasion improves. This should be reported beside every
seen-Judge and held-out-Judge evasion metric.

Report:

- behavior gate pass/fail,
- KL to reference behavior,
- task or benchmark accuracy retention,
- safety-likelihood retention,
- tax curves over adaptive/curriculum rounds.
