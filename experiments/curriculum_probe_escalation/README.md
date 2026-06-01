# Curriculum Probe Escalation

## Experimental Goal

Escalate from simpler Probes to stronger Judges as the model learns to evade
earlier monitors. Measure how far the model can progress before it fails to
evade, loses functionality, or stops generalizing to held-out probes.

This track answers: what is the model's obfuscation frontier?

## Config

```text
configs/experiment/curriculum_probe_escalation.yaml
```

The default curriculum implementation is `ProbeEscalationCurriculum`.

## Default Escalation

1. Paper linear probe.
2. MLP plus attention probes.
3. Four-layer ensemble.
4. Eight-layer ensemble.
5. Active probe-pool Judge.
6. Oracle-style Judge.

## Key Knobs

- `rounds`: number of train/evaluate/promote cycles.
- `promotion.promote_top_k`: how many discovered Probes join the next Judge.
- `promotion.diversity_weight`: how much to prefer new probe families.
- `promotion.min_detection_rate`: minimum candidate strength.
- curriculum turn budgets: start small, increase as Judge strength increases.

## Metrics

- frontier depth: hardest Judge level passed,
- zero-shot evasion capacity: held-out evasion after each escalation,
- capability retention per round,
- obfuscation tax per round,
- promoted probe diversity,
- round-by-round Judge score curves.

## Interpretation

Strong result:
The model repeatedly evades stronger, newly promoted Probes while retaining
capability and target behavior.

Important negative result:
The model passes seen probes but fails immediately on newly discovered probes.

Safety-relevant result:
Frontier depth increases with model capability or scale, suggesting stronger
models learn monitor evasion more readily.

## Commands

```bash
uv run next-chameleons config-check curriculum_probe_escalation
uv run next-chameleons adaptive-smoke \
  --experiment adaptive_2b \
  --rounds 6 \
  --obfuscation-strength 0.45 \
  --output-dir runs/curriculum_smoke
```

## Results To Add Later

- frontier-depth curve,
- promoted probe list by round,
- seen/held-out Judge table,
- capability tax over rounds,
- stopping reason for each model.
