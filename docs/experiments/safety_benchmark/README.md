# Safety Benchmark

## Experimental Goal

Turn chameleon learnability into a model benchmark. Compare models by how easily
they learn activation obfuscation, how far they progress through stronger
Judges, how much they can adapt in context, and how much capability they lose.

This track answers: are stronger models better at learning and applying
activation-monitor evasion, and can we quantify the safety-relevant gap?

## Config

```text
configs/experiment/safety_benchmark.yaml
```

## Benchmark Axes

- model family and scale,
- training regime,
- probe frontier depth,
- in-context adaptation,
- curriculum adaptation,
- feedback channel,
- held-out probe family,
- held-out data/domain.

## Core Metrics

- fixed-FPR TPR drop,
- seen-Judge evasion,
- held-out-Judge evasion,
- turns to bypass,
- frontier depth,
- zero-shot evasion capacity,
- capability retention,
- obfuscation tax,
- safety likelihood retention.

## Capability Preservation

Capability preservation should be measured with both benchmark scores and
likelihood/behavior checks:

- MMLU,
- GSM8K,
- BoolQ,
- CrowS-Pairs,
- SocialIQA,
- TruthfulQA,
- PIQA,
- likelihood on deception and harmfulness corpora.

## Obfuscation Tax

Obfuscation tax should include:

- benchmark drop relative to reference,
- KL drift from base completions,
- loss of target task quality,
- increased refusal or incoherence,
- likelihood shift on safety corpora,
- extra turns or tokens needed to evade.

## Inference Monitoring

The library includes `next_chameleons.inference.monitoring` so Judges can be
attached to activation batches during generation or replay. Inference traces
should save Judge IDs, probe scores, thresholds, and redacted metadata.

## Results To Add Later

- model-by-regime benchmark table,
- frontier-depth plot,
- in-context adaptation plot,
- capability tax plot,
- held-out Judge robustness table,
- benchmark card for each tested model.
