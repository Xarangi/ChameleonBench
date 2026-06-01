# Language Generalization

## Experimental Goal

Replicate the paper appendix idea that a model can learn a general evasion
mechanism over language concepts and transfer it to unseen languages.

This track answers: is chameleon behavior tied to specific trained concepts, or
does it become a reusable semantic operation?

## Config

```text
configs/experiment/language_generalization.yaml
```

## Default Split

Train languages:

- Japanese,
- German,
- Chinese,
- French,
- Spanish,
- Italian,
- Russian,
- Korean.

Test languages:

- Portuguese,
- Turkish,
- Dutch,
- Polish,
- Swedish,
- Arabic,
- Hindi,
- Thai,
- Vietnamese,
- Greek.

## Metrics

- language probe TPR drop at fixed 1% FPR,
- seen vs unseen language evasion,
- capability tax,
- semantic transfer strength,
- base-trigger control.

## Commands

```bash
uv run next-chameleons config-check language_generalization
uv run next-chameleons real-run-plan language_generalization \
  --output-path runs/language_generalization_plan.json \
  --allow-unpinned
```

## Results To Add Later

- seen and unseen language TPR table,
- language-family breakdown,
- capability tax comparison,
- notes on whether transfer looks semantic or keyword-like.
