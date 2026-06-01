# Next Chameleons

This context defines the research language for the Next Chameleons benchmark and
training kit. It is a glossary, not an implementation spec.

## Language

**Probe**:
One detector trained or evaluated on model internal states.
_Avoid_: Judge, monitor, classifier when referring to the project-level detector contract

**Probe Suite**:
A named set of Probes that are trained, calibrated, evaluated, or promoted together.
_Avoid_: Judge when no aggregation or escalation policy is implied

**Judge**:
An ensemble plus aggregation and escalation policy over one or more Probes.
_Avoid_: Probe when referring to ensemble behavior

**Model Organism**:
A deliberately trained model used to study a specific safety-relevant capability.
_Avoid_: production model, deployed model

**Neural Chameleon**:
A Model Organism trained to preserve visible behavior while reducing detectability
of a triggered concept in activations.
_Avoid_: jailbreak model, generic adversarial model

**Chameleon Benchmark**:
The evaluation suite that measures evasion, selectivity, robustness to stronger
Judges, capability retention, and source/data coverage.
_Avoid_: single paper run, one-off smoke test

**Training Regime**:
A repeatable strategy for inducing or testing chameleon behavior, such as paper
SFT, multi-probe evasion, curriculum escalation, multi-turn adaptation, or GRPO.
_Avoid_: backend when referring to the research objective instead of PEFT/full/RL machinery

**Curriculum**:
A schedule that changes probe difficulty, Judge strength, feedback, data domains,
or turn budget across rounds.
_Avoid_: sweep when the change is adaptive rather than a static matrix

**Integrated Mechanism**:
A reusable research component that only has meaning inside a Training Regime,
Adaptive Episode, Chameleon Benchmark, or artifact policy.
_Avoid_: experiment family when the component does not define an independent
model-training or evaluation question

**Adaptive Episode**:
One multi-turn attempt in which a model receives feedback and tries to adjust its
in-context behavior or latent trajectory.
_Avoid_: training round when weights are not necessarily updated

**Feedback Channel**:
The information returned to a model during an Adaptive Episode, such as binary
verdict, coarse reason, probe scores, or highest feedback.
_Avoid_: reward unless used by an RL objective

**Probe Discovery**:
The process of training candidate Probes on a latest checkpoint, calibrating
them, scoring detection plus diversity, and promoting stronger Probes into a Judge.
_Avoid_: static evaluation

**Activation Cache**:
A stored set of internal model states for configured layers, token positions, or sites.
_Avoid_: raw model dump, checkpoint

**Controlled Raw Cache**:
A non-git artifact area for raw safety corpora, completions, activation shards,
and checkpoints.
_Avoid_: fixture, committed dataset

**Capability Retention**:
The degree to which model behavior and benchmark performance remain close to the
reference model after chameleon training or adaptation.
_Avoid_: safety success

**Obfuscation Tax**:
The cost of evasion measured as loss of capability, task performance, likelihood
on target corpora, or behavioral fidelity.
_Avoid_: training loss

## Relationships

- A **Neural Chameleon** is a **Model Organism**.
- A **Chameleon Benchmark** evaluates one or more **Training Regimes**.
- A **Judge** contains a **Probe Suite** plus aggregation and escalation policy.
- A **Probe** reads an **Activation Cache** during training or evaluation.
- **Probe Discovery** promotes stronger **Probes** into later **Judges**.
- A **Curriculum** may alter **Probe Suites**, **Feedback Channels**, domains, and turn budgets.
- A **Controlled Raw Cache** may contain raw data that must not appear in committed fixtures.

## Example Dialogue

> **Dev:** "Did the model pass the probe?"
> **Domain expert:** "Say it passed one **Probe** only if that detector missed it; say it passed the **Judge** only if the ensemble policy did not flag it."

## Flagged Ambiguities

- "Probe" and "judge" were initially used interchangeably; resolved: a **Probe** is one detector, while a **Judge** is the ensemble and escalation policy.
- "Training mode" now means **Training Regime** when discussing research logic; **Training Backend** is reserved for implementation machinery such as LoRA, full fine-tuning, or GRPO.
