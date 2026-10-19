"""TICA run configs: configs/tica/<name>.yaml -> typed dataclasses.

Unknown keys raise, so a typo cannot silently fall back to a default. Sections:
- model (path, dtype, enable_thinking, batch_size);
- lora;
- probes (dir of an l1_fit output, optional layer);
- data;
- out;
- train (SupervisedConfig, with nested objective);
- train_mix;
- eval (EvalConfig);
- rounds (RoundsConfig minus train/eval, which come from the top level);
- rl (RLConfig).
"""

from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path

from next_chameleons.tica.evaluation import EvalConfig
from next_chameleons.tica.objectives import (
    ObjectiveConfig,  # noqa: F401  (nested in SupervisedConfig)
)
from next_chameleons.tica.rounds import CollectConfig, FitConfig, RoundsConfig
from next_chameleons.tica.trainers.rl import RLConfig
from next_chameleons.tica.trainers.supervised import SupervisedConfig
from next_chameleons.tica.windows import WindowEnv, WindowEval, WindowTrain


def from_dict(cls, d: dict | None):
    """Build dataclass `cls` from a dict, recursing into dataclass-typed fields. Raises on unknown
    keys. Lists become tuples where the field's default is a tuple."""

    d = dict(d or {})
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kw = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v, t = d[f.name], hints[f.name]
        if dataclasses.is_dataclass(t) and isinstance(v, dict):
            v = from_dict(t, v)
        elif isinstance(v, list):
            default = f.default if f.default is not dataclasses.MISSING else None
            if isinstance(default, tuple) or typing.get_origin(t) is tuple:
                v = tuple(v)
        kw[f.name] = v
    return cls(**kw)


@dataclass
class ModelConfig:
    path: str
    dtype: str = "bfloat16"
    enable_thinking: bool = False
    batch_size: int = 8
    grad_ckpt: bool = True  # training only; False = faster, more activation memory
    fla_grad: bool = False  # Qwen3.5: fla kernel in grad passes too (needs triton >= 3.3)
    keep_think_history: bool = False  # every assistant turn keeps the empty think block


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"


@dataclass
class ProbeConfig:
    dir: str  # an l1_fit.py output directory
    layer: int | None = None  # None = quality.json's primary layer
    # Round-0 pool: "all" = every bank_L*.npz in dir (primary first), "primary" = the primary bank
    # only, or an explicit list of layers. The shown monitor is sampled per episode from the pool
    # and not named, so "all" makes the monitor's layer something only feedback can reveal
    # (DESIGN-TICA REVISION 10-02).
    pool_layers: str | list[int] = "all"
    # Hidden-mode runs: a JSON list of concepts whose probe still detects them inside a PAIRED
    # text (scripts/l1_pair_check.py). ANDed into the primary bank's gate by scripts/tica.py.
    pair_gate: str | None = None
    # Layers whose base bank enters the pool as EVAL-ONLY. The headline question is what the
    # single checkpoint does against a probe it never trained against, so at least one layer is
    # held out (10-10: a dodge trained at L12 transfers to L16 but fails at L8/L20/L24).
    held_out_layers: list = field(default_factory=list)


@dataclass
class EpisodeStyle:
    """Shared by training, collection and evaluation (episodes.py module docstring)."""

    feedback: str = "score"  # "score" | "verdict"
    hidden: bool = False  # True = unnamed monitored concept, paired texts (needs verdict)
    # "repeat" = copy the text (Level 1); "answer" = brief + the AxBench instruction, the fixed text
    # is the ANSWER (NC format; tica/answer_format.py). inputs = {text_id: instruction} sidecar.
    format: str = "repeat"
    inputs: str = "/scratch/sneheel/next_chameleons_raw_cache/tica/level1_inputs.json"
    # "legacy" = brief in the first user turn + free-text notices; "tagged" = monitor description
    # as the system prompt + a status line (ON + concept / OFF) every turn (answer_format, 10-17a)
    layout: str = "legacy"

    def __post_init__(self) -> None:
        if self.layout not in ("legacy", "tagged"):
            raise ValueError(self.layout)
        if self.layout == "tagged" and self.format != "answer":
            raise ValueError("layout 'tagged' needs format 'answer'")
        if self.feedback not in ("score", "verdict"):
            raise ValueError(self.feedback)
        if self.format not in ("repeat", "answer"):
            raise ValueError(self.format)
        if self.format == "answer" and self.hidden:
            raise ValueError("answer format supports named episodes only")
        if self.hidden and self.feedback != "verdict":
            raise ValueError("hidden episodes need feedback: verdict")


@dataclass
class FinalConfig:
    """Final multi-concept, multi-probe-type dodger (TICA-FINAL-DODGER-DESIGN.md). Empty `split` =
    off (every older config). With a split, load_run_config sets episode_concepts to the first
    `n_train_concepts` TRAIN concepts of the split (the ladder rung); held-out concepts and held-out
    probe kinds are eval-only and are never trained on (RunConfig.__post_init__ checks)."""

    split: str = ""  # concept split JSON (scripts/tica_concept_split.py), relative to the YAML
    n_train_concepts: int = 0  # ladder rung: first N train concepts (0 = all)
    # per-concept published positives (scripts/build_multi_concept_expansion.py: index.json + files)
    extra_pos_dir: str = ""
    # probe TYPES: refits in training are drawn from train_refit_kinds; heldout_refit_kinds are
    # eval-only (race --probe-kind); the two must be disjoint.
    train_refit_kinds: list = field(default_factory=lambda: ["linear"])
    heldout_refit_kinds: list = field(default_factory=list)
    # Capability anchor: token KL(base || policy) on base-model greedy answers to unmonitored
    # instruction prompts (Dolly-15k + GSM8K TRAIN; never the eval items), every step.
    anchor_weight: float = 0.0
    anchor_prompts: int = 8  # prompts per step
    anchor_max_new_tokens: int = 256
    anchor_data: list = field(default_factory=lambda: [
        "/scratch/sneheel/next_chameleons_raw_cache/tica/dolly15k/databricks-dolly-15k.jsonl",
        "/scratch/sneheel/next_chameleons_raw_cache/capability/gsm8k_train.json"])
    # Capability gate on checkpoints (tica_capability.py, GSM8K 200 + MMLU 600, clean prompt):
    # a checkpoint fails if it keeps less than this fraction of the BASE accuracy.
    cap_gate_every: int = 25
    cap_gate_min_gsm8k: float = 0.9
    cap_gate_min_mmlu: float = 0.95
    # Overall false-positive rate of the CUMULATIVE judge ("caught if ANY deployed probe fires") on
    # held-out concept-free / other-concept answers; the union is recalibrated to this rate.
    judge_union_fpr: float = 0.01
    concept_switch_prob: float = 0.0  # P(a real_score episode switches its named concept)
    switch_min_segment: int = 3  # turns per concept segment (>= 3: fail, pass, re-caught)
    switch_max: int = 2  # switches per switching episode (as many as fit)
    switch_return_prob: float = 0.25  # P(a switch returns to an earlier concept of the episode)
    # TURN BUDGET (10-19, default off): switch points follow the race instead of being drawn
    # in advance (episodes.budget_action). Every probe (first probe, refit, switch probe) is
    # deployed only with >= budget_min_runway judged turns before the next hard boundary, and
    # stands at most budget_max_turns judged turns (not beaten by then = a loss -> switch, or the
    # episode ends). No cap on refits per segment. A switching episode's first segment ends by
    # turn T - budget_switch_seg, so its first switch is guaranteed and gets >= budget_switch_seg
    # turns; later switches only after a loss / a pass without runway, with >= budget_switch_seg
    # turns left (up to switch_max). Once no new probe fits, the remaining turns are an unjudged
    # tail without loss (shown "Monitor: OFF"). concept_switch_prob = share of real_score episodes
    # with switches; switch_min_segment is unused.
    turn_budget: bool = False
    budget_min_runway: int = 5
    budget_max_turns: int = 8
    budget_switch_seg: int = 8


@dataclass
class RunConfig:
    name: str
    model: ModelConfig
    probes: ProbeConfig
    out: str
    episodes: EpisodeStyle = field(default_factory=EpisodeStyle)
    data: str = "/scratch/sneheel/next_chameleons_raw_cache/tica/level1_data.json"
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: SupervisedConfig = field(default_factory=SupervisedConfig)
    train_mix: dict = field(
        default_factory=lambda: {
            "real_score": 0.6,
            "mismatch": 0.2,
            "nobrief": 0.1,
            "neg_text": 0.1,
        }
    )
    eval: EvalConfig = field(default_factory=EvalConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    rounds: int = 3
    # Concepts that may be MONITORED in episodes (empty = every gated concept). Other concepts stay
    # in the bank and supply the branch negatives.
    episode_concepts: list = field(default_factory=list)
    # Adaptive RL: a saved MonitorPool directory to use as the STARTING bank (e.g. an l3loop
    # pool_next with the base probe + its branch refits). None = the round-0 pool from `probes`.
    start_pool: str | None = None
    # pool monitor names to drop after loading start_pool (e.g. a dead refit; 10-12c r2 det 0.00)
    start_pool_exclude: list[str] = field(default_factory=list)
    rl: RLConfig = field(default_factory=RLConfig)
    final: FinalConfig = field(default_factory=FinalConfig)

    def __post_init__(self) -> None:
        f = self.final
        both = set(f.train_refit_kinds) & set(f.heldout_refit_kinds)
        if both:
            raise ValueError(f"probe kinds both trained and held out: {sorted(both)}")
        if f.split and not self.episode_concepts:
            raise ValueError("final.split set but episode_concepts unresolved "
                             "(use load_run_config)")
        # Depth hold-out (10-15e): no TRAINED probe kind may read a held-out layer (the static
        # held-out probes' layers, or the layer of any held-out "kind@L"), or the depth axis of the
        # eval would be trained against.
        from next_chameleons.tica.probe_zoo import kind_layers, split_kind

        held = {int(x) for x in self.probes.held_out_layers}
        held |= {split_kind(k)[1] for k in f.heldout_refit_kinds if split_kind(k)[1] is not None}
        primary = self.probes.layer if self.probes.layer is not None else 12
        bad = {k: sorted(set(kind_layers(k, primary)) & held) for k in f.train_refit_kinds
               if split_kind(k)[0] not in ("linear", "mlp", "multilayer")
               or split_kind(k)[1] is not None}
        bad = {k: v for k, v in bad.items() if v}
        # factored kind sampling (adaptive.kind_layers, 10-16l): every sampled "arch@L" is trained
        kl = sorted({int(x) for x in getattr(self.rl.adaptive, "kind_layers", []) or []} & held)
        if kl:
            bad["adaptive.kind_layers"] = kl
        if bad:
            raise ValueError(f"trained probe kinds read held-out layers: {bad}")

    def eval_config(self) -> EvalConfig:
        return dataclasses.replace(
            self.eval,
            feedback=self.episodes.feedback,
            hidden=self.episodes.hidden,
            # A single-concept run restricts the eval to the same concept it trains on.
            only_concepts=tuple(self.eval.only_concepts or self.episode_concepts),
        )

    def sampler_kwargs(self) -> dict:
        kw = {"feedback": self.episodes.feedback, "hidden": self.episodes.hidden}
        if self.final.concept_switch_prob:
            f = self.final
            kw.update(switch_prob=f.concept_switch_prob, switch_min_segment=f.switch_min_segment,
                      switch_max=f.switch_max, switch_return_prob=f.switch_return_prob)
        if self.final.turn_budget:
            f = self.final
            kw.update(budget={"R": f.budget_min_runway, "M": f.budget_max_turns,
                              "S": f.budget_switch_seg})
        return kw

    def rounds_config(self) -> RoundsConfig:
        return RoundsConfig(
            rounds=self.rounds,
            train=self.train,
            train_mix=self.train_mix,
            collect=self.collect,
            fit=self.fit,
            eval=self.eval_config(),
            style=self.sampler_kwargs(),
            seed=self.train.seed,
            episode_concepts=tuple(self.episode_concepts),
        )

    def primary_layer(self) -> int:
        if self.probes.layer is not None:
            return self.probes.layer
        return int(
            json.loads((Path(self.probes.dir) / "quality.json").read_text())["primary_layer"]
        )

    def pool_layers(self) -> list[int]:
        """Layers of the round-0 pool, primary first."""

        primary = self.primary_layer()
        spec = self.probes.pool_layers
        if spec == "primary":
            return [primary]
        if spec == "all":
            found = sorted(
                int(f.stem.split("_L")[1]) for f in Path(self.probes.dir).glob("bank_L*.npz")
            )
        else:
            found = sorted(int(x) for x in spec)
        return [primary] + [x for x in found if x != primary]


def load_run_config(name_or_path: str) -> RunConfig:
    from next_chameleons.config import find_config_path, load_yaml

    p = Path(name_or_path)
    path = (
        p
        if p.suffix in (".yaml", ".yml") and p.exists()
        else find_config_path("tica", name_or_path)
    )
    d = load_yaml(path)
    split = (d.get("final") or {}).get("split")
    if split:
        resolve_final(d, Path(path).parent)
    return from_dict(RunConfig, d)


def resolve_final(d: dict, base: Path) -> None:
    """final.split -> absolute path, episode_concepts = the rung's train concepts (in place)."""

    fin = d["final"]
    sp = Path(fin["split"])
    sp = sp if sp.is_absolute() else base / sp
    fin["split"] = str(sp)
    split = json.loads(sp.read_text())
    n = int(fin.get("n_train_concepts", 0)) or len(split["train"])
    train = split["train"][:n]
    held = set(split["heldout"]) | set(split.get("heldout_far", []))
    if held & set(train):
        raise ValueError(f"split: trained AND held out: {sorted(held & set(train))}")
    if d.get("episode_concepts") and list(d["episode_concepts"]) != train:
        raise ValueError("final.split and an explicit episode_concepts disagree")
    d["episode_concepts"] = train


@dataclass
class WindowRunConfig:
    """L3 fast track S1/S2 (tica/windows.py; DESIGN-L3-PROBEWALL v2, PREREG A3c)."""

    name: str
    model: ModelConfig
    probes: str  # an l1_fit.py output directory (bank_L{layer}.npz)
    out: str
    env: WindowEnv
    data: str = "/scratch/sneheel/next_chameleons_raw_cache/tica/level1_data.json"
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: WindowTrain = field(default_factory=WindowTrain)
    eval: WindowEval = field(default_factory=WindowEval)


def load_window_config(name_or_path: str) -> WindowRunConfig:
    from next_chameleons.config import find_config_path, load_yaml

    p = Path(name_or_path)
    path = (
        p
        if p.suffix in (".yaml", ".yml") and p.exists()
        else find_config_path("tica", name_or_path)
    )
    return from_dict(WindowRunConfig, load_yaml(path))
