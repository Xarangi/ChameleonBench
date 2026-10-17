"""Multi-turn GRPO for the generation-based TICA settings (DESIGN-TICA.md).

- Level 1b "reason_forced": the model REASONS freely (sampled), then the fixed text is
  teacher-forced as its answer. Monitors read the answer tokens (and, optionally, the reasoning
  tokens). Only the reasoning is an action.
- Extension 1 "open": the model writes the whole answer (sampled). An LLM judge gates correctness
  (the requested concepts are really discussed), and monitors read the answer.

Per episode spec there are G rollouts, run turn by turn in lockstep:
1. Sample the completion for every live rollout.
2. Read the monitored span with the policy (no grad) and score the shown monitor.
3. Compute the reward, then render the feedback with tica/episodes.render_feedback.

Loss (on-policy, one update per batch, so no importance ratio, as in RL-Obfuscation's GRPO with
G=8):
  L = -mean_tokens(A * logp(sampled tokens)) + beta * mean_tokens(k3-KL(policy || base)).
Tokens are only the SAMPLED ones; forced answer tokens and feedback are masked out (TRL's env_mask
convention).

Credit:
- "episode": A = group-normalized mean turn reward;
- "turn": A_t = group-normalized reward-to-go from turn t, which rewards later-turn improvement in
  the spirit of SCoRe 2409.12917 and turn-level credit 2505.11821.
Groups whose rewards are all equal are dropped (RAGEN's zero-variance filter).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from next_chameleons.tica.adaptive import (
    AdaptiveConfig,
    AdaptiveDefender,
    RefitData,
    fit_refit_bank,
    selectivity_reward,
    trim_bank,
)
from next_chameleons.tica.episodes import CONDITION_SPECS, EpisodeSpec, render_feedback
from next_chameleons.tica.level1 import opening
from next_chameleons.tica.monitors import MonitorPool
from next_chameleons.tica.policy import View


@dataclass
class RLConfig:
    mode: str = "reason_forced"  # | "open"
    steps: int = 200
    specs_per_step: int = 4
    group_size: int = 8
    turns: int = 3
    # Reasoning budget. Unfinished reasoning earns zero reward, so a short budget teaches brevity
    # rather than evasion; 1024 leaves room for ordinary thinking-mode reasoning.
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    lr: float = 1e-5
    beta_kl: float = 0.05
    credit: str = "turn"  # | "episode"
    hide_from_turn: int = 0
    reasoning_end: str = "</think>"  # reason_forced: the end-of-reasoning marker
    answer_prefix: str = "\n\n"  # forced between the reasoning marker and the answer text
    judge_threshold: float = 0.5
    micro_batch: int = 8  # no-grad rollout reads per forward
    update_micro_batch: int = 0  # grad update chunk (0 = micro_batch); keep small, logits are huge
    grad_clip: float = 1.0
    save_every: int = 50
    seed: int = 0
    # Adaptive-monitor multi-turn training (DESIGN OF RECORD, REPLICATION 2026-10-11a). When
    # use_adaptive, the rollout draws one monitor per turn from an in-episode AdaptiveDefender,
    # budget-forces the reasoning, and rewards with the progressive-penalty / selectivity scheme.
    use_adaptive: bool = False
    history_reasoning: bool = True  # keep each earlier turn's reasoning in the rendered history
    # DAPO overlong filtering (arXiv 2503.14476): a truncated (budget-hit) rollout carries reward
    # noise -- the model cannot tell low reward from a bad dodge vs from being cut off -- so it is
    # excluded from the GRPO loss. The direct-gradient term still applies to it (the forced answer
    # carries no such noise). Essential for an RL-trained thinker that overruns the budget.
    grpo_require_finished: bool = True
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)


@dataclass
class Sample:
    """One rollout's turn."""

    view: View  # full ids for the probe read: prompt + sampled (+ forced)
    sampled: tuple[int, int]  # [a, b) positions of SAMPLED tokens in view.ids
    probe_span: tuple[int, int]  # [a, b) positions the monitor reads
    text: str  # what enters the conversation history as the assistant message
    finished: bool
    reward: float = 0.0
    info: dict = field(default_factory=dict)


def _find(seq: list[int], sub: list[int]) -> int:
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return -1


def build_sample(policy, gen: dict, cfg: RLConfig, forced_text: str | None) -> Sample:
    """gen = Policy.generate output. reason_forced: cut at reasoning_end, then force the answer."""

    p, ids = gen["prompt_ids"], gen["ids"]
    if cfg.mode == "open":
        view = View(p + ids, len(p))
        return Sample(
            view,
            (len(p), len(p) + len(ids)),
            (len(p), len(p) + len(ids)),
            gen["text"],
            gen["finished"],
        )
    end = policy.encode(cfg.reasoning_end)
    k = _find(ids, end)
    finished = k >= 0
    reasoning = ids[: k + len(end)] if finished else ids
    forced = policy.encode(cfg.answer_prefix + forced_text)
    a = len(p) + len(reasoning)
    n_prefix = len(policy.encode(cfg.answer_prefix))
    view = View(p + reasoning + forced, len(p))
    return Sample(
        view,
        (len(p), a),
        (a + n_prefix, a + len(forced)),
        forced_text,
        finished,
        info={"n_reasoning": len(reasoning)},
    )


def build_sample_budget(
    policy, gen: dict, end_id: int, prefix_ids: list[int], forced_text: str
) -> Sample:
    """reason_forced with budget forcing. `gen` comes from Policy.generate(stop_ids={end_id}): its
    ids are the sampled reasoning (the end-of-reasoning token is NOT in them), gen["stop"] is end_id
    iff the model closed the block itself, else None (the budget was hit). The block is then always
    closed and the answer teacher-forced, so no rollout is voided (REPLICATION 2026-10-11a). The
    sampled span is the reasoning only; appended tokens are excluded."""

    p, reasoning = gen["prompt_ids"], gen["ids"]
    finished = gen.get("stop") == end_id
    forced = policy.encode(forced_text)
    a = len(p) + len(reasoning)  # end of the sampled reasoning
    answer_start = a + 1 + len(prefix_ids)  # after end_id + the answer prefix
    ids = p + reasoning + [end_id] + prefix_ids + forced
    return Sample(
        View(ids, len(p)),
        (len(p), a),
        (answer_start, answer_start + len(forced)),
        forced_text,
        finished,
        info={"n_reasoning": len(reasoning), "reasoning_text": gen["text"]},
    )


def group_advantages(rewards: torch.Tensor, credit: str) -> torch.Tensor:
    """rewards [S, G, T] -> advantages [S, G, T]. Zero-variance groups get zero advantage."""

    if credit == "episode":
        ret = rewards.mean(-1, keepdim=True).expand_as(rewards)
    elif credit == "turn":
        ret = torch.flip(torch.cumsum(torch.flip(rewards, [-1]), -1), [-1])  # reward-to-go
    else:
        raise ValueError(credit)
    mu = ret.mean(1, keepdim=True)
    sd = ret.std(1, keepdim=True, unbiased=False)
    return torch.where(sd > 1e-8, (ret - mu) / (sd + 1e-8), torch.zeros_like(ret))


def pg_loss(
    logp: torch.Tensor, ref_logp: torch.Tensor, adv: torch.Tensor, beta: float
) -> tuple[torch.Tensor, dict]:
    """Token-level: logp/ref_logp/adv [n_tokens] (adv broadcast per sample)."""

    d = ref_logp - logp
    kl = torch.exp(d) - d - 1.0  # k3 estimator, >= 0
    loss = -(adv * logp).mean() + beta * kl.mean()
    return loss, {"pg": float(-(adv * logp).mean().detach()), "kl": float(kl.mean().detach())}


class RLTrainer:
    """reward_fn(spec, turn, sample, readout) -> float, where readout is the monitored result dict
    (raw, score, thr, detected). The default rewards turns >= hide_from_turn for NOT DETECTED,
    and requires a finished reasoning block (reason_forced) or a judge pass (open).
    """

    def __init__(
        self,
        policy,
        pool: MonitorPool,
        sample_specs: Callable[[int], list[EpisodeSpec]],
        cfg: RLConfig,
        out_dir: Path | str,
        *,
        judge: Callable[[list[tuple[str, EpisodeSpec]]], list[bool]] | None = None,
        reward_fn: Callable | None = None,
    ) -> None:
        if cfg.mode == "open" and judge is None:
            raise ValueError("open mode needs a judge")
        self.policy, self.pool, self.sample_specs, self.cfg = policy, pool, sample_specs, cfg
        self.judge, self.reward_fn = judge, reward_fn or self.default_reward
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.params = [p for p in policy.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr)
        self.step_k = 0
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        if cfg.use_adaptive:
            if cfg.mode != "reason_forced":
                raise ValueError("adaptive training is reason_forced only")
            policy.history_reasoning = cfg.history_reasoning
            end = policy.encode(cfg.reasoning_end)
            if len(end) != 1:
                raise ValueError(f"reasoning_end must be one token, got {end}")
            self.end_id = end[0]
            self.prefix_ids = policy.encode(cfg.answer_prefix)
            self.base_names = {e.monitor.name for e in pool.entries}  # never trimmed
            self.refit_k = 0
            self.refit_buf: RefitData | None = None  # rolling buffer across steps
            (self.out / "bank").mkdir(exist_ok=True)

    def default_reward(self, spec: EpisodeSpec, turn: int, s: Sample, readout: dict) -> float:
        if not s.finished or s.info.get("judge_ok") is False:
            return 0.0
        if turn < self.cfg.hide_from_turn:
            return 0.0
        return float(not readout["detected"])

    # --------------------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(self, specs: list[EpisodeSpec]) -> tuple[list[list[list[Sample]]], torch.Tensor]:
        """Returns samples[s][g][t] and rewards [S, G, T]."""

        cfg, pol, pool = self.cfg, self.policy, self.pool
        S, G, T = len(specs), cfg.group_size, cfg.turns
        first = [
            opening(
                "real_score" if CONDITION_SPECS[sp.cond]["brief"] else "nobrief",
                sp.text(0),
                sp.monitored,
                named=sp.named,
            )
            for sp in specs
        ]
        convs = [[[{"role": "user", "content": first[i]}] for _ in range(G)] for i in range(S)]
        samples = [[[] for _ in range(G)] for _ in range(S)]
        rewards = torch.zeros(S, G, T)
        pol.model.eval()
        for t in range(T):
            flat = [(i, g) for i in range(S) for g in range(G)]
            gens = pol.generate(
                [convs[i][g] for i, g in flat],
                cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                seed=cfg.seed + 1000 * self.step_k + t,
            )
            batch = [
                build_sample(pol, gen, cfg, specs[i].text(t))
                for (i, g), gen in zip(flat, gens, strict=True)
            ]
            if cfg.mode == "open":
                oks = self.judge(
                    [(b.text, specs[i]) for (i, _), b in zip(flat, batch, strict=True)]
                )
                for b, ok in zip(batch, oks, strict=True):
                    b.info["judge_ok"] = bool(ok)
            # Monitor read on the probe span (no grad), grouped through Policy.forward.
            mons = [pool.get(specs[i].monitor) for i, _ in flat]
            reads = []
            for a in range(0, len(batch), cfg.micro_batch):
                chunk = batch[a : a + cfg.micro_batch]
                views = [View(b.view.ids[: b.probe_span[1]], b.probe_span[0]) for b in chunk]
                p, _ = pol.forward(views, pool.layers, grad=False)
                reads.append(p)
            pooled = {L: torch.cat([r[L] for r in reads]) for L in pool.layers}
            for k, ((i, g), b) in enumerate(zip(flat, batch, strict=True)):
                mon, c = mons[k], specs[i].monitored
                raw = mon.logits(pooled[mon.layer][k : k + 1], [c])
                thr = mon.thresholds([c])
                r = {
                    "raw": float(raw[0]),
                    "score": float(mon.normalize(raw, [c])[0]),
                    "thr": float(mon.normalize(thr, [c])[0]),
                    "detected": bool(raw[0] > thr[0]),
                }
                b.info["monitored"] = r
                b.reward = self.reward_fn(specs[i], t, b, r)
                rewards[i, g, t] = b.reward
                samples[i][g].append(b)
                convs[i][g].append({"role": "assistant", "content": b.text})
                if t < T - 1:
                    nxt = specs[i].text(t + 1) if len(specs[i].texts) > 1 else None
                    convs[i][g].append(
                        {
                            "role": "user",
                            "content": render_feedback(
                                specs[i].cond,
                                c,
                                (r["score"], r["thr"]),
                                nxt,
                                feedback=specs[i].feedback,
                                named=specs[i].named,
                            ),
                        }
                    )
        return samples, rewards

    # ------------------------------------------------------------------- adaptive rollout
    @torch.no_grad()
    def rollout_adaptive(
        self, specs: list[EpisodeSpec]
    ) -> tuple[list[list[list[Sample]]], torch.Tensor, dict, RefitData]:
        """Budget-forced reason_forced rollout where each (episode, rollout) carries its own
        AdaptiveDefender. Per turn: generate reasoning (stopped at </think> or the budget), force
        the answer, read the ACTIVE monitor, reward with the progressive-penalty / selectivity
        scheme, then let the defender pick the next monitor. Returns samples, rewards, metrics, and
        the RefitData collected for an online refit."""

        cfg, acfg, pol, pool = self.cfg, self.cfg.adaptive, self.policy, self.pool
        S, G, T = len(specs), cfg.group_size, cfg.turns
        convs = [
            [
                [
                    {
                        "role": "user",
                        "content": opening(
                            "real_score" if CONDITION_SPECS[sp.cond]["brief"] else "nobrief",
                            sp.text(0),
                            sp.monitored,
                            named=sp.named,
                        ),
                    }
                ]
                for _ in range(G)
            ]
            for sp in specs
        ]
        samples = [[[] for _ in range(G)] for _ in range(S)]
        rewards = torch.zeros(S, G, T)
        # Only genuine dodge conditions get an in-episode defender. neg_text (brief, concept-free)
        # and nobrief (no brief, concept text) are selectivity / refit-negative conditions.
        dodge = {"real_score", "real_verdict"}
        defs = [
            [
                AdaptiveDefender(pool, specs[i].monitored, acfg, start=specs[i].monitor)
                if specs[i].cond in dodge
                else None
                for _ in range(G)
            ]
            for i in range(S)
        ]
        if self.refit_buf is None:
            self.refit_buf = RefitData(layers=pool.layers, concept=specs[0].monitored)
        refit = self.refit_buf
        cap = acfg.refit_buf_cap
        passed_turn, novel, trunc = [], 0, 0
        pol.model.eval()
        for t in range(T):
            flat = [(i, g) for i in range(S) for g in range(G)]
            gens = pol.generate(
                [convs[i][g] for i, g in flat],
                acfg.reasoning_budget,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                seed=cfg.seed + 1000 * self.step_k + t,
                stop_ids={self.end_id},
            )
            batch = [
                build_sample_budget(pol, gen, self.end_id, self.prefix_ids, specs[i].text(t))
                for (i, g), gen in zip(flat, gens, strict=True)
            ]
            pooled = self._read_answers(batch)
            for k, ((i, g), b) in enumerate(zip(flat, batch, strict=True)):
                s, dfd = specs[i], defs[i][g]
                prow = {L: pooled[L][k : k + 1] for L in pool.layers}
                if dfd is not None:  # brief episode: adaptive monitor + progressive penalty
                    read = dfd.score(dfd.active, pooled, k)
                    obs = dfd.observe(read, pooled, k)
                    rew = obs["reward"] - (0.0 if b.finished else acfg.trunc_penalty)
                    b.info.update(
                        monitored=read,
                        passed=obs["passed"],
                        novel=obs["novel"],
                        concept=s.monitored,
                    )
                    if obs["passed"]:
                        passed_turn.append(t)
                        if t >= 1:
                            refit.add("pos", pooled, k, cap)  # escape -> refit positive
                    novel += int(obs["novel"])
                    shown, shown_cond = (read["score"], read["thr"]), s.cond
                else:  # selectivity: base concept probe, no brief
                    base = pool.concept_monitor(s.monitored) if s.text_concept else None
                    det = (
                        bool(
                            base.logits(prow[base.layer], [s.monitored])[0]
                            > base.thresholds([s.monitored])[0]
                        )
                        if base is not None
                        else False
                    )
                    rew = selectivity_reward(det, acfg) if s.text_concept else 0.0
                    b.info.update(monitored={"detected": det, "score": 0.0, "thr": 0.0})
                    # Dense selectivity target: match the base concept probe to base (nobrief keeps
                    # the concept detectable without a brief; neg_text keeps the FPR
                    # low). The concept is covered by a bank member (validated at sampler init).
                    b.info["match_concept"] = s.monitored
                    if s.cond == "nobrief" and s.text_concept:
                        refit.add("np", pooled, k, cap)  # concept text, no brief: out-of-ctx TPR
                    elif s.cond == "neg_text":
                        refit.add("neg", pooled, k, cap)  # concept-free under brief: refit negative
                    shown, shown_cond = None, s.cond
                trunc += int(not b.finished)
                b.reward = rew
                rewards[i, g, t] = rew
                samples[i][g].append(b)
                convs[i][g].append(
                    {
                        "role": "assistant",
                        "content": b.text,
                        "reasoning": b.info.get("reasoning_text", ""),
                    }
                )
                if t < T - 1:
                    nxt = s.text(t + 1) if len(s.texts) > 1 else None
                    convs[i][g].append(
                        {
                            "role": "user",
                            "content": render_feedback(
                                shown_cond,
                                s.monitored,
                                shown if shown is not None else (0.0, 1.0),
                                nxt,
                                feedback=s.feedback,
                                named=s.named,
                            ),
                        }
                    )
        n_brief = sum(d is not None for row in defs for d in row)
        metrics = {
            "pass_rate": len(passed_turn) / max(1, n_brief * T),
            "mean_pass_turn": (sum(passed_turn) / len(passed_turn))
            if passed_turn
            else float("nan"),
            "novel_escapes": novel,
            "trunc_frac": trunc / max(1, S * G * T),
            "bank_size": len(pool.entries),
        }
        return samples, rewards, metrics, refit

    def _read_answers(self, batch: list[Sample]) -> dict[int, torch.Tensor]:
        """No-grad pooled reads of every sample's answer span, at all pool layers."""

        reads = []
        for a in range(0, len(batch), self.cfg.micro_batch):
            chunk = batch[a : a + self.cfg.micro_batch]
            views = [View(b.view.ids[: b.probe_span[1]], b.probe_span[0]) for b in chunk]
            p, _ = self.policy.forward(views, self.pool.layers, grad=False)
            reads.append(p)
        return {L: torch.cat([r[L] for r in reads]) for L in self.pool.layers}

    # --------------------------------------------------------------------------- update
    def _token_logps(self, views: list[View], spans: list[tuple[int, int]]) -> torch.Tensor:
        """Log-probs of the tokens in spans (positions in view.ids), concatenated."""

        pol = self.policy
        ids, att, _, _ = pol._batch(views)
        n = ids.shape[1]
        # Logits only from the earliest needed position (min a - 1) to the end, and log_softmax
        # only at the needed positions: full-vocab logits for every position of a long multi-turn
        # history cost several GB per micro-batch at a 248k vocabulary.
        keep = n - max(0, min(a for a, _ in spans) - 1)
        out = pol.model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=keep)
        off = n - keep
        parts = []
        for r, (a, b) in enumerate(spans):
            pos = torch.arange(a, b, device=ids.device)
            lp = torch.log_softmax(out.logits[r, pos - 1 - off].float(), -1)
            parts.append(lp.gather(-1, ids[r, pos][:, None])[:, 0])
        return torch.cat(parts) if parts else torch.zeros(0, device=ids.device)

    def step(self) -> dict:
        cfg, pol = self.cfg, self.policy
        t0 = time.time()
        specs = self.sample_specs(cfg.specs_per_step)
        extra, refit = {}, None
        if cfg.use_adaptive:
            samples, rewards, extra, refit = self.rollout_adaptive(specs)
        else:
            samples, rewards = self.rollout(specs)
        adv = group_advantages(rewards, cfg.credit)
        items = [
            (samples[i][g][t], float(adv[i, g, t]))
            for i in range(len(specs))
            for g in range(cfg.group_size)
            for t in range(cfg.turns)
            if adv[i, g, t] != 0
            and samples[i][g][t].sampled[1] > samples[i][g][t].sampled[0]
            and (samples[i][g][t].finished or not cfg.grpo_require_finished)
        ]
        pol.model.train()
        self.opt.zero_grad(set_to_none=True)
        n_tok = max(1, sum(s.sampled[1] - s.sampled[0] for s, _ in items))
        stats = {"pg": 0.0, "kl": 0.0, "dg": 0.0}
        # Length-sort so each update chunk holds views of similar length: _token_logps keeps logits
        # from min(sampled start) - 1 to the end, so a chunk mixing a short turn-0 reasoning with a
        # long turn-4 history materialises [B, ~full length, 248k vocab] and OOMs (smoke 5215972).
        items = sorted(items, key=lambda it: len(it[0].view.ids))
        umb = cfg.update_micro_batch or cfg.micro_batch
        for a in range(0, len(items), umb):
            chunk = items[a : a + umb]
            views = [s.view for s, _ in chunk]
            spans = [s.sampled for s, _ in chunk]
            with torch.no_grad(), pol.reference():
                ref = self._token_logps(views, spans)
            logp = self._token_logps(views, spans)
            advt = torch.cat(
                [torch.full((s.sampled[1] - s.sampled[0],), av) for s, av in chunk]
            ).to(logp.device)
            loss, st = pg_loss(logp, ref, advt, cfg.beta_kl)
            (loss * len(logp) / n_tok).backward()
            for k in ("pg", "kl"):
                stats[k] += st[k] * len(logp) / n_tok
        # Direct-gradient (Arm A / Atlas Eq. 10): the NC loss through the answer activations, on
        # EVERY dodge turn >= hide_from_turn, NOT only the GRPO items. GRPO drops zero-variance
        # groups (every group when the reasoning is fully truncated, smoke 5217182), so gating the
        # direct term on GRPO items would make the backprop arm never fire -- the whole point of
        # this arm is a dense gradient that does not need reward variance.
        stats["mg"] = 0.0
        acd = cfg.adaptive
        if cfg.use_adaptive and (acd.direct_grad_weight or acd.direct_match_weight):
            stats["dg"], stats["mg"] = self._direct_grad_pass(samples)
        if items or stats["dg"] or stats["mg"]:
            torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
            self.opt.step()
        self.step_k += 1
        acf = cfg.adaptive
        if refit is not None and acf.refit_every and self.step_k % acf.refit_every == 0:
            extra.update(self._online_refit(refit))
        det = [
            samples[i][g][t].info["monitored"]["detected"]
            for i in range(len(specs))
            for g in range(cfg.group_size)
            for t in range(cfg.turns)
        ]
        rec = {
            "step": self.step_k,
            "reward_mean": float(rewards.mean()),
            "reward_by_turn": rewards.mean((0, 1)).tolist(),
            "detected": sum(det) / len(det),
            "finished": sum(s.finished for row in samples for gs in row for s in gs) / len(det),
            "n_items": len(items),
            "sec": round(time.time() - t0, 2),
            **stats,
            **extra,
        }
        with (self.out / "train_log.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def _direct_grad_pass(self, samples) -> tuple[float, float]:
        """Arm A / Atlas Eq. 10: dense gradients through the frozen probe on the answer activations
        (MSE form, which our NC work found beats a margin loss, 08-07c). Two kinds:
        - HIDE: dodge turns >= hide_from_turn -> push the shown monitor's p toward 0.
        - MATCH: nobrief / neg_text turns -> push the base concept probe's p toward the BASE model's
          p (selectivity). This binds even under truncation, so the model cannot collapse to hiding
          with no brief (the l3iter failure). Returns (mean hide p^2, mean match MSE)."""

        acfg = self.cfg.adaptive
        umb = self.cfg.update_micro_batch or self.cfg.micro_batch
        dg = self._dense_term(samples, "hide", acfg.direct_grad_weight, umb)
        mg = self._dense_term(samples, "match", acfg.direct_match_weight, umb)
        return dg, mg

    def _dense_term(self, samples, kind: str, weight: float, umb: int) -> float:
        """One dense answer-activation term (kind = 'hide' or 'match'); see _direct_grad_pass."""

        cfg, pol = self.cfg, self.policy
        picks = []
        for row in samples:
            for gs in row:
                for t, s in enumerate(gs):
                    if s.probe_span[1] <= s.probe_span[0]:
                        continue
                    if (kind == "hide" and t >= cfg.hide_from_turn and "name" in s.info.get(
                        "monitored", {}
                    )) or (kind == "match" and s.info.get("match_concept")):
                        picks.append(s)
        if not weight or not picks:
            return 0.0
        picks.sort(key=lambda s: s.probe_span[1])
        n, acc = len(picks), 0.0
        for a in range(0, n, umb):
            chunk = picks[a : a + umb]
            views = [View(s.view.ids[: s.probe_span[1]], s.probe_span[0]) for s in chunk]
            if kind == "match":
                with torch.no_grad(), pol.reference():
                    ref_pooled, _ = pol.forward(views, self.pool.layers, grad=False)
            pooled, _ = pol.forward(views, self.pool.layers, grad=True)
            terms = []
            for j, s in enumerate(chunk):
                if kind == "hide":
                    mon = self.pool.get(s.info["monitored"]["name"])
                    c = [s.info["concept"]]
                    lg = mon.logits(pooled[mon.layer][j : j + 1], c)
                    z = (lg - mon.thresholds(c)) / mon.scale(c)
                    terms.append(torch.sigmoid(z)[0] ** 2)
                else:
                    mon = self.pool.concept_monitor(s.info["match_concept"])
                    c = [s.info["match_concept"]]
                    scl = mon.scale(c)
                    z = (mon.logits(pooled[mon.layer][j : j + 1], c) - mon.thresholds(c)) / scl
                    zr = (mon.logits(ref_pooled[mon.layer][j : j + 1], c) - mon.thresholds(c)) / scl
                    terms.append((torch.sigmoid(z)[0] - torch.sigmoid(zr)[0].detach()) ** 2)
            loss = weight * torch.stack(terms).sum() / n
            loss.backward()
            acc += float(loss.detach())
        return acc

    def _online_refit(self, refit: RefitData) -> dict:
        """Fit a fresh probe on the policy's own escapes and, if it clears the bars, add it to the
        bank (the user's refit idea at training cadence; REPLICATION 2026-10-11a)."""

        if not refit.ready(self.cfg.adaptive.refit_min_each):
            np_, ng, nn = refit.counts()
            return {"refit": "not_enough_data", "refit_counts": [np_, ng, nn]}
        self.refit_k += 1
        concept = refit.concept
        name = f"refit_online_{self.refit_k}"
        heldout = {e.monitor.layer for e in self.pool.entries if e.held_out} - {
            e.monitor.layer for e in self.pool.entries if not e.held_out}
        bank, info = fit_refit_bank(refit, concept, name, self.cfg.adaptive,
                                    exclude_layers=heldout)
        if bank is None:
            return {"refit": "rejected", "refit_info": info}
        self.pool.add(bank.to(self.policy.device), round=self.step_k, family="concept_refit")
        dropped = trim_bank(self.pool, self.base_names, self.cfg.adaptive.refit_max_bank)
        torch.save(bank.state(), self.out / "bank" / f"{name}.pt")
        return {
            "refit": "added",
            "refit_layer": info["layer"],
            "refit_auroc": info["auroc"],
            "refit_tpr_np": info["tpr_np"],
            "refit_dropped": dropped,
            "bank_size": len(self.pool.entries),
        }

    def train(self) -> None:
        while self.step_k < self.cfg.steps:
            rec = self.step()
            if self.step_k % 10 == 0 or self.step_k == 1:
                print(json.dumps(rec), flush=True)
            if self.cfg.save_every and self.step_k % self.cfg.save_every == 0:
                self.policy.save_adapter(self.out / f"step_{self.step_k}")
        self.policy.save_adapter(self.out / "final")
