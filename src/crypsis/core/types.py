# ruff: noqa: E501
"""Plain data types shared by every layer: concepts, texts, conversations, token views, readouts, verdicts.

Supersets of what the tica code passes around today (dict messages, policy.View / MultiView, the
score_readouts dicts, AdaptiveDefender.observe dicts) so later phases can adopt them without changing rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def text_id(text: str) -> str:
    """Content hash of a text; keys the elicit-inputs, neg-label and data files (== tica text_id)."""

    return "ax-" + hashlib.sha256(text.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Concept:
    """A monitored concept: name plus optional family and split (train / dev / heldout / ood)."""

    name: str
    family: str | None = None
    split: str | None = None


@dataclass
class Text:
    """One text, optionally about a concept, with its data split (fit / calib / episode) and metadata."""

    text: str
    concept: str | None = None
    split: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return text_id(self.text)


@dataclass
class Turn:
    """One chat message (role: system / user / assistant)."""

    role: str
    content: str

    def to_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_message(cls, m: dict[str, str]) -> Turn:
        return cls(m["role"], m["content"])


@dataclass
class Conversation:
    """An ordered list of turns; converts to / from the list-of-dict messages the tica code uses."""

    turns: list[Turn] = field(default_factory=list)

    def messages(self) -> list[dict[str, str]]:
        return [t.to_message() for t in self.turns]

    @classmethod
    def from_messages(cls, msgs: list[dict[str, str]]) -> Conversation:
        return cls([Turn.from_message(m) for m in msgs])

    def append(self, role: str, content: str) -> Conversation:
        self.turns.append(Turn(role, content))
        return self

    def __len__(self) -> int:
        return len(self.turns)


@dataclass
class View:
    """Token ids of a rendered conversation and the probed span: content starts at `start`, the last
    `n_tail` ids are end-of-turn tokens (in the NLL, not in the probe read). == tica policy.View."""

    ids: list[int]
    start: int
    n_tail: int = 0

    @property
    def end(self) -> int:
        return len(self.ids) - self.n_tail

    @property
    def n_content(self) -> int:
        return self.end - self.start


@dataclass
class MultiView:
    """A conversation tokenised once with every assistant turn's (content start, content end, tail end);
    turn t's per-turn view is ids[:spans[t][2]]. == tica policy.MultiView."""

    ids: list[int]
    spans: list[tuple[int, int, int]]

    def view(self, t: int) -> View:
        s, e, te = self.spans[t]
        return View(self.ids[:te], s, te - e)


@dataclass
class Readout:
    """One monitor's reading of one turn (keys of tica episodes.score_readouts rows)."""

    raw: float
    score: float
    thr: float
    p: float
    detected: bool
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"raw": self.raw, "score": self.score, "thr": self.thr, "p": self.p,
                             "detected": self.detected}
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Readout:
        return cls(float(d["raw"]), float(d["score"]), float(d["thr"]), float(d["p"]),
                   bool(d["detected"]), d.get("name"))


@dataclass
class Verdict:
    """The defender's decision on a turn (keys of tica AdaptiveDefender.observe's result)."""

    passed: bool
    novel: bool = False
    reward: float | None = None
    next_monitor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"reward": self.reward, "passed": self.passed, "novel": self.novel,
                "next_monitor": self.next_monitor}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Verdict:
        return cls(bool(d["passed"]), bool(d.get("novel", False)), d.get("reward"),
                   d.get("next_monitor"))
