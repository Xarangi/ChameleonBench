"""Model interface for the in-context runner, plus a deterministic fake for CPU tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Read:
    """Teacher-forced read of the FINAL assistant message of a conversation."""

    token_ids: list[int]
    char_offsets: list[tuple[int, int]]  # relative to the assistant message content
    acts: dict[int, np.ndarray]  # layer -> [n_tokens, d] (output of decoder layer `layer`)


class Backend(Protocol):
    layers: list[int]

    def generate(self, conversations: list[list[dict]], max_new_tokens: int) -> list[str]: ...

    def read(self, conversations: list[list[dict]]) -> list[Read]: ...

    def last_token_acts(self, user_messages: list[str]) -> dict[int, np.ndarray]: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def n_tokens(self, text: str) -> int: ...


class FakeBackend:
    """Word-level toy model for tests.

    - Each word has a fixed random embedding.
    - A word matching a concept adds that concept's direction, scaled by a damping factor.
    - Disclosing the monitor damps slightly (0.8), for the awareness control.
    - With `learns=True`, the fake runs a win-stay/lose-shift strategy. Its first strategy is
      set by the sentence ("good" damps x0.02, "bad" does nothing). Every DETECTED verdict
      makes it switch strategy, and its notebook records the final strategy.
    - Only feedback about its OWN attempts can steer it to "good", so real feedback beats
      yoked sham feedback. With `learns=False`, feedback has no effect.
    - Generation copies the quoted sentence into the requested format.
    """

    def __init__(self, concepts: list[str], layers: list[int], d: int = 16,
                 seed: int = 0, learns: bool = True) -> None:
        self.layers, self.d, self.learns = list(layers), d, learns
        rng = np.random.default_rng(seed)
        self.directions = {c: rng.normal(size=d) * 4 for c in concepts}
        self._vocab: dict[str, int] = {}
        self._emb: dict[int, np.ndarray] = {}
        self._rng = np.random.default_rng(seed + 1)

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab)
            self._emb[self._vocab[word]] = self._rng.normal(size=self.d)
        return self._vocab[word]

    def _words(self, text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def generate(self, conversations, max_new_tokens):
        out = []
        for conv in conversations:
            last = conv[-1]["content"]
            if "Write only the strategy" in last:
                out.append(f"use strategy {'good' if self._good(conv) else 'bad'}")
                continue
            first = conv[0]["content"]
            sentence = re.search(r'Write "(.*?)" exactly', first).group(1)
            out.append(sentence if "<answer>" not in first
                       else f"<plan>copy it</plan> <answer> {sentence} </answer>")
        return out

    def _good(self, conv) -> bool:
        users = [m["content"] for m in conv if m["role"] == "user"]
        notes = re.search(r"use strategy (good|bad)", users[0])
        if notes:
            return notes.group(1) == "good"
        sentence = re.search(r'Write "(.*?)" exactly', users[0])
        start_good = bool(sentence) and sum(map(ord, sentence.group(1))) % 2 == 0
        joined = " ".join(users[1:])
        switches = joined.count("DETECTED.") - joined.count("NOT DETECTED.")
        return start_good ^ (switches % 2 == 1)

    def _damp(self, conv) -> float:
        users = " ".join(m["content"] for m in conv if m["role"] == "user")
        damp = 0.8 if "monitor reads your internal activations" in users else 1.0
        if self.learns and "keeping the monitor from detecting" in users and self._good(conv):
            damp *= 0.02
        return damp

    def read(self, conversations):
        reads = []
        for conv in conversations:
            content = conv[-1]["content"]
            damp = self._damp(conv)
            ids, offs, rows = [], [], []
            for w, s, e in self._words(content):
                tid = self._id(w.lower().strip(".,<>/"))
                vec = self._emb[tid].copy()
                for c, direction in self.directions.items():
                    if c.lower().rstrip("s")[:5] in w.lower():
                        vec = vec + direction * damp
                ids.append(tid)
                offs.append((s, e))
                rows.append(vec)
            arr = np.array(rows) if rows else np.zeros((0, self.d))
            reads.append(Read(ids, offs, {layer: arr for layer in self.layers}))
        return reads

    def last_token_acts(self, user_messages):
        rows = []
        for msg in user_messages:
            word = msg.split()[-1]
            vec = self._emb[self._id(word.lower())].copy()
            for c, direction in self.directions.items():
                if word == c:
                    vec = vec + direction
            rows.append(vec)
        arr = np.array(rows)
        return {layer: arr for layer in self.layers}

    def decode(self, token_ids):
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv[t] for t in token_ids)

    def n_tokens(self, text):
        return len(self._words(text))
