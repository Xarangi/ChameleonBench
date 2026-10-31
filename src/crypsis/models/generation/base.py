"""The Generator protocol: conversations (or prompt token ids) in, completions out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

Prompt = list[dict] | list[int]  # a conversation ending in a user message, or prompt token ids


@dataclass
class Generation:
    """One completion. ids exclude the end token; stop = the id that ended it (None = budget
    reached or unknown); finished = an end token was produced."""

    text: str
    prompt_ids: list[int]
    ids: list[int]
    finished: bool
    stop: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class Generator(Protocol):
    def generate(self, prompts: Sequence[Prompt], max_new_tokens: int, *,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 0,
                 stop_ids: set[int] | None = None) -> list[Generation]:
        """temperature 0 = greedy."""
        ...


def prompt_ids(policy, prompt: Prompt) -> list[int]:
    """Token ids of a prompt: a conversation is rendered with the policy's chat (generation
    prompt); a list of ints is taken as is."""

    if prompt and isinstance(prompt[0], dict):
        return policy.chat.prompt_ids(prompt)
    return list(prompt)
