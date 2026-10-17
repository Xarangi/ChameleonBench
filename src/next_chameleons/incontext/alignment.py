"""Sentence-span alignment and compliance, vendored from Kowalski et al.

Source: github.com/mkobalski/activation-control @4189f93, src/analysis/alignment.py and
src/utils/compliance.py (MIT License, (c) 2026 Kowalski, Fonseca Rivera, Macar, Africa).
The logic is unchanged. Their "normalized_levenshtein" is difflib's SequenceMatcher ratio,
not edit distance, and we keep that exact semantics.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher

COMPLIANCE_THRESHOLD = 0.85


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def check_compliance(generated: str, target: str, threshold: float = COMPLIANCE_THRESHOLD):
    """(passed, score): Kowalski's `normalized_levenshtein` method."""

    score = similarity(generated, target)
    return score >= threshold, score


def exact_match(generated: str, target: str) -> bool:
    def clean(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s).lower().strip())

    return clean(generated) == clean(target)


def align_sentence_span(
    decode: Callable[[Sequence[int]], str],
    token_ids: Sequence[int],
    target_sentence: str,
    n_sentence_tokens: int,
) -> tuple[int, int, float]:
    """[start, end) window of `n_sentence_tokens` tokens whose decoded text best matches."""

    n_rec = len(token_ids)
    if n_rec == 0:
        return 0, 0, 0.0
    span_len = min(n_sentence_tokens, n_rec)
    best = (0, span_len, -1.0)
    for start in range(0, n_rec - span_len + 1):
        sim = similarity(decode(token_ids[start:start + span_len]), target_sentence)
        if sim > best[2]:
            best = (start, start + span_len, sim)
    return best
