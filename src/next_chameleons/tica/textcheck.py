"""Does a generated reply repeat the requested text exactly? (Level 1 generation check.)

Teacher forcing makes the forced read equal to the on-policy read only while the model really
repeats the text token for token. So every evaluation of a trained policy measures its greedy
exact-repeat rate (review 10-03). Same categories as scripts/l1_base_test.py.
"""

from __future__ import annotations

from collections import Counter

REFUSAL = ("i cannot", "i can't", "i can not", "i'm sorry", "i am sorry", "i won't", "i will not")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _norm(t: str) -> str:
    return " ".join(t.translate(_QUOTES).split())


def classify(gen: str, text: str) -> str:
    """exact | tags (exact once <text> tags are removed) | extra (text plus other content) |
    refusal | edited. Whitespace and curly quotes are normalized (Qwen3.5 straightens quotes)."""

    g, t = _norm(gen), _norm(text)
    if g == t:
        return "exact"
    if _norm(gen.replace("<text>", " ").replace("</text>", " ")) == t:
        return "tags"
    if t in g:
        return "extra"
    if g.lower().startswith(REFUSAL):
        return "refusal"
    return "edited"


def rates(cats: list[str]) -> dict[str, float]:
    return {c: m / len(cats) for c, m in Counter(cats).items()} if cats else {}
