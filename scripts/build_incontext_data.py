"""Build the Level-0 in-context evasion battery from public sources (PREREGISTRATION-ICA §2).

Neutral sentences and concepts are Kowalski et al.'s released battery
(github.com/mkobalski/activation-control, pinned commit). Concept-bearing sentences
come from English Wikipedia articles, with the revision id and a text hash recorded.
Selection is deterministic.

Run on the login node (needs network):
    uv run python scripts/build_incontext_data.py \
        --kowalski-repo /scratch/sneheel/external/activation-control \
        --output $SCRATCH/next_chameleons_raw_cache/incontext/l0_battery.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

KOWALSKI_COMMIT = "4189f93"
CONCEPTS = [
    "Denim", "Lightning", "Volcanoes", "Bread", "Trees",
    "Cameras", "Vegetables", "Milk", "Dynasties", "Deserts",
]
# Articles are concatenated in this order and the first PER_CONCEPT kept sentences are used.
# The supplements (after the first title) were fixed after a yield check on 2026-09-27, before
# any model run: Denim, Vegetable and Dynasty alone yield fewer than 40 qualifying sentences.
ARTICLES = {
    "Denim": ["Denim", "Jeans"], "Lightning": ["Lightning"], "Volcanoes": ["Volcano"],
    "Bread": ["Bread"], "Trees": ["Tree"], "Cameras": ["Camera"],
    "Vegetables": ["Vegetable", "Vegetable farming"], "Milk": ["Milk"],
    "Dynasties": ["Dynasty", "Dynasties of China", "Egyptian dynasties"], "Deserts": ["Desert"],
}
LEMMA = {
    "Denim": r"denims?", "Lightning": r"lightnings?", "Volcanoes": r"volcano(?:e?s)?",
    "Bread": r"breads?", "Trees": r"trees?", "Cameras": r"cameras?",
    "Vegetables": r"vegetables?", "Milk": r"milks?", "Dynasties": r"dynast(?:y|ies)",
    "Deserts": r"deserts?",
}
MIN_WORDS, MAX_WORDS, PER_CONCEPT = 7, 22, 40
BANNED_CHARS = set('"\'()[]{};:')
PRONOUN_STARTS = ("It ", "They ", "This ", "These ", "He ", "She ", "Its ", "Their ")
USER_AGENT = "next-chameleons-research/0.1 (academic; incontext battery builder)"


def read_kowalski(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(s\d+):\s*(.+)$", line.strip())
        if m:
            rows.append({"id": m.group(1), "text": m.group(2).strip()})
    return rows


def fetch_extract(title: str) -> dict:
    params = {
        "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
        "prop": "extracts|revisions", "explaintext": "1", "rvprop": "ids|timestamp",
        "titles": title,
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        time.sleep(2.0 * (2**attempt) if attempt else 2.0)  # polite pacing; back off on 429
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                page = json.load(resp)["query"]["pages"][0]
            break
        except urllib.error.HTTPError as err:
            if err.code != 429 or attempt == 5:
                raise
    text = page["extract"]
    rev = page["revisions"][0]
    return {
        "title": page["title"], "revid": rev["revid"], "timestamp": rev["timestamp"],
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "text": text,
    }


def split_sentences(text: str) -> list[str]:
    out = []
    for para in text.splitlines():
        para = para.strip()
        if not para or para.startswith("="):
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", para) if s.strip())
    return out


def keep(sentence: str, lemma: str) -> bool:
    words = sentence.split()
    return (
        MIN_WORDS <= len(words) <= MAX_WORDS
        and sentence.isascii()
        and not (BANNED_CHARS & set(sentence))
        and sentence[-1] in ".!?"
        and not sentence.startswith(PRONOUN_STARTS)
        and re.search(rf"\b{lemma}\b", sentence, flags=re.IGNORECASE) is not None
    )


def select(sentences: list[str], lemma: str) -> list[str]:
    seen, kept = set(), []
    for s in sentences:
        key = s.lower()
        if key in seen or not keep(s, lemma):
            continue
        seen.add(key)
        kept.append(s)
        if len(kept) == PER_CONCEPT:
            break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kowalski-repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    head = subprocess.run(
        ["git", "-C", str(args.kowalski_repo), "rev-parse", "--short", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not head.startswith(KOWALSKI_COMMIT):
        raise SystemExit(f"Kowalski repo at {head}, expected {KOWALSKI_COMMIT}")
    primary = read_kowalski(args.kowalski_repo / "sentences.txt")
    spare = read_kowalski(args.kowalski_repo / "extra_sentences.txt")
    assert [r["id"] for r in primary] == [f"s{i}" for i in range(50)], "sentences.txt changed"

    wiki, concept_sentences = {}, {}
    for concept in CONCEPTS:
        pages, sentences = [], []
        for title in ARTICLES[concept]:
            page = fetch_extract(title)
            sentences.extend(split_sentences(page.pop("text")))
            pages.append(page)
        kept = select(sentences, LEMMA[concept])
        wiki[concept] = pages
        rows = [{"id": f"{concept}-w{i}", "text": s} for i, s in enumerate(kept)]
        concept_sentences[concept] = {"train": rows[0::2], "test": rows[1::2]}
        print(f"{concept:11s} revs {[p['revid'] for p in pages]}  kept {len(kept)}")

    payload = {
        "meta": {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "kowalski_repo": "github.com/mkobalski/activation-control",
            "kowalski_commit": head,
            "wikipedia": wiki,
            "filters": {
                "min_words": MIN_WORDS, "max_words": MAX_WORDS, "per_concept": PER_CONCEPT,
                "banned_chars": "".join(sorted(BANNED_CHARS)), "pronoun_starts": PRONOUN_STARTS,
                "lemma": LEMMA,
            },
        },
        "concepts": CONCEPTS,
        "neutral": {"primary": primary, "heldout": spare},
        "concept_sentences": concept_sentences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
