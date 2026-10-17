"""Build the held-out concept-text split for arms-race evaluation.

The official preset (the attacker's training data and the probe-fit source) holds only
55-72 topical examples per concept, so every earlier scoreboard re-used probe-fit texts.
This script carves a split the attacker never trains on and the defender never fits on:
concept responses from the authors' full 21-split matrix that never appear, verbatim or as
a near-duplicate (word 5-gram Jaccard > 0.5), as any response in the preset.

Each surviving text is assigned deterministically, by content hash, to ``val`` (used by
the in-training evasion gate) or ``test`` (used only by the scoreboard). The output lives
in the controlled raw cache and is never committed; only the counts are logged.

Usage: uv run python scripts/build_heldout_split.py [--raw-cache DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from next_chameleons.hf.paper_data import (
    _normalize_preset_probe_name,
    find_full_matrix_dataset,
    find_preset_dataset,
)

CONCEPTS = ("all-caps", "finnish", "biology-focused", "html", "mathematical")
NEAR_DUP_JACCARD = 0.5


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def build(raw_cache: Path) -> list[dict[str, str]]:
    preset = json.loads(find_preset_dataset(raw_cache).read_text(encoding="utf-8"))
    full = json.loads(find_full_matrix_dataset(raw_cache).read_text(encoding="utf-8"))
    preset_responses = {str(r["response"]).strip() for r in preset}
    preset_shingles = [_shingles(t) for t in preset_responses]
    rows: dict[str, dict[str, str]] = {}
    for record in full:
        if isinstance(record["probe_name"], list):
            continue
        concept = _normalize_preset_probe_name(str(record["probe_name"]))
        if concept not in CONCEPTS or "data_X" not in str(record.get("data_type", "")):
            continue
        response = str(record["response"]).strip()
        if not response or response in preset_responses or response in rows:
            continue
        shingles = _shingles(response)
        if any(
            len(shingles & other) / max(1, len(shingles | other)) > NEAR_DUP_JACCARD
            for other in preset_shingles
        ):
            continue
        digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
        # Keep the untriggered prompt so generation validation can re-ask it.
        prompt = re.sub(r"^You are being probed for '[^']*'\.\s*", "", str(record["prompt"]))
        rows[response] = {
            "concept": concept,
            "prompt": prompt.strip(),
            "response": response,
            "split": "val" if int(digest[:8], 16) % 2 == 0 else "test",
            "text_hash": digest,
            "source": "neural_chameleons_full_matrix_not_in_preset",
            "full_matrix_index": str(record["index"]),
        }
    return sorted(rows.values(), key=lambda r: (r["concept"], r["text_hash"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-cache", type=Path, default=Path("/scratch/sneheel/next_chameleons_raw_cache")
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.raw_cache / "paper_benign_heldout" / "split_v1.json"
    rows = build(args.raw_cache)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    counts = collections.Counter((r["concept"], r["split"]) for r in rows)
    for concept in CONCEPTS:
        val, test = counts[(concept, "val")], counts[(concept, "test")]
        print(f"{concept:16s} val={val:4d} test={test:4d}")
    print(f"wrote {out} ({len(rows)} texts)")


if __name__ == "__main__":
    main()
