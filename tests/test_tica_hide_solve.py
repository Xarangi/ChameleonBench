"""tica_hide_solve.summarize: accuracy among undetected / detected solutions per layer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tica_hide_solve import summarize


def test_summarize():
    correct = [True, True, False, False]
    r = summarize(correct, {8: [False, True, False, True], "any_train": [True] * 4})
    L = r["by_layer"]["8"]
    assert r["accuracy"] == 0.5 and L["detected"] == 0.5
    assert L["acc_undetected"] == 0.5 and L["acc_detected"] == 0.5
    assert L["solved_and_undetected"] == 0.25 and L["n_undetected"] == 2
    assert r["by_layer"]["any_train"]["acc_undetected"] is None
