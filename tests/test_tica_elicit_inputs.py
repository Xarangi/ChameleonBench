"""Concept-eliciting requests (scripts/build_elicit_inputs.py) and the probe-cache request stamp."""

import importlib.util
import json
from pathlib import Path

from next_chameleons.tica.answer_format import LEGACY_INPUTS, _inputs_stamp, prompts

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bei", ROOT / "scripts" / "build_elicit_inputs.py")
bei = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bei)

ENC = "Write a short encyclopedia entry."


def _data():
    def c(name, ids):
        return {"concept": name, "split": "train", "fit": [{"id": ids[0], "text": "a"}],
                "calib": [], "episode": [{"id": i, "text": "b"} for i in ids[1:]]}
    return {"source": {"request": ENC}, "negatives": {},
            "concepts": [c("sports", ["s1", "s2"]), c("emotion: sad", ["e1", "e2"]),
                         c("German language", ["g1", "g2"]), c("casual tone", ["t1", "t2"]),
                         c("all-caps text", ["a1", "a2"]), c("Python code", ["p1", "p2"]),
                         c("mathematical reasoning", ["m1", "m2"]),
                         c("deception: roleplaying", ["d1", "d2"]),
                         c("jokes and humor", ["j1", "j2"])]}


def test_only_prompt_less_concepts_get_new_requests():
    inputs = {i: ENC for i in ("s1", "s2")}
    inputs.update({i: "Write a story." for i in ("e1", "e2")})
    inputs.update({i: f"REQ-{i}" for i in ("g1", "g2", "t1", "t2", "a1", "a2", "p1", "p2",
                                           "m1", "m2", "d1", "d2", "j1", "j2")})
    inputs["neg1"] = "untouched"
    rows, meta = bei.plan(_data(), inputs)
    assert {r["id"] for r in rows if r["new"] is None} == {"s1", "s2", "e1", "e2"}  # LLM-written
    for r in rows:  # no LLM: template fallbacks
        if r["new"] is None:
            r["new"] = bei.fallback(r)
    out = bei.finish(inputs, rows, meta, None)
    topic = 'Write a short encyclopedia entry about a subject related to "sports".'
    assert out["s1"] == out["s2"] == topic
    assert out["e2"].startswith("Write a story.") and "feels sad" in out["e2"]
    assert out["g1"] == "REQ-g1\n\nRespond in German."
    assert out["t2"] == "REQ-t2\n\nRespond in a casual tone."
    assert out["j1"] == "REQ-j1\n\nRespond in a humorous, joking tone."  # jokes = a tone
    assert out["a1"] == "REQ-a1\n\nWrite your entire response in capital letters."
    assert out["p1"] == "REQ-p1\n\nAnswer with Python code."
    for i in ("m1", "d2"):  # domain, safety keep their original prompt
        assert out[i] == f"REQ-{i}"
    assert out["neg1"] == "untouched"
    rules = {c: m["rule"] for c, m in meta["concepts"].items()}
    assert rules["sports"] == "topic" and rules["emotion: sad"] == "emotion"
    assert rules["German language"] == "language" and rules["jokes and humor"] == "tone"
    assert rules["all-caps text"] == rules["Python code"] == "format"
    assert {rules[c] for c in ("mathematical reasoning", "deception: roleplaying")} == {"original"}
    assert meta["concepts"]["sports"]["rewritten"] == 2


def test_clean_request():
    assert bei.clean_request('Request: "Write an entry about X, a Swedish swimmer."\nmore') == (
        "Write an entry about X, a Swedish swimmer.")
    assert bei.clean_request("  ") is None and bei.clean_request("short") is None


def test_probe_cache_key_changes_only_for_new_request_files(tmp_path):
    assert "final_v5_inputs.json" in LEGACY_INPUTS
    legacy = tmp_path / "final_v5_inputs.json"
    new = tmp_path / "final_v5_elicit_inputs.json"
    for p in (legacy, new):
        p.write_text(json.dumps({"x": "y"}))
    assert _inputs_stamp(prompts("answer", str(legacy), "tagged")[0]) == {}
    st = _inputs_stamp(prompts("answer", str(new), "tagged")[0])
    assert st["inputs"] == "final_v5_elicit_inputs.json" and st["inputs_size"] > 0
    assert _inputs_stamp(prompts("repeat")[0]) == {}
