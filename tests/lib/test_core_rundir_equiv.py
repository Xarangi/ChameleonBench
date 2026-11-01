"""core.rundir == the run-directory parsers it replaces, on synthetic run trees."""

from __future__ import annotations

import json
import re
from pathlib import Path

from crypsis.core.rundir import RunDir


def heredoc_newest(out: Path):
    """Verbatim selection rule of the scripts/slurm/tica_resume.sh heredoc."""

    best = (0, False, "")
    st = lambda d: Path(d, "trainer_state.pt").exists()  # noqa: E731
    for d in out.glob("adaptive/step_*"):
        best = max(best, (int(d.name[5:]), st(d), str(d)))
    for d in out.glob("resume_*/adaptive/step_*"):
        off = int(re.match(r"resume_(\d+)", d.parent.parent.name).group(1))
        best = max(best, (off + int(d.name[5:]), st(d), str(d)))
    for fin in [out / "adaptive" / "final", *out.glob("resume_*/adaptive/final")]:
        log = fin.parent / "train_log.jsonl"
        if fin.is_dir() and log.exists() and log.read_text().strip():
            m = re.match(r"resume_(\d+)", fin.parent.parent.name)
            last = int(log.read_text().strip().splitlines()[-1].split('"step": ')[1].split(",")[0])
            best = max(best, ((int(m.group(1)) if m else 0) + last, st(fin), str(fin)))
    return best[0], best[2]


def v4c_newest(out: Path):
    """Verbatim rule of scripts/slurm/tamia_prod_v4c_switch.sh."""

    best = (0, "")
    for d in out.glob("adaptive/step_*"):
        best = max(best, (int(d.name[5:]), str(d)))
    for d in out.glob("resume_*/adaptive/step_*"):
        off = int(re.match(r"resume_(\d+)", d.parent.parent.name).group(1))
        best = max(best, (off + int(d.name[5:]), str(d)))
    return best


def _ck(p: Path, state=False):
    p.mkdir(parents=True, exist_ok=True)
    (p / "adapter_model.safetensors").write_text("")
    if state:
        (p / "trainer_state.pt").write_text("")


def _log(d: Path, steps):
    d.mkdir(parents=True, exist_ok=True)
    (d / "train_log.jsonl").write_text("".join(json.dumps({"step": s, "loss": 0.1}) + "\n"
                                               for s in steps))


def _trees(tmp: Path):
    # 1: first chunk + two resumes; newest is a state-less copy tied with a state copy
    a = tmp / "a"
    _ck(a / "adaptive/step_100"), _ck(a / "adaptive/step_200", True)
    _log(a / "adaptive", range(1, 211))
    _ck(a / "resume_200/adaptive/step_60")
    _log(a / "resume_200/adaptive", range(1, 81))
    _ck(a / "resume_260/adaptive/step_20", True)
    _ck(a / "resume_260/adaptive/step_40")
    _ck(a / "resume_280/adaptive/step_20", True)  # == global 300, same as resume_260/step_40
    (a / "resume_260_k2_aborted/adaptive").mkdir(parents=True)  # no checkpoints (as on disk)
    _log(a / "resume_260_k2_aborted/adaptive", range(1, 4))
    # 2: a chunk that ran to its end (final newer than any step dir)
    b = tmp / "b"
    _ck(b / "adaptive/step_50", True)
    _ck(b / "resume_50/adaptive/step_20")
    _ck(b / "resume_50/adaptive/final")
    _log(b / "resume_50/adaptive", range(1, 31))
    # 3: branch with symlinked checkpoints into another run
    other = tmp / "other"
    _ck(other / "resume_1250/adaptive/step_20", True)
    c = tmp / "c"
    _ck(c / "adaptive/step_1200")
    (c / "resume_1250/adaptive").mkdir(parents=True)
    (c / "resume_1250/adaptive/step_20").symlink_to(other / "resume_1250/adaptive/step_20")
    _ck(c / "resume_1250/adaptive/step_10")
    return [a, b, c]


def test_newest_matches_bash_heredocs(tmp_path):
    for out in _trees(tmp_path):
        rd = RunDir(out)
        ck = rd.newest()
        assert (ck.global_step, str(ck.path)) == heredoc_newest(out), out
        ck2 = rd.newest(prefer_state=False, include_final=False)
        assert (ck2.global_step, str(ck2.path)) == v4c_newest(out), out
        assert rd.global_step(ck.path) == ck.global_step
    a = RunDir(tmp_path / "a")
    assert a.newest().path == tmp_path / "a/resume_280/adaptive/step_20"
    assert a.global_step("resume_260/adaptive/step_40") == 300
    assert RunDir(tmp_path / "b").newest().path.name == "final"
    assert RunDir(tmp_path / "c").checkpoint_at(1270).name == "step_20"
    assert RunDir(tmp_path / "x", base_offset=360).global_step("adaptive/step_5") == 365


def test_chunks_and_rows_match_run_report(tmp_path):
    from next_chameleons.tica.run_report import chunks, load_run

    for out in _trees(tmp_path):
        for base in (0, 7):
            rd = RunDir(out, base)
            assert rd.chunks() == chunks(out, base)
            assert rd.rows("train_log.jsonl") == load_run(out, base)["train"]
            assert [o for o, _ in rd.train_logs()] == [o for o, _ in chunks(out, base)]
