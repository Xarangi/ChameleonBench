"""Interactive race launcher fixes (11-03, race mbpp_pass64 / 664118 lost 7 of 16 pieces silently):
vLLM stall watchdog (restart + retry), race_node.sh retries / summary / non-zero exit, the report
skipping unfinished pieces, and the launcher's piece plan (template flags, refit-on-pass only)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- vLLM stall watchdog
def _fake_server(hang_first: float):
    """/v1/completions server: the FIRST request hangs `hang_first` s; answers echo the prompt."""

    calls = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(len(body["prompt"]))
            if len(calls) == 1:
                time.sleep(hang_first)
            ch = [{"index": i, "text": f"a{p[0]}", "token_ids": p, "finish_reason": "stop"}
                  for i, p in reversed(list(enumerate(body["prompt"])))]
            data = json.dumps({"choices": ch}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:  # the client gave up on the hung request
                pass

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


def _bare_generator(url: str):
    from next_chameleons.tica.vllm_gen import VLLMGenerator

    g = object.__new__(VLLMGenerator)
    g.url, g.model, g.mode, g.loaded, g.lora = url, "m", "merged", "base", None
    g.call_timeout, g.retries = 0.5, 2
    g.stats = {"calls": 0, "prompts": 0, "gen_tokens": 0, "sec_gen": 0.0, "restarts": 0}
    return g


def test_watchdog_restarts_and_retries_in_chunks(monkeypatch):
    srv, calls = _fake_server(hang_first=3.0)
    monkeypatch.setenv("TICA_VLLM_RETRY_CHUNK", "2")
    g = _bare_generator(f"http://127.0.0.1:{srv.server_address[1]}")
    restarts = []
    g.restart = lambda: restarts.append(1)
    out = g.generate([[1], [2], [3], [4], [5]], 8, temperature=1.0, seed=3)
    srv.shutdown()
    assert restarts == [1]  # one stall -> one restart
    assert calls == [5, 2, 2, 1]  # the hung full batch, then retried in chunks of 2
    assert [o["text"] for o in out] == ["a1", "a2", "a3", "a4", "a5"]  # prompt order kept


def test_watchdog_gives_up_after_retries(monkeypatch):
    from next_chameleons.tica.vllm_gen import VLLMGenerator

    g = _bare_generator("http://127.0.0.1:9")
    n = []
    g.restart = lambda: n.append(1)

    def stall(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(VLLMGenerator, "_post", lambda self, *a, **k: stall())
    with pytest.raises(TimeoutError):
        g.generate([[1]], 4, temperature=0.0)
    assert len(n) == 2  # retries = 2 restarts, then the error surfaces (piece fails, node retries)


def test_restart_restores_served_weights(tmp_path):
    g = _bare_generator("http://x")
    g.local = tmp_path / "vllm_merged"
    g.log = open(tmp_path / "vllm.log", "w")  # noqa: SIM115
    seen = []
    g._kill = lambda: seen.append("kill")
    g._start = lambda: seen.append("start")
    g._reload = lambda path: seen.append(("reload", path))
    g.loaded = ("policy", 7)
    g.restart()
    assert seen == ["kill", "start", ("reload", str(g.local))]
    g.loaded = "base"
    seen.clear()
    g.restart()
    assert seen == ["kill", "start"]  # a fresh server already holds the base weights
    assert g.stats["restarts"] == 2
    g.log.close()


# ---------------------------------------------------------------- race_node.sh
def test_race_node_retries_reports_and_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    ok, flaky, dead = out / "ok.json", out / "flaky.json", out / "dead.json"
    cnt = tmp_path / "flaky_count"
    lines = [
        (ok, f"echo done > {ok}"),
        # fails on its first attempt (like a vLLM stall), succeeds on the retry
        (flaky, f"n=$(cat {cnt} 2>/dev/null || echo 0); echo $((n+1)) > {cnt}; "
                f"[ $n -ge 1 ] && echo done > {flaky} "
                "|| { echo 'TimeoutError: timed out'; exit 1; }"),
        (dead, "echo 'cp: cannot create regular file x: Permission denied'; exit 1"),
    ]
    (out / "dead_work").mkdir()
    (out / "ok_work").mkdir()
    tsv = out / "pieces.tsv"
    tsv.write_text("".join(f"{o}\t{c}\n" for o, c in lines))
    env = {**os.environ, "NG": "2", "RACE_POLL": "0.2", "RACE_RETRIES": "1",
           "TICA_REPO": str(repo), "TICA_VLLM_PY": "/bin/true", "SLURM_TMPDIR": str(tmp_path),
           "SLURM_JOB_ID": "t1", "SLURM_JOB_NAME": "race-test", "PATH": "/usr/bin:/bin"}
    r = subprocess.run(["bash", str(ROOT / "scripts/slurm/race_node.sh"), str(tsv)], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, r.stdout + r.stderr  # a missing piece fails the job
    assert "missing: 1" in r.stdout and f"MISSING {dead}" in r.stdout
    assert "Permission denied" in r.stdout  # the last error line of the piece log
    status = [ln.split("\t") for ln in (out / "pieces.tsv.status").read_text().splitlines()]
    assert [(s[0], s[1], s[2]) for s in status] == [
        (str(ok), "ok", "1"), (str(flaky), "ok", "2"), (str(dead), "missing", "2")]
    assert not (out / "ok_work").exists()  # per-piece scratch removed after success
    assert not (tmp_path / "race_t1").exists()  # node-local dir cleaned
    # resubmitting skips finished pieces
    r2 = subprocess.run(["bash", str(ROOT / "scripts/slurm/race_node.sh"), str(tsv)], env=env,
                        capture_output=True, text=True, timeout=60)
    assert f"skip (done): {ok}" in r2.stdout and r2.returncode == 1


def test_model_copy_tolerates_read_only_snapshot_files():
    sh = (ROOT / "scripts/slurm/tica_switch_eval.sh").read_text()
    assert "--remove-destination" in sh and ".tica_copied" in sh


# ---------------------------------------------------------------- report: unfinished pieces
def test_report_skips_unfinished_pieces(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import tica_math_race_report as rep

    row = {"episode": 0, "turn": 0, "concept": "c", "answer": "1"}
    for stem, done in (("m_p1", True), ("m_p2", False)):
        (tmp_path / f"{stem}.jsonl").write_text(json.dumps(row) + "\n")
        (tmp_path / f"{stem}_forks.jsonl").write_text("")
        if done:
            (tmp_path / f"{stem}.json").write_text("{}")
    rep.PIECES.clear()
    runs = rep.load_runs([f"m={tmp_path}/m_p*.jsonl"], [], [])
    assert len(runs["m"][0]) == 1
    assert rep.piece_info("m") == {"pieces": ["m_p1"], "pieces_incomplete": ["m_p2"]}


# ---------------------------------------------------------------- launcher plan
def test_launcher_dry_run_plan(tmp_path):
    spec = {"name": "t", "task": "sql,mbpp", "episodes": 32, "root": str(tmp_path),
            "models": {"a": ["cfgA", "base"],
                       "b": ["cfgB", "/nonexistent/adapter", {"seed": 2, "episodes": 48}]}}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/race_launch.py"), "--spec", str(sp),
                        "--dry-run", "--dependency", "afterok:1"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    tsv = (tmp_path / "t_sql" / "pieces.tsv").read_text().splitlines()
    outs = [ln.split("\t")[0] for ln in tsv]
    names = sorted(Path(o).name for o in outs if "/t_sql/" in o)
    assert names == ["a_p1.json", "a_p2.json", "b_p2.json", "b_p3.json", "b_p4.json"]
    assert sum("/t_mbpp/" in o for o in outs) == 5
    for ln in tsv:
        cmd = ln.split("\t")[1]
        for flag in ("--batch 16", "--turns 33", "--fork-panel 8", "--fast-forks",
                     "--switch-prob 0", "--cond real_score", "--max-k 12", "TICA_VLLM=1",
                     "TICA_VLLM_SLEEP=0", "TICA_VLLM_MEM=0.35", "TICA_MEM_FRACTION=0.60",
                     "TICA_VLLM_MAXLEN=32768", "TICA_VLLM_VENV=vllm-latest"):
            assert flag in cmd
        assert "refit-every" not in cmd and "refit_every" not in cmd
    assert "b_p4.json" in tsv[-1] or any("--seed 4" in ln for ln in tsv)
    assert "--dependency=afterok:1" in r.stdout
    assert "WARNING: adapter of b does not exist" in r.stdout
    bad = subprocess.run([sys.executable, str(ROOT / "scripts/race_launch.py"), "--spec", str(sp),
                          "--dry-run", "--extra", "--refit-every 4"],
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode != 0 and "refit" in bad.stderr
