"""HumanEval translated to JavaScript (MultiPL-E humaneval-js; Node assert tests). Ported from
benchmarks/capability_heldout.py (`js_prompt`, `run_js_tests`) with the node binary and limits as
explicit arguments (default: `node` on PATH). Behaviour is identical."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from crypsis.data.tasks.form import extract_js_code


def prompt(item: dict[str, Any]) -> str:
    """The generation prompt of a HumanEval-JS item ({fn, prompt, tests})."""
    return (f"You are an expert JavaScript programmer. Implement the JavaScript function "
            f"`{item['fn']}` described below:\n\n```javascript\n{item['prompt'].rstrip()}\n```\n\n"
            "Write the complete function, signature included, in a single ```javascript code "
            "block.")


def run_tests(code: str, tests: str, *, node: str | None = None, timeout: float = 10.0,
              mem_mb: int = 2048) -> dict[str, Any]:
    """Run `code` + the problem's Node assert tests in a fresh `node` process (temp dir, no
    inherited environment, heap cap, timeout, own process group); passed iff exit code 0."""
    node = node or shutil.which("node")
    if node is None:
        return {"passed": False, "status": "spawn_error:NodeNotFound"}
    env = {"PATH": "/usr/bin:/bin", "http_proxy": "http://0.0.0.0:9",
           "https_proxy": "http://0.0.0.0:9", "NODE_OPTIONS": ""}
    with tempfile.TemporaryDirectory(prefix="hejs_") as td:
        path = Path(td) / "prog.js"
        path.write_text(f"{code}\n\n{tests}\n", encoding="utf-8")
        try:
            proc = subprocess.Popen([node, f"--max-old-space-size={int(mem_mb)}", str(path)],
                                    cwd=td, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    start_new_session=True)
        except OSError as e:
            return {"passed": False, "status": f"spawn_error:{type(e).__name__}"}
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, 9)
            proc.communicate()
            return {"passed": False, "status": "timeout"}
    if proc.returncode == 0:
        return {"passed": True, "status": "passed"}
    text = (err or b"").decode("utf-8", "replace")
    return {"passed": False, "status": "assertion" if "AssertionError" in text else "error"}


def check(answer: str, item: dict[str, Any], *, node: str | None = None,
          timeout: float = 10.0) -> bool:
    """Does the generation pass the item's tests? (the `jseval` task's correctness)."""
    return bool(run_tests(extract_js_code(answer), item["tests"], node=node,
                          timeout=timeout)["passed"])
