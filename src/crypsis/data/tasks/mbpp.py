"""MBPP (sanitized): generation prompt and sandboxed execution of the dataset's asserts. Ported from
benchmarks/capability_heldout.py (`mbpp_prompt`, `run_tests`); the interpreter is an explicit
argument (default: the running one). Behaviour is identical."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from crypsis.data.tasks.form import extract_code


def prompt(item: dict[str, Any]) -> str:
    """The generation prompt of an MBPP item ({prompt, test_list})."""
    tests = "\n".join(item["test_list"])
    return (f"You are an expert Python programmer, and here is your task: {item['prompt']} "
            f"Your code should pass these tests:\n\n{tests}\n\n"
            "Write the complete Python code in a single ```python code block.")


# Prepended to every program: resource limits, no sockets (no network), no os.system.
# (Limits are set by the child itself: preexec_fn is unsafe with the executor's threads.)
_SANDBOX_PRELUDE = """\
import resource as _r
_r.setrlimit(_r.RLIMIT_AS, ({mem}, {mem}))
_r.setrlimit(_r.RLIMIT_FSIZE, (16 << 20, 16 << 20))
_r.setrlimit(_r.RLIMIT_CORE, (0, 0))
import socket as _s
def _nonet(*a, **k):
    raise OSError("network disabled in sandbox")
_s.socket = _nonet; _s.create_connection = _nonet
import os as _o
_o.system = _nonet
del _r, _s, _o, _nonet
"""


def run_tests(code: str, test_imports: list[str], test_list: list[str], *, timeout: float = 10.0,
              mem_bytes: int = 4 << 30, python: str | None = None) -> dict[str, Any]:
    """Run `code` + imports + asserts in an isolated subprocess; passed iff exit code 0."""
    prelude = _SANDBOX_PRELUDE.format(mem=int(mem_bytes))
    program = "\n".join([prelude, *test_imports, code, "", *test_list, ""])
    env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "http_proxy": "http://0.0.0.0:9", "https_proxy": "http://0.0.0.0:9",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(prefix="mbpp_") as td:
        path = Path(td) / "prog.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.Popen([python or sys.executable, "-I", "-B", str(path)], cwd=td,
                                    env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, start_new_session=True)
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
    last = (err or b"").decode("utf-8", "replace").strip().splitlines()
    kind = last[-1].split(":")[0][:60] if last else f"exit{proc.returncode}"
    return {"passed": False, "status": "assertion" if kind == "AssertionError" else "error",
            "error_type": kind}


def check(answer: str, item: dict[str, Any], *, timeout: float = 10.0,
          python: str | None = None) -> bool:
    """Does the generation pass the item's asserts? (the `mbpp` task's correctness)."""
    return bool(run_tests(extract_code(answer), item["test_imports"], item["test_list"],
                          timeout=timeout, python=python)["passed"])
