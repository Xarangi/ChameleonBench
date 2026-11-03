"""Colocated vLLM generation for TICA trainers / evals (REPLICATION 2026-10-14s).

HF generate on Qwen3.5 is launch-bound (~120 ms per decode step at batch 20; 5581212); vLLM 0.24 is
~5x faster (5584371) and applies our LoRA exactly once the adapter keys are renamed to the
multimodal module tree (5602999: vLLM+LoRA vs HF+LoRA greedy agreement == vLLM base vs HF base).

A vLLM OpenAI server runs as a SUBPROCESS on the same GPU, in its own environment (vllm-serving
venv + cluster modules; torch differs from the training env). Between generations it SLEEPS
(level 1: weights offloaded to CPU, KV cache freed), so training keeps the GPU memory. Before a
generation the current adapter is saved, its keys renamed, and loaded in place under one name
(only when the weights changed: Policy.weights_version). Prompts are sent as TOKEN IDS rendered
by the Policy (same chat template as training); outputs come back as token ids + text.

Enable: set TICA_VLLM_PY (the vllm-serving python, e.g. unpacked node-local by the job script)
and call Policy.attach_vllm(...) (tica.py does this when model.vllm is true).

On TamIA (11-03) scripts/slurm/_local_env.sh does the setup when TICA_VLLM=1: it unpacks
venvs/$TICA_VLLM_VENV.tar (default vllm-latest = vLLM 0.21, Qwen3.5 OK; vllm-g4 = vLLM 0.25 for
Gemma-4) node-local and exports TICA_VLLM_PY + TICA_VLLM_MODULES. Evals with frozen weights set
TICA_VLLM_SLEEP=0: the server stays awake (no wake / sleep per call, and its prefix cache survives
between turns), so TICA_VLLM_MEM must leave room for the HF model beside it.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# cluster modules for the vllm-serving python; TICA_VLLM_MODULES overrides (_local_env.sh exports
# the right set per venv). Default = TamIA's vllm-latest set (the Narval default was dead here).
MODULES_DEFAULT = "StdEnv/2023 python/3.11 arrow/24.0.0 cuda/12.6"
MODULES = os.environ.get("TICA_VLLM_MODULES", MODULES_DEFAULT)

# Per served checkpoint family (config.json model_type; 10-24 Gemma-4 port). Absent (Qwen3.5) =
# unchanged.
#   flags:   extra server flags. Gemma-4 12B ("unified": text + image + audio inputs) is served
#            TEXT-ONLY: --language-model-only sets every modality limit to 0.
#   buffers: buffers (by name) the merged state carries besides the parameters (Gemma-4's
#            per-layer layer_scalar is a checkpoint buffer of the decoder layer).
#   Gemma-4 needs vLLM >= 0.23 (Gemma4UnifiedForConditionalGeneration): the vllm-g4 venv (vLLM
#   0.25, transformers 5.14.1), TICA_VLLM_MODULES "StdEnv/2023 gcc python/3.11 arrow/24.0.0
#   cuda/12.6 opencv".
VLLM_FAMILY = {
    "gemma4_unified": {"flags": "--language-model-only ", "buffers": ("layer_scalar",)},
}


def model_family(model_path) -> dict:
    """VLLM_FAMILY entry of the checkpoint at model_path ({} if none / no config.json)."""

    p = Path(model_path) / "config.json"
    if not p.is_file():
        return {}
    return VLLM_FAMILY.get(json.loads(p.read_text()).get("model_type"), {})


def convert_adapter(src: Path, dst: Path) -> None:
    """PEFT adapter on the text-only module tree -> the multimodal tree vLLM loads."""

    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(src / "adapter_model.safetensors"))
    save_file({k.replace("base_model.model.model.layers.",
                         "base_model.model.model.language_model.layers."): v
               for k, v in sd.items()}, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")


def merged_text_state(policy, buffers: tuple[str, ...] = ()) -> dict:
    """The policy's CURRENT weights with the LoRA merged in (W + scale * B @ A, fp32 math, bf16
    out), in the checkpoint's naming (model.language_model.*), as CPU tensors. The training weights
    are not touched (no merge/unmerge round trip, which is lossy in bf16). Tied lm_head skipped.
    buffers: also the buffers with these names (VLLM_FAMILY; () = parameters only, as before)."""

    import torch

    model = policy.model
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Inside Policy.merged() (adaptive.merged_reads: races run the rollout in it) every adapted
    # base weight ALREADY holds W0 + scaling * B @ A (the same fp32 merge, cast to bf16): take it as
    # is. Adding the delta again served W0 + 2 * B @ A (11-03 fix; only vLLM-generation runs whose
    # first sync fell inside merged() were affected).
    premerged = bool(getattr(policy, "_merged_active", False))
    lora = {n: m for n, m in base.named_modules()
            if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})}
    out = {}
    with torch.no_grad():
        for name, p in base.named_parameters():
            if ".lora_" in name or name.startswith("lm_head"):
                continue
            key = name.replace(".base_layer", "")
            t = p.detach()
            mod = key.rsplit(".", 1)[0]
            if mod in lora and key.endswith(".weight") and not premerged:
                m = lora[mod]
                d = (m.lora_B["default"].weight.float() @ m.lora_A["default"].weight.float())
                t = (t.float() + d * m.scaling["default"]).to(t.dtype)
            if key.startswith("model.") and not key.startswith("model.language_model."):
                key = "model.language_model." + key[len("model."):]
            out[key] = t.to("cpu", non_blocking=True)
        for name, b in (base.named_buffers() if buffers else ()):
            if name.rsplit(".", 1)[-1] not in buffers:
                continue
            key = name
            if key.startswith("model.") and not key.startswith("model.language_model."):
                key = "model.language_model." + key[len("model."):]
            out[key] = b.detach().to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VLLMGenerator:
    def __init__(self, model_path: str, workdir: Path, *, python: str | None = None,
                 mem: float = 0.4, max_len: int = 8192, max_rank: int = 16,
                 log: Path | None = None, timeout: float = 1500.0, mode: str = "merged",
                 local: Path | None = None, keep_awake: bool = False,
                 extra_args: str = "") -> None:
        """mode "merged" (default): serve the base architecture and RELOAD the LoRA-merged weights
        on every weights change (vLLM's LoRA decode is ~4x slower than base decode here, 5609973);
        base generations reload the original weights. mode "lora": serve the adapter (slow)."""
        self.py = python or os.environ.get("TICA_VLLM_PY")
        if not self.py:
            raise RuntimeError("vLLM generation needs TICA_VLLM_PY (the vllm-serving python)")
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.model = model_path
        self.version = None  # Policy.weights_version of the loaded adapter
        self.lora = None  # served adapter name (None = base model)
        self.sleeping = False
        self.mode = mode
        self.loaded = "base"  # which weights the server holds: "base" or ("policy", version)
        # keep_awake (TICA_VLLM_SLEEP=0): never sleep -- frozen-weight evals keep the KV / prefix
        # cache between calls and skip the wake / sleep round trips
        self.keep_awake = keep_awake
        self.stats = {"reloads": 0, "calls": 0, "prompts": 0, "gen_tokens": 0, "sec_gen": 0.0,
                      "sec_reload": 0.0}
        self.local = Path(local or os.environ.get("SLURM_TMPDIR", "/tmp")) / "vllm_merged"
        lora_flags = f"--enable-lora --max-lora-rank {max_rank} --max-loras 1 " if mode == "lora" \
            else ""
        self.family = model_family(model_path)  # 10-24: per-model server flags / merged buffers
        lora_flags += self.family.get("flags", "")
        cmd = (f"module --force purge >/dev/null 2>&1; module load {MODULES} >/dev/null 2>&1; "
               f"exec {self.py} -m vllm.entrypoints.openai.api_server --model {model_path} "
               f"--host 127.0.0.1 --port {{PORT}} --dtype bfloat16 "
               f"--gpu-memory-utilization {mem} --max-model-len {max_len} "
               f"--max-num-seqs {os.environ.get('TICA_VLLM_MAX_SEQS', '256')} "  # vLLM 0.21 default 1024 > Qwen3.5 Mamba blocks (637200)
               f"{lora_flags}--enable-sleep-mode --enable-prefix-caching {extra_args}")
        self.cmd = cmd
        self.env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        self.env.pop("PYTHONPATH", None)  # the training env's src / site dirs must not leak in
        self.log = open(log or self.work / "vllm.log", "w")  # noqa: SIM115
        self.start_timeout = timeout
        # STALL WATCHDOG (11-03, race mbpp_pass64 / 664118): vLLM 0.21 + Qwen3.5-2B (Mamba 'align'
        # prefix caching, experimental) sometimes wedges mid-batch -- requests stay Running /
        # Waiting at 0 tok/s and the engine logs nothing more. With the old 3600 s socket timeout
        # each stall burned an hour and then killed the piece (kl03_2b_p1, strict2b_p4). Now one
        # generation call may take TICA_VLLM_CALL_TIMEOUT s (default 300; the slowest whole race
        # turn, several calls, was 59 s); on a timeout / dropped connection the server is killed,
        # restarted, given back the weights it served, and the call is retried (TICA_VLLM_RETRIES,
        # default 2; see _completions).
        self.call_timeout = float(os.environ.get("TICA_VLLM_CALL_TIMEOUT", "300"))
        self.retries = int(os.environ.get("TICA_VLLM_RETRIES", "2"))
        self.stats["restarts"] = 0
        self._start()

    def _start(self) -> None:
        """Launch the server (own process group, so close() also kills the EngineCore child) and
        wait for /health."""

        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = self.cmd.replace("--port {PORT}", f"--port {self.port}")
        self.proc = subprocess.Popen(["bash", "-lc", cmd], env=self.env, stdout=self.log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited ({self.proc.returncode}); see {self.log.name}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.time() - t0 > self.start_timeout:
                self._kill()
                raise RuntimeError("vLLM server did not come up")
            time.sleep(5)
        self.startup_sec = round(time.time() - t0)

    def _kill(self) -> None:
        """Terminate the server's whole process group (API server + EngineCore)."""

        import contextlib
        import signal

        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.proc.pid, sig)
            try:
                self.proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue

    def restart(self) -> None:
        """Kill a stalled server, start a fresh one and give it back the weights it served."""

        self._kill()
        self.log.write(f"\n### TICA watchdog: restarting vLLM server ({time.ctime()})\n")
        self.log.flush()
        self.stats["restarts"] += 1
        self._start()
        self.sleeping = False
        if self.mode == "merged":
            if self.loaded != "base":  # the merged policy weights are still on disk
                self._reload(str(self.local))
        elif self.lora is not None:
            self._post("/v1/load_lora_adapter", {"lora_name": "pol",
                                                 "lora_path": str(self.work / "adapter_vllm"),
                                                 "load_inplace": True})

    def _completions(self, body: dict) -> list[dict]:
        """/v1/completions with the stall watchdog: per-call timeout; on a stall restart the server
        and retry, the retries in sub-requests of TICA_VLLM_RETRY_CHUNK prompts (default 32; a
        different batch shape in case the stall depends on it -- every request carries its seed,
        so the samples do not depend on the batching). Returns the choices in prompt order."""

        prompts = body["prompt"]
        chunk = max(1, int(os.environ.get("TICA_VLLM_RETRY_CHUNK", "32")))
        for attempt in range(self.retries + 1):
            step = len(prompts) if attempt == 0 else chunk
            try:
                out = []
                for i in range(0, len(prompts), max(step, 1)):
                    res = self._post("/v1/completions", {**body, "prompt": prompts[i : i + step]},
                                     timeout=self.call_timeout)
                    out.extend(sorted(res["choices"], key=lambda c: c["index"]))
                return out
            except (TimeoutError, ConnectionError, urllib.error.URLError) as err:
                if isinstance(err, urllib.error.HTTPError) or attempt == self.retries:
                    raise
                import sys

                print(json.dumps({"vllm_watchdog": {"attempt": attempt + 1, "prompts": len(prompts),
                                                    "error": repr(err)[:200]}}),
                      file=sys.stderr, flush=True)
                self.restart()
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------ http
    def _post(self, path: str, body: dict | None = None, timeout: float = 3600.0):
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(self.url + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as err:  # surface vLLM's reason (e.g. context too long)
            raise RuntimeError(f"vLLM {path}: HTTP {err.code}: "
                               f"{err.read().decode(errors='replace')[:500]}") from err
        try:  # load_lora_adapter / sleep reply with plain text, completions with JSON
            return json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return raw.decode(errors="replace")

    # ------------------------------------------------------------------ lifecycle
    def sleep(self) -> None:
        if not self.sleeping and not self.keep_awake:
            self._post("/sleep?level=1")
            self.sleeping = True

    def wake(self) -> None:
        if self.sleeping:
            self._post("/wake_up")
            self.sleeping = False

    def _reload(self, path: str) -> None:
        t0 = time.time()
        self._post("/collective_rpc", {"method": "reload_weights",
                                       "kwargs": {"weights_path": path}})
        if self.keep_awake:  # cached prefixes were computed with the old weights
            self._post("/reset_prefix_cache")
        self.stats["reloads"] += 1
        self.stats["sec_reload"] += time.time() - t0

    def use_base(self) -> None:
        if self.mode == "merged" and self.loaded != "base":
            self._reload(self.model)
            self.loaded = "base"

    def sync(self, policy) -> None:
        """Serve the policy's CURRENT weights (no-op if already served)."""

        ver = getattr(policy, "weights_version", 0)
        if self.mode == "merged":
            if not hasattr(policy.model, "peft_config"):
                self.use_base()
                return
            if self.loaded == ("policy", ver):
                return
            from safetensors.torch import save_file

            self.local.mkdir(parents=True, exist_ok=True)
            save_file(merged_text_state(policy, tuple(self.family.get("buffers", ()))),
                      str(self.local / "model.safetensors"))
            self._reload(str(self.local))
            self.loaded = ("policy", ver)
            return
        if self.version == ver:
            return
        if not hasattr(policy.model, "peft_config"):
            self.lora, self.version = None, ver
            return
        src = self.work / "adapter_hf"
        policy.save_adapter(src)
        convert_adapter(src, self.work / "adapter_vllm")
        self._post("/v1/load_lora_adapter", {"lora_name": "pol",
                                             "lora_path": str(self.work / "adapter_vllm"),
                                             "load_inplace": True})
        self.lora, self.version = "pol", ver

    def generate(self, prompt_ids: list[list[int]], max_new_tokens: int, *, temperature: float,
                 top_p: float = 1.0, seed: int = 0, base: bool = False) -> list[dict]:
        """One completion per prompt (Policy.generate's contract: text, ids, finished)."""

        if self.mode == "merged":
            name = self.model
        else:
            name = self.model if base or self.lora is None else self.lora
        body = {"model": name,
                "prompt": prompt_ids, "max_tokens": max_new_tokens,
                "temperature": temperature, "top_p": top_p, "seed": seed, "n": 1,
                "return_token_ids": True, "skip_special_tokens": True}
        t0 = time.time()
        ch = self._completions(body)
        self.stats["calls"] += 1
        self.stats["prompts"] += len(prompt_ids)
        self.stats["gen_tokens"] += sum(len(c.get("token_ids") or []) for c in ch)
        self.stats["sec_gen"] += time.time() - t0
        return [{"text": c["text"].strip(), "ids": list(c.get("token_ids") or []),
                 "finished": c.get("finish_reason") == "stop", "stop": None}
                for c in ch]

    def summary(self) -> dict:
        """Counters (reloads, calls, prompts, generated tokens, seconds) + the server's prefix-cache
        hit rate (Prometheus /metrics; None if unavailable)."""

        out = {**self.stats, "startup_sec": getattr(self, "startup_sec", None)}
        out["sec_gen"], out["sec_reload"] = round(out["sec_gen"], 1), round(out["sec_reload"], 1)
        try:
            with urllib.request.urlopen(self.url + "/metrics", timeout=10) as r:
                txt = r.read().decode(errors="replace")
            tot = {}
            for line in txt.splitlines():
                for k in ("prefix_cache_hits", "prefix_cache_queries"):
                    if line.startswith(f"vllm:{k}_total"):
                        tot[k] = tot.get(k, 0.0) + float(line.rsplit(" ", 1)[-1])
            if tot.get("prefix_cache_queries"):
                out["prefix_cache_hit_rate"] = round(tot["prefix_cache_hits"]
                                                     / tot["prefix_cache_queries"], 3)
        except Exception:  # metrics are a diagnostic only
            pass
        return out

    def close(self) -> None:
        try:
            self._kill()
        except Exception:
            self.proc.kill()
        self.log.close()


def maybe_attach(pol, model_path: str, workdir: Path) -> VLLMGenerator | None:
    """Attach a colocated vLLM server to `pol` when TICA_VLLM=1 (job scripts unpack the vLLM env
    and export TICA_VLLM_PY). TICA_VLLM_MEM = its GPU memory fraction while awake (default 0.4;
    0.25 left no room for the KV cache, 5608325). TICA_VLLM_MAXLEN = max context (default 16384:
    8192 rejected the 24-turn capability forks at k = 23 with HTTP 400, 5623494). TICA_VLLM_SLEEP=0:
    never sleep (frozen-weight evals). TICA_VLLM_ARGS: extra server flags."""

    if os.environ.get("TICA_VLLM") != "1":
        return None
    import atexit

    gen = VLLMGenerator(model_path, Path(workdir),
                        mem=float(os.environ.get("TICA_VLLM_MEM", "0.4")),
                        max_len=int(os.environ.get("TICA_VLLM_MAXLEN", "16384")),
                        mode=os.environ.get("TICA_VLLM_MODE", "merged"),
                        max_rank=int(os.environ.get("TICA_VLLM_MAX_RANK", "128")),
                        keep_awake=os.environ.get("TICA_VLLM_SLEEP", "1") == "0",
                        extra_args=os.environ.get("TICA_VLLM_ARGS", ""))
    gen.sleep()  # training owns the GPU until the first generation (no-op with keep_awake)
    pol.attach_vllm(gen)
    atexit.register(gen.close)
    print(json.dumps({"vllm": {"startup_sec": gen.startup_sec, "port": gen.port}}), flush=True)
    return gen
