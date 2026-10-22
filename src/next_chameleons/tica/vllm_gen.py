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

# cluster modules for the vllm-serving python; TICA_VLLM_MODULES overrides (TamIA: the vllm-latest venv needs
# "StdEnv/2023 python/3.11 arrow/24.0.0 cuda/12.6", as paper2_runs/vllm_steer_tp2)
MODULES = os.environ.get("TICA_VLLM_MODULES",
                         "StdEnv/2023 gcc/12.3 cuda/12.9 python/3.12 opencv/4.13.0 scipy-stack/2026a")


def convert_adapter(src: Path, dst: Path) -> None:
    """PEFT adapter on the text-only module tree -> the multimodal tree vLLM loads."""

    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(src / "adapter_model.safetensors"))
    save_file({k.replace("base_model.model.model.layers.",
                         "base_model.model.model.language_model.layers."): v
               for k, v in sd.items()}, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")


def merged_text_state(policy) -> dict:
    """The policy's CURRENT weights with the LoRA merged in (W + scale * B @ A, fp32 math, bf16
    out), in the checkpoint's naming (model.language_model.*), as CPU tensors. The training weights
    are not touched (no merge/unmerge round trip, which is lossy in bf16). Tied lm_head skipped."""

    import torch

    model = policy.model
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
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
            if mod in lora and key.endswith(".weight"):
                m = lora[mod]
                d = (m.lora_B["default"].weight.float() @ m.lora_A["default"].weight.float())
                t = (t.float() + d * m.scaling["default"]).to(t.dtype)
            if key.startswith("model.") and not key.startswith("model.language_model."):
                key = "model.language_model." + key[len("model."):]
            out[key] = t.to("cpu", non_blocking=True)
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
                 local: Path | None = None) -> None:
        """mode "merged" (default): serve the base architecture and RELOAD the LoRA-merged weights
        on every weights change (vLLM's LoRA decode is ~4x slower than base decode here, 5609973);
        base generations reload the original weights. mode "lora": serve the adapter (slow)."""
        self.py = python or os.environ.get("TICA_VLLM_PY")
        if not self.py:
            raise RuntimeError("vLLM generation needs TICA_VLLM_PY (the vllm-serving python)")
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.model = model_path
        self.version = None  # Policy.weights_version of the loaded adapter
        self.lora = None  # served adapter name (None = base model)
        self.sleeping = False
        self.mode = mode
        self.loaded = "base"  # which weights the server holds: "base" or ("policy", version)
        self.local = Path(local or os.environ.get("SLURM_TMPDIR", "/tmp")) / "vllm_merged"
        lora_flags = f"--enable-lora --max-lora-rank {max_rank} --max-loras 1 " if mode == "lora" \
            else ""
        cmd = (f"module --force purge >/dev/null 2>&1; module load {MODULES} >/dev/null 2>&1; "
               f"exec {self.py} -m vllm.entrypoints.openai.api_server --model {model_path} "
               f"--host 127.0.0.1 --port {self.port} --dtype bfloat16 "
               f"--gpu-memory-utilization {mem} --max-model-len {max_len} "
               f"{lora_flags}--enable-sleep-mode --enable-prefix-caching")
        env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1",
               "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        env.pop("PYTHONPATH", None)  # the training env's src / site dirs must not leak in
        self.log = open(log or self.work / "vllm.log", "w")  # noqa: SIM115
        self.proc = subprocess.Popen(["bash", "-lc", cmd], env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT)
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
            if time.time() - t0 > timeout:
                self.close()
                raise RuntimeError("vLLM server did not come up")
            time.sleep(5)
        self.startup_sec = round(time.time() - t0)

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
        if not self.sleeping:
            self._post("/sleep?level=1")
            self.sleeping = True

    def wake(self) -> None:
        if self.sleeping:
            self._post("/wake_up")
            self.sleeping = False

    def _reload(self, path: str) -> None:
        self._post("/collective_rpc", {"method": "reload_weights",
                                       "kwargs": {"weights_path": path}})

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
            save_file(merged_text_state(policy), str(self.local / "model.safetensors"))
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
        out = self._post("/v1/completions", body)
        ch = sorted(out["choices"], key=lambda c: c["index"])
        return [{"text": c["text"].strip(), "ids": list(c.get("token_ids") or []),
                 "finished": c.get("finish_reason") == "stop", "stop": None}
                for c in ch]

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        self.log.close()


def maybe_attach(pol, model_path: str, workdir: Path) -> VLLMGenerator | None:
    """Attach a colocated vLLM server to `pol` when TICA_VLLM=1 (job scripts unpack the vLLM env
    and export TICA_VLLM_PY). TICA_VLLM_MEM = its GPU memory fraction while awake (default 0.4;
    0.25 left no room for the KV cache, 5608325). TICA_VLLM_MAXLEN = max context (default 16384:
    8192 rejected the 24-turn capability forks at k = 23 with HTTP 400, 5623494)."""

    if os.environ.get("TICA_VLLM") != "1":
        return None
    import atexit

    gen = VLLMGenerator(model_path, Path(workdir),
                        mem=float(os.environ.get("TICA_VLLM_MEM", "0.4")),
                        max_len=int(os.environ.get("TICA_VLLM_MAXLEN", "16384")),
                        mode=os.environ.get("TICA_VLLM_MODE", "merged"),
                        max_rank=int(os.environ.get("TICA_VLLM_MAX_RANK", "128")))
    gen.sleep()  # training owns the GPU until the first generation
    pol.attach_vllm(gen)
    atexit.register(gen.close)
    print(json.dumps({"vllm": {"startup_sec": gen.startup_sec, "port": gen.port}}), flush=True)
    return gen
