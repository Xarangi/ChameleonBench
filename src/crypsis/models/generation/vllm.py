"""Colocated vLLM generation (from tica/vllm_gen.py). UNTESTED in CI: vllm is not installed in
the training venv; the server runs from a separate interpreter.

A vLLM OpenAI server runs as a SUBPROCESS on the same GPU, in its own environment. Between
generations it SLEEPS (level 1: weights offloaded, KV cache freed), so training keeps the GPU
memory. Before a generation the policy's current weights are served (merged mode: the
LoRA-merged state is written and reloaded; lora mode: the adapter is saved, renamed and loaded),
only when policy.weights_version changed. Prompts go as TOKEN IDS rendered by the policy.

Cluster setup (module loads, the vllm interpreter) is NOT done here: the caller passes the
`python` path, an `env` dict and optionally a `wrapper` command prefix (e.g. a site script that
loads modules and execs its arguments); the site profile supplies them.
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
from collections.abc import Mapping, Sequence
from pathlib import Path

from crypsis.models.generation.base import Generation, Generator, Prompt, prompt_ids

# Per served checkpoint family (config.json model_type). Absent (Qwen3.5) = defaults.
#   flags:   extra server flags (Gemma-4 12B is served text-only).
#   buffers: buffers (by name) the merged state carries besides the parameters.
VLLM_FAMILY = {
    "gemma4_unified": {"flags": ("--language-model-only",), "buffers": ("layer_scalar",)},
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
    """The policy's CURRENT weights with the LoRA merged in (W + scale * B @ A, fp32 math, cast
    back), in the checkpoint's naming (model.language_model.*), as CPU tensors. The training
    weights are not touched. Tied lm_head skipped. buffers: also the buffers with these names."""

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


class VLLMServer:
    """The vLLM server subprocess and its HTTP API (tica VLLMGenerator).

    mode "merged" (default): serve the base architecture and RELOAD the LoRA-merged weights on
    every weights change; base generations reload the original weights. mode "lora": serve the
    adapter (slower decode). local: where the merged state is written (default workdir/merged).
    """

    def __init__(self, model_path: str, workdir: Path, *, python: str,
                 env: Mapping[str, str] | None = None, wrapper: Sequence[str] = (),
                 mem: float = 0.4, max_len: int = 8192, max_rank: int = 16,
                 max_num_seqs: int = 256, log: Path | None = None, timeout: float = 1500.0,
                 mode: str = "merged", local: Path | None = None) -> None:
        self.py = python
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.model = model_path
        self.version = None  # policy.weights_version of the loaded adapter (lora mode)
        self.lora = None  # served adapter name (None = base model)
        self.sleeping = False
        self.mode = mode
        self.loaded: object = "base"  # "base" or ("policy", version)
        self.local = Path(local) if local else self.work / "merged"
        self.family = model_family(model_path)
        cmd = [*wrapper, python, "-m", "vllm.entrypoints.openai.api_server", "--model",
               model_path, "--host", "127.0.0.1", "--port", str(self.port), "--dtype", "bfloat16",
               "--gpu-memory-utilization", str(mem), "--max-model-len", str(max_len),
               "--max-num-seqs", str(max_num_seqs)]
        if mode == "lora":
            cmd += ["--enable-lora", "--max-lora-rank", str(max_rank), "--max-loras", "1"]
        cmd += [*self.family.get("flags", ()), "--enable-sleep-mode", "--enable-prefix-caching"]
        full_env = {**os.environ, **(env or {}), "VLLM_SERVER_DEV_MODE": "1",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        full_env.pop("PYTHONPATH", None)  # the training env's src / site dirs must not leak in
        self.log = open(log or self.work / "vllm.log", "w")  # noqa: SIM115
        self.proc = subprocess.Popen(cmd, env=full_env, stdout=self.log,
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

        ver = policy.weights_version
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

    def complete(self, prompt_ids: list[list[int]], max_new_tokens: int, *, temperature: float,
                 top_p: float = 1.0, seed: int = 0, base: bool = False) -> list[dict]:
        """One completion per prompt: {text, ids, finished, stop=None}."""

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


class VLLMGenerator:
    """Generator over a VLLMServer for a policy (tica Policy._generate_vllm): same prompt token
    ids as HF; inside policy.reference() the base weights are served. stop_ids are not supported
    by the server path: those calls go to `fallback` (tica fell back to HF generate)."""

    def __init__(self, policy, server: VLLMServer, *, fallback: Generator | None = None) -> None:
        self.policy = policy
        self.server = server
        self.fallback = fallback

    def generate(self, prompts: Sequence[Prompt], max_new_tokens: int, *,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 0,
                 stop_ids: set[int] | None = None) -> list[Generation]:
        if stop_ids:
            if self.fallback is None:
                raise ValueError("VLLMGenerator: stop_ids need a fallback generator")
            return self.fallback.generate(prompts, max_new_tokens, temperature=temperature,
                                          top_p=top_p, seed=seed, stop_ids=stop_ids)
        import torch

        pol, gen = self.policy, self.server
        ids = [prompt_ids(pol, c) for c in prompts]
        base = pol.in_reference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # give vLLM back the memory training cached
        gen.wake()
        try:
            if base:
                gen.use_base()
            else:
                gen.sync(pol)
            outs = gen.complete(ids, max_new_tokens, temperature=temperature, top_p=top_p,
                                seed=seed, base=base)
        finally:
            gen.sleep()
        return [Generation(o["text"], p, o["ids"], o["finished"], o["stop"])
                for p, o in zip(ids, outs, strict=True)]
