"""The policy model: a base causal LM plus an optional LoRA adapter (from tica/policy.py).

`Policy` owns loading, the adapter switches (`reference()`: adapter off, `merged()`: adapter merged
into the base weights for fast no-grad reads), saving the adapter, and the decoder blocks. Chat
rendering lives in `ChatRenderer` (`policy.chat`), activation reads in `ActivationReader`
(`policy.reader()`), generation in `crypsis.models.generation`.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from crypsis.models.chat import ChatFormat, ChatRenderer

log = logging.getLogger(__name__)

DECODER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "base_model.model.model.layers",
    "base_model.model.model.language_model.layers",
)

# TEXT-ONLY loading of multimodal checkpoints (Gemma-4 12B): config.json model_type ->
# checkpoint-key renames onto the text-only causal-LM class of its text_config (no vision / audio
# modules exist in the loaded model, so LoRA "all-linear", hooks and the LM head only see the text
# decoder). Qwen3.5 keeps the plain path (its AutoModelForCausalLM class is already text-only).
TEXT_ONLY_KEY_MAPPING = {
    "gemma4_unified": {r"^model\.language_model\.": "model."},
}


def configure_linear_attention(model, fla_grad: bool) -> int:
    """Qwen3.5 gated-DeltaNet layers: use the fla kernel for no-grad passes always, and for grad
    passes only if fla_grad (fla 0.4.1's backward does not compile under triton < 3.3). Returns
    the number of layers patched (0 = no fla kernel, nothing to do)."""

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    except ImportError:
        return 0
    if fla_grad:
        try:
            import triton

            if tuple(int(x) for x in triton.__version__.split(".")[:2]) < (3, 3):
                log.warning("fla_grad needs triton >= 3.3 (have %s); grad passes use the torch "
                            "fallback", triton.__version__)
                fla_grad = False
        except ImportError:
            fla_grad = False
    n = 0
    for m in model.modules():
        fast = getattr(m, "chunk_gated_delta_rule", None)
        if fast is None or fast is mq.torch_chunk_gated_delta_rule or fla_grad:
            continue

        def pick(*a, _fast=fast, **k):
            fn = mq.torch_chunk_gated_delta_rule if torch.is_grad_enabled() else _fast
            return fn(*a, **k)

        m.chunk_gated_delta_rule = pick
        n += 1
    return n


def text_config_attr(model, name: str, default=None):
    """A text-decoder config attribute (looks through PEFT wrappers and multimodal text_config)."""

    cfg = getattr(model, "config", None)
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is not None and getattr(c, name, None) is not None:
            return getattr(c, name)
    return default


def load_causal_lm(path: str, dtype: str = "bfloat16"):
    """AutoModelForCausalLM.from_pretrained(path), except for checkpoints listed in
    TEXT_ONLY_KEY_MAPPING: there the TEXT-ONLY causal LM of the checkpoint's text_config is built
    and only the (renamed) text weights load; a missing text weight raises (the tied LM head is
    the only allowed one). The model keeps the snapshot path as its name (cache keys)."""

    from transformers import AutoConfig, AutoModelForCausalLM

    mt = None
    cj = Path(path) / "config.json"
    if cj.is_file():
        mt = json.loads(cj.read_text()).get("model_type")
    if mt not in TEXT_ONLY_KEY_MAPPING:
        return AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype),
                                                    device_map="auto")
    text_cfg = AutoConfig.from_pretrained(path).get_text_config()
    model, info = AutoModelForCausalLM.from_pretrained(
        path, config=text_cfg, dtype=getattr(torch, dtype), device_map="auto",
        key_mapping=TEXT_ONLY_KEY_MAPPING[mt], output_loading_info=True)
    tied = {"lm_head.weight"} if getattr(text_cfg, "tie_word_embeddings", False) else set()
    missing = sorted(set(info.get("missing_keys") or ()) - tied)
    if missing:
        raise RuntimeError(f"text-only load of {path}: {len(missing)} weights missing, e.g. "
                           f"{missing[:5]}")
    model.config._name_or_path = str(path)
    return model


def find_decoder_layers(model):
    """The ModuleList of decoder blocks (through PEFT and multimodal wrappers)."""

    for path in DECODER_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("cannot locate decoder layers")


def lora_config(lora: dict):
    """peft.LoraConfig from a dict: rank (or r, default 16), alpha (32), dropout (0),
    target_modules ("all-linear")."""

    import peft

    return peft.LoraConfig(
        r=lora.get("rank", lora.get("r", 16)),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


class Policy:
    """A causal LM (optionally PEFT-wrapped with one "default" LoRA adapter) and its tokenizer.

    batch_size is the default batch of readers and generators built on this policy (callers may
    change it temporarily). weights_version is bumped by trainers after every optimizer step so a
    served copy (vLLM) knows when to re-sync.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        enable_thinking: bool = False,
        batch_size: int = 8,
        history_reasoning: bool = False,
        keep_think_history: bool = False,
        chat_format: ChatFormat | None = None,
        adapter_path: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.chat = ChatRenderer(tokenizer, enable_thinking=enable_thinking,
                                 history_reasoning=history_reasoning,
                                 keep_think_history=keep_think_history, chat_format=chat_format)
        # final-logit soft-capping (Gemma-4: 30); the chunked LM terms apply it as the model does
        self.logit_softcap = text_config_attr(model, "final_logit_softcapping")
        self.layers = find_decoder_layers(model)
        self.eos_ids = self._eos_ids()
        self.adapter_path = adapter_path
        self.weights_version = 0
        self._ref_depth = 0
        self._merged_active = False
        self._merged_mods: list = []
        self._w0: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls,
        path: str,
        *,
        dtype: str = "bfloat16",
        lora: dict | None = None,
        adapter: str | None = None,
        trainable: bool = False,
        grad_ckpt: bool = True,
        fla_grad: bool = False,
        **kw,
    ) -> Policy:
        """Load a model and its tokenizer. `lora` (rank, alpha, dropout, target_modules) attaches
        a fresh adapter; `adapter` loads a saved one (takes precedence). trainable=True enables
        gradient checkpointing unless grad_ckpt=False. Other keywords go to Policy()."""

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        model = load_causal_lm(path, dtype)
        adapter_path = None
        if adapter or lora:
            import peft

            if adapter:
                model = peft.PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
                adapter_path = str(adapter)
            else:
                model = peft.get_peft_model(model, lora_config(lora))
        configure_linear_attention(model, fla_grad)
        if trainable:
            if grad_ckpt:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                model.enable_input_require_grads()
            model.train()
        else:
            model.eval()
        return cls(model, tok, adapter_path=adapter_path, **kw)

    def save_adapter(self, path: Path | str) -> None:
        self.model.save_pretrained(str(path))

    # ------------------------------------------------------------------------------ properties
    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.get_input_embeddings().weight.dtype

    @property
    def chat_format(self) -> ChatFormat:
        return self.chat.format

    @property
    def has_adapter(self) -> bool:
        return hasattr(self.model, "peft_config")

    @property
    def in_reference(self) -> bool:
        """True inside reference() (generation then targets the base model)."""

        return self._ref_depth > 0

    def _eos_ids(self) -> set[int]:
        ids = set()
        e = getattr(self.tokenizer, "eos_token_id", None)
        ids |= set(e) if isinstance(e, list) else {e}
        gen = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        ids |= set(gen) if isinstance(gen, list) else {gen}
        return {i for i in ids if i is not None}

    def reader(self, **kw):
        """An ActivationReader on this policy (keywords: batch_size, lm_chunk, sort_reads,
        truncate)."""

        from crypsis.models.reading import ActivationReader

        return ActivationReader(self, **kw)

    # ------------------------------------------------------------------------------ adapter modes
    @contextmanager
    def reference(self):
        """The base model (LoRA adapters disabled). Without adapters, the model itself. Inside
        merged() the original base weights are copied back for the block (adapters are already
        disabled there) and re-merged after."""

        if self._merged_active:
            with self._unmerged():
                yield
            return
        dis = getattr(self.model, "disable_adapter", None)
        self._ref_depth += 1
        try:
            with dis() if callable(dis) else nullcontext():
                yield
        finally:
            self._ref_depth -= 1

    @contextmanager
    def merged(self):
        """No-grad READ mode with the LoRA merged into the base weights: each adapted base weight
        holds W0 + scaling * B @ A (fp32 math, cast to the weight dtype) and the adapters are
        disabled. On exit the ORIGINAL weights are copied back from a CPU copy taken once (the base
        never trains), so training is unchanged bit for bit. Reads differ from the adapter path
        only by rounding of the merge. Not for gradients. A no-op without adapters / inside
        reference() / when already merged."""

        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else None
        if base is None or self._ref_depth > 0 or self._merged_active:
            yield
            return
        mods = [m for m in base.modules()
                if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})
                and hasattr(m, "base_layer")]
        if self._w0 is None:  # original base weights, once (pinned host memory on CUDA)
            pin = self.device.type == "cuda"
            self._w0 = [(lambda t: t.pin_memory() if pin else t)(
                m.base_layer.weight.detach().to("cpu", copy=True)) for m in mods]
        self._merged_mods = mods
        self._merge_into(mods)
        self._merged_active = True
        try:
            with self.model.disable_adapter():
                yield
        finally:
            with torch.no_grad():
                for m, w0 in zip(mods, self._w0, strict=True):
                    m.base_layer.weight.copy_(w0, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._merged_active = False

    @staticmethod
    def _merge_into(mods) -> None:
        """Base weight <- original + scaling * B @ A (fp32 math, cast to the weight dtype)."""

        with torch.no_grad():
            for m in mods:
                w = m.base_layer.weight
                d = m.lora_B["default"].weight.float() @ m.lora_A["default"].weight.float()
                w.copy_((w.float() + d * m.scaling["default"]).to(w.dtype))

    @contextmanager
    def _unmerged(self):
        """Inside merged(): the original base weights for the block (the adapters stay disabled,
        so the model IS the base model), merged again on exit."""

        mods = self._merged_mods
        with torch.no_grad():
            for m, w0 in zip(mods, self._w0, strict=True):
                m.base_layer.weight.copy_(w0, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._merged_active = False
        self._ref_depth += 1
        try:
            yield
        finally:
            self._ref_depth -= 1
            self._merge_into(mods)
            self._merged_active = True
