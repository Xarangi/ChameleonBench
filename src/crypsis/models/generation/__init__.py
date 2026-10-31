"""Generation backends behind one protocol: HFGenerator (in-process) and VLLMGenerator
(colocated vLLM server subprocess; optional, lazily imported)."""

from crypsis.models.chat import chat_prompt_ids
from crypsis.models.generation.base import Generation, Generator, Prompt, prompt_ids
from crypsis.models.generation.hf import HFGenerator, chat_eos_ids, generate_batched

__all__ = ["Generation", "Generator", "HFGenerator", "Prompt", "VLLMGenerator", "VLLMServer",
           "chat_eos_ids", "chat_prompt_ids", "generate_batched", "prompt_ids"]


def __getattr__(name: str):
    if name in ("VLLMGenerator", "VLLMServer"):
        from crypsis.models.generation import vllm

        return getattr(vllm, name)
    raise AttributeError(name)
