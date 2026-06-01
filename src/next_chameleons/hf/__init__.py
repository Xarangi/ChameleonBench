"""Optional Hugging Face real-run integrations."""

from next_chameleons.hf.activations import HFActivationExtractor
from next_chameleons.hf.data import HFTextExample, load_hf_text_examples
from next_chameleons.hf.training import run_real_chameleon_training

__all__ = [
    "HFActivationExtractor",
    "HFTextExample",
    "load_hf_text_examples",
    "run_real_chameleon_training",
]
