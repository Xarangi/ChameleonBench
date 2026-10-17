from types import SimpleNamespace

from next_chameleons.hf import models


class _FakeTensorDtype:
    pass


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"


class _FakeModel:
    config = SimpleNamespace(use_cache=True)


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeTokenizer()


class _FakeAutoModelForCausalLM:
    kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.kwargs = kwargs
        return _FakeModel()


def test_load_causal_lm_uses_bitsandbytes_config_for_qlora(monkeypatch) -> None:
    fake_transformers = SimpleNamespace(
        AutoTokenizer=_FakeAutoTokenizer,
        AutoModelForCausalLM=_FakeAutoModelForCausalLM,
        BitsAndBytesConfig=_FakeBitsAndBytesConfig,
    )
    fake_torch = SimpleNamespace(
        float16=_FakeTensorDtype(),
        bfloat16=_FakeTensorDtype(),
        float32=_FakeTensorDtype(),
    )
    monkeypatch.setattr(models, "require_transformers", lambda: fake_transformers)
    monkeypatch.setattr(models, "require_torch", lambda: fake_torch)

    loaded = models.load_causal_lm(hf_id="fake/model", quantization="4bit")

    quantization_config = _FakeAutoModelForCausalLM.kwargs["quantization_config"]
    assert loaded.tokenizer.pad_token == "<eos>"
    assert _FakeAutoModelForCausalLM.kwargs.get("load_in_4bit") is None
    assert quantization_config.kwargs["load_in_4bit"] is True
    assert quantization_config.kwargs["bnb_4bit_quant_type"] == "nf4"


def test_extractor_validates_layer_indices_against_hidden_states():
    """selected_layer + offset beyond available hidden states fails clearly."""
    import pytest

    from next_chameleons.hf.activations import HFActivationExtractor

    extractor = HFActivationExtractor.__new__(HFActivationExtractor)
    extractor.selected_layers = [4, 8, 28]
    extractor.hidden_states_offset = 1
    # Qwen-2.5-7B: 28 layers -> 29 hidden states (indices 0..28). Layer 28+1=29 overflows.
    with pytest.raises(ValueError, match=r"selected_layers out of range"):
        extractor._validate_layer_indices(29)
    # In-range case (layer 27 -> index 28) passes.
    extractor.selected_layers = [4, 8, 27]
    extractor._validate_layer_indices(29)


def test_extractor_chat_template_render_wraps_prompt_and_keeps_response():
    """chat_template renders the prompt as a user turn; response stays appended."""
    from types import SimpleNamespace

    from next_chameleons.hf.activations import HFActivationExtractor
    from next_chameleons.hf.data import HFTextExample

    def fake_apply(messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        content = messages[0]["content"]
        return f"<bos><user>{content}</user><model>"

    extractor = HFActivationExtractor.__new__(HFActivationExtractor)
    extractor.chat_template = True
    extractor.loaded = SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=fake_apply))

    example = HFTextExample(
        example_id="e",
        text="Tell me a fact.\nThe sky is blue.",
        label=1,
        domain="d",
        source="s",
        text_hash="h",
        generation_prefix="Tell me a fact.\n",
    )
    text, prefix = extractor._render(example)
    assert prefix == "<bos><user>Tell me a fact.</user><model>"
    assert text == "<bos><user>Tell me a fact.</user><model>The sky is blue."

    # Disabled (default) is a pass-through; missing prefix is a pass-through too.
    extractor.chat_template = False
    assert extractor._render(example) == (example.text, example.generation_prefix)
    extractor.chat_template = True
    no_prefix = HFTextExample(
        example_id="e2", text="raw", label=0, domain="d", source="s", text_hash="h"
    )
    assert extractor._render(no_prefix) == ("raw", "")
