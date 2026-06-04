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
