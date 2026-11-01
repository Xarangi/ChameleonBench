"""Unit tests for the crypsis.models API (no model weights)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from crypsis.models.chat import (
    GEMMA4_TURNS,
    QWEN_CHATML,
    SYS_CLOSE,
    SYS_OPEN,
    ChatRenderer,
    View,
    chat_format_for,
    split_system,
)
from crypsis.models.generation import Generation, Generator, HFGenerator, prompt_ids


def test_split_system_and_view():
    conv = [{"role": "user", "content": f"{SYS_OPEN}S{SYS_CLOSE}body"}]
    assert split_system(conv) == [{"role": "system", "content": "S"},
                                  {"role": "user", "content": "body"}]
    assert split_system([{"role": "user", "content": "x"}]) == [{"role": "user", "content": "x"}]
    assert View([1, 2, 3, 4], 1, 1).end == 3


def test_chat_format_and_renderer_rules():
    assert chat_format_for(SimpleNamespace(chat_template="<|turn> <|channel>")) is GEMMA4_TURNS
    assert chat_format_for(SimpleNamespace()) is QWEN_CHATML
    with pytest.raises(ValueError):
        ChatRenderer(SimpleNamespace(), enable_thinking=True, keep_think_history=True)
    r = ChatRenderer(SimpleNamespace(), keep_think_history=True)
    s = r.render([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], False)
    assert s.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\na<|im_end|>\n")
    with pytest.raises(ValueError):
        r.view([{"role": "user", "content": "q"}])


def test_generator_protocol_and_prompt_ids():
    assert isinstance(HFGenerator(None), Generator)
    assert prompt_ids(None, [5, 6]) == [5, 6]
    g = Generation("t", [1], [2], True, 3)
    assert g.as_dict() == {"text": "t", "prompt_ids": [1], "ids": [2], "finished": True,
                           "stop": 3}
