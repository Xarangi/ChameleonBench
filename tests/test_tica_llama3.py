"""Llama-3.1 chat format (11-18): golden render, detection, keep-think prefix property, and the real template."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from next_chameleons.tica.policy import (
    LLAMA3_ASSISTANT_HEAD,
    LLAMA3_HEADERS,
    QWEN_CHATML,
    chat_format_for,
    render_keep_reasoning_llama3,
)

SNAP = Path("/scratch/s/sneheel/hf/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/"
            "a2856192dd7c25b842431f39c179a6c2c2f627d1")
CONV = [{"role": "system", "content": "S"},
        {"role": "user", "content": " u0 "},
        {"role": "assistant", "content": "a0", "reasoning": ""},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"}]
PRE = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
       "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n")
H = "<|start_header_id|>{}<|end_header_id|>\n\n"


def test_llama3_render_golden_and_detection():
    want = (PRE + "S<|eot_id|>" + H.format("user") + "u0<|eot_id|>" + H.format("assistant") + "a0<|eot_id|>"
            + H.format("user") + "u1<|eot_id|>" + H.format("assistant") + "a1<|eot_id|>" + H.format("assistant"))
    assert render_keep_reasoning_llama3(CONV, True, False) == want
    assert render_keep_reasoning_llama3(CONV[1:2], True, False) == PRE + "<|eot_id|>" + H.format("user") + "u0<|eot_id|>" + H.format("assistant")
    tpl = "{{ '<|start_header_id|>' }}{{ '<|eot_id|>' }}"
    assert chat_format_for(SimpleNamespace(chat_template=tpl)) is LLAMA3_HEADERS
    assert chat_format_for(SimpleNamespace(chat_template="{{ '<|im_start|>' }}")) is QWEN_CHATML
    assert LLAMA3_HEADERS.assistant_head == LLAMA3_ASSISTANT_HEAD and LLAMA3_HEADERS.end_of_turn == "<|eot_id|>"
    with pytest.raises(ValueError):
        render_keep_reasoning_llama3([dict(CONV[2], reasoning="r")], False, False)
    # each finished turn = its generation prompt + content + end of turn
    full = render_keep_reasoning_llama3(CONV, False, False)
    for i, m in enumerate(CONV):
        if m["role"] == "assistant":
            assert full.startswith(render_keep_reasoning_llama3(CONV[:i], True, False) + m["content"] + "<|eot_id|>")


@pytest.mark.skipif(not SNAP.is_dir(), reason="Llama-3.1 snapshot not on this machine")
def test_llama3_render_matches_real_template_and_views():
    from transformers import AutoTokenizer

    from next_chameleons.tica.policy import Policy

    tok = AutoTokenizer.from_pretrained(str(SNAP))
    assert chat_format_for(tok) is LLAMA3_HEADERS
    plain = [{"role": m["role"], "content": m["content"]} for m in CONV]
    for conv in (plain, plain[1:], plain[:2], plain[1:4]):
        for gen in (True, False):
            if gen and conv[-1]["role"] == "assistant":
                continue
            assert render_keep_reasoning_llama3(conv, gen, False) == tok.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=gen)
    ids = tok(render_keep_reasoning_llama3(plain, True, False), add_special_tokens=False).input_ids
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
    assert tok("<|eot_id|>", add_special_tokens=False).input_ids == [tok.eos_token_id]
    # segment-wise multi_view tokenisation == per-turn views (no model needed)
    pol = Policy.__new__(Policy)
    pol.tokenizer, pol.enable_thinking, pol.keep_think_history, pol.history_reasoning = tok, False, True, False
    pol.chat_format = LLAMA3_HEADERS
    long = [{"role": "user", "content": "<system>Be brief.</system>\nWhat is 2+2?"},
            {"role": "assistant", "content": "4\n\nBecause 2 + 2 = 4."},
            {"role": "user", "content": "Und auf Deutsch?"},
            {"role": "assistant", "content": "  Vier.  "}]
    mv = pol.multi_view(long)
    assert len(mv.spans) == 2
    for k, i in enumerate((1, 3)):
        v = pol.view(long[: i + 1])
        st, en, te = mv.spans[k]
        assert v.ids == mv.ids[:te] and tok.decode(mv.ids[st:en]) == long[i]["content"].strip()
        assert mv.ids[en] == tok.eos_token_id
        p = pol.encode(pol.render(long[:i], True))
        assert p == mv.ids[:st]  # the generation prompt is a token prefix of the finished conversation
