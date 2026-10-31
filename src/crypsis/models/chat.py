"""Chat rendering and token views (from tica/policy.py and answer_format.split_system).

A *view* is what the model computes while writing the final message of a conversation:
- ids = tokenize(generation prompt of conv[:-1]) + tokenize(conv[-1].content) + end-of-turn token;
- the content span is [start, end).
Templates can render a finished turn differently from the generation prompt: Qwen3.5 and Gemma-4
drop the empty think block. So a view is NOT a slice of the rendered conversation. Earlier turns
appear as the template renders history, exactly as in real multi-turn use.

This module owns the system-prompt split (`split_system`), so the activation reader no longer
depends on the episode layer (plan §12.1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Tagged layout (10-17a): a system part carried in front of the first user turn, between these
# markers, becomes a system message when rendered.
SYS_OPEN, SYS_CLOSE = "\x1esystem\x1e", "\x1e/system\x1e"


def split_system(conv: list[dict]) -> list[dict]:
    """Lift a SYS_OPEN...SYS_CLOSE prefix of the first user message into a system message."""

    if not conv or conv[0]["role"] != "user" or not conv[0]["content"].startswith(SYS_OPEN):
        return conv
    head, _, body = conv[0]["content"][len(SYS_OPEN):].partition(SYS_CLOSE)
    return [{"role": "system", "content": head}, dict(conv[0], content=body), *conv[1:]]


@dataclass
class View:
    """Token ids of one final assistant message in context; [start, end) is the probe read span."""

    ids: list[int]
    start: int  # first content token
    n_tail: int = 0  # end-of-turn token(s) after the content: in the NLL, not in the probe read

    @property
    def end(self) -> int:
        return len(self.ids) - self.n_tail


@dataclass
class MultiView:
    """A whole conversation tokenised once (keep-think history), with every assistant turn's
    (content start, content end, end of its end-of-turn token). Turn t's per-turn view is the
    prefix ids[:tail_end_t], so one forward reads every turn exactly as the per-turn views do."""

    ids: list[int]
    spans: list[tuple[int, int, int]]


def render_keep_reasoning(conv: list[dict], add_generation_prompt: bool, thinking: bool) -> str:
    """Qwen3.5 / Qwen3 ChatML rendering of a plain system/user/assistant conversation that KEEPS
    every assistant turn's reasoning, in the exact form the template uses for the turns it keeps
    ('<think>\\n' + reasoning + '\\n</think>\\n\\n' + content). Assistant turns without a
    "reasoning" key render as the template renders history (no think block)."""

    out = []
    for i, m in enumerate(conv):
        role, content = m["role"], m["content"].strip()
        if role == "system":
            if i:
                raise ValueError("system message must be first")
            out.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            r = m.get("reasoning")
            body = content if r is None else f"<think>\n{r.strip()}\n</think>\n\n{content}"
            out.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
        else:
            raise ValueError(f"unsupported role {role!r}")
    if add_generation_prompt:
        head = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
        out.append("<|im_start|>assistant\n" + head)
    return "".join(out)


GEMMA4_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"  # the thinking-off generation prompt's block


def render_keep_reasoning_gemma4(conv: list[dict], add_generation_prompt: bool,
                                 thinking: bool) -> str:
    """Gemma-4 analogue of render_keep_reasoning (its chat_template.jinja, text messages only):
    BOS, an optional system turn ('<|think|>' first when thinking), user / model turns. An assistant
    turn with a "reasoning" key keeps it in the channel form; an EMPTY reasoning renders as the
    thinking-off generation prompt's empty block, so a keep-think conversation contains each turn
    exactly as its generation prompt + content. Turns without the key render as history."""

    out = ["<bos>"]
    msgs = list(conv)
    system = None
    if msgs and msgs[0]["role"] == "system":
        system = msgs.pop(0)["content"].strip()
    if thinking or system is not None:
        out.append("<|turn>system\n" + ("<|think|>\n" if thinking else "") + (system or "")
                   + "<turn|>\n")
    for m in msgs:
        role, content = m["role"], m["content"].strip()
        if role == "user":
            out.append(f"<|turn>user\n{content}<turn|>\n")
        elif role == "assistant":
            r = m.get("reasoning")
            if r is None:
                body = content
            elif r.strip():
                body = f"<|channel>thought\n{r.strip()}\n<channel|>{content}"
            else:
                body = GEMMA4_EMPTY_THOUGHT + content
            out.append(f"<|turn>model\n{body}<turn|>\n")
        elif role == "system":
            raise ValueError("system message must be first")
        else:
            raise ValueError(f"unsupported role {role!r}")
    if add_generation_prompt:
        out.append("<|turn>model\n" + ("" if thinking else GEMMA4_EMPTY_THOUGHT))
    return "".join(out)


@dataclass(frozen=True)
class ChatFormat:
    """What the keep-think render and multi_view need from a chat template: the render itself, the
    text before every assistant content (thinking off) and the end-of-turn marker after it."""

    name: str
    render: Callable[[list[dict], bool, bool], str]  # (conv, add_generation_prompt, thinking)
    assistant_head: str
    end_of_turn: str


QWEN_CHATML = ChatFormat("qwen_chatml", render_keep_reasoning,
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_end|>")
GEMMA4_TURNS = ChatFormat("gemma4", render_keep_reasoning_gemma4,
                          "<|turn>model\n" + GEMMA4_EMPTY_THOUGHT, "<turn|>")


def chat_format_for(tokenizer) -> ChatFormat:
    """GEMMA4_TURNS if the tokenizer's chat template is Gemma-4's turn format, else QWEN_CHATML
    (tokenizers without a template, e.g. test char tokenizers, stay on ChatML)."""

    tpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tpl, dict):
        tpl = " ".join(str(v) for v in tpl.values())
    if isinstance(tpl, str) and "<|turn>" in tpl and "<|channel>" in tpl:
        return GEMMA4_TURNS
    return QWEN_CHATML


class ChatRenderer:
    """Renders conversations with a tokenizer's chat template and builds token views.

    enable_thinking: the template's thinking switch (generation prompts).
    history_reasoning: keep each earlier assistant turn's "reasoning" in the rendered history (the
        Qwen3.5 template drops reasoning before the last user message).
    keep_think_history: render EVERY assistant turn with the empty think block the generation prompt
        uses, so a whole conversation contains each turn exactly as its per-turn view did and one
        forward reads all turns (MultiView, single-pass training). Needs enable_thinking=False.
    """

    def __init__(self, tokenizer, *, enable_thinking: bool = False,
                 history_reasoning: bool = False, keep_think_history: bool = False,
                 chat_format: ChatFormat | None = None) -> None:
        if keep_think_history and enable_thinking:
            raise ValueError("keep_think_history needs enable_thinking=False")
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.history_reasoning = history_reasoning
        self.keep_think_history = keep_think_history
        self.format = chat_format or chat_format_for(tokenizer)

    def render(self, conv: list[dict], add_generation_prompt: bool) -> str:
        """The chat-template text of `conv` (system part of the first user turn split off)."""

        conv = split_system(conv)
        if self.keep_think_history:
            conv = [
                dict(m, reasoning=m.get("reasoning") or "") if m["role"] == "assistant" else m
                for m in conv
            ]
            return self.format.render(conv, add_generation_prompt, self.enable_thinking)
        if self.history_reasoning and any(
            m["role"] == "assistant" and m.get("reasoning") is not None for m in conv
        ):
            return self.format.render(conv, add_generation_prompt, self.enable_thinking)
        return self.tokenizer.apply_chat_template(
            [{"role": m["role"], "content": m["content"]} for m in conv],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def view(self, conv: list[dict]) -> View:
        """ids = generation prompt + content + the first token of the template's end-of-turn
        marker. The marker is part of the LM target (so training keeps the model stopping after
        the text) but not of the probe read."""

        if conv[-1]["role"] != "assistant":
            raise ValueError("a view needs a final assistant message")
        if self.keep_think_history:  # a prefix of the whole-conversation tokenisation
            mv = self.multi_view(conv)
            st, en, te = mv.spans[-1]
            return View(mv.ids[:te], st, te - en)
        p = self.encode(self.render(conv[:-1], True))
        tail = self.turn_end(conv)
        return View(p + self.encode(conv[-1]["content"]) + tail, len(p), len(tail))

    def multi_view(self, conv: list[dict]) -> MultiView:
        """keep_think_history only. Segment-wise tokenisation (template text / each assistant
        content separately) so every turn's span and prefix are identical to its per-turn view."""

        if not self.keep_think_history:
            raise ValueError("multi_view needs keep_think_history=True")
        full = self.render(conv, False)
        head = self.format.assistant_head
        end_tok = self.encode(self.format.end_of_turn)[:1]
        ids: list[int] = []
        spans: list[tuple[int, int, int]] = []
        pos = 0
        for m in conv:
            if m["role"] != "assistant":
                continue
            a = full.find(head, pos)
            if a < 0:
                raise ValueError("assistant turn not found in keep-think render")
            k = a + len(head)
            content = m["content"].strip()
            if full[k : k + len(content)] != content:
                raise ValueError("assistant content misaligned in keep-think render")
            ids += self.encode(full[pos:k])
            st = len(ids)
            ids += self.encode(content)
            en = len(ids)
            spans.append((st, en, en + len(end_tok)))
            pos = k + len(content)
        rest = self.encode(full[pos:])
        if end_tok and rest[: len(end_tok)] != end_tok:
            raise ValueError("end-of-turn token not where expected")
        return MultiView(ids + rest, spans)

    def turn_end(self, conv: list[dict]) -> list[int]:
        """First token of what the template writes right after the final assistant content."""

        content = conv[-1]["content"]
        full = self.render(conv, False)
        k = full.rfind(content)
        if not content or k < 0:
            return []
        rest = full[k + len(content) :]
        return self.encode(rest)[:1] if rest else []

    def prompt_ids(self, conv: list[dict]) -> list[int]:
        """Generation-prompt ids for a conversation ending in a user message."""

        return self.encode(self.render(conv, True))


def chat_prompt_ids(tokenizer, content: str, *, enable_thinking: bool | None = None) -> list[int]:
    """Generation-prompt ids for ONE user message, as the eval scripts build them
    (benchmarks.capability._chat_prompt_ids): system part split off, then the tokenizer's own
    chat template with tokenize=True. enable_thinking=None leaves the template default (the eval
    scripts pass False through a proxy tokenizer)."""

    messages = split_system([{"role": "user", "content": content}])
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False, **kw
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                             **kw)
