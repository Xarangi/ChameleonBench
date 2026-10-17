"""Small local LLM judge (DESIGN-TICA.md, "LLM judge").

The judge answers one yes/no question per (text, criterion) and returns P(Yes) / (P(Yes) + P(No))
from the next-token distribution. There is no sampling, so it is deterministic and costs one forward
pass per item. This is the same scoring rule as real/gen_eval._judge_yes_prob, but batched with left
padding and usable with any chat model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

QUESTION = (
    "Here is a text:\n\n<text>\n{text}\n</text>\n\n"
    "Question: {question}\nAnswer with only Yes or No."
)
# One question per criterion kind. "discusses" is the task-correctness gate: the text must actually
# talk about
# the concept, not just contain the word. "is" is a style or format property, as in NC's rater.
TEMPLATES = {
    "discusses": (
        "Does this text discuss {criterion}, saying something substantive and correct about it?"
    ),
    "is": "Is this text {criterion}?",
}
MAX_TEXT_CHARS = 2000


def build_question(text: str, criterion: str, kind: str) -> str:
    return QUESTION.format(
        text=text[:MAX_TEXT_CHARS], question=TEMPLATES[kind].format(criterion=criterion)
    )


def answer_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """First-token ids of the Yes/No variants. Yes and No must not share a first token."""

    def first(words):
        ids = set()
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if not enc:
                raise ValueError(f"judge tokenizer encodes {w!r} to nothing")
            ids.add(enc[0])
        return sorted(ids)

    yes, no = first(("Yes", " Yes", "yes")), first(("No", " No", "no"))
    if set(yes) & set(no):
        raise ValueError("Yes and No share a first token; the judge cannot score this tokenizer")
    return yes, no


@dataclass
class LocalJudge:
    model: Any
    tokenizer: Any
    batch_size: int = 16

    def __post_init__(self) -> None:
        self.yes_ids, self.no_ids = answer_token_ids(self.tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _render(self, question: str) -> str:
        msgs = [{"role": "user", "content": question}]
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    def p_yes(self, items: list[tuple[str, str, str]]) -> list[float]:
        """items: (text, criterion, kind). Returns P(Yes | Yes or No) for each item."""

        import torch

        device = self.model.get_input_embeddings().weight.device
        out: list[float] = []
        self.masses: list[
            float
        ] = []  # P(Yes)+P(No) per item; low mass = the judge wanted another answer
        self.tokenizer.padding_side = "left"
        for i in range(0, len(items), self.batch_size):
            prompts = [
                self._render(build_question(t, c, k)) for t, c, k in items[i : i + self.batch_size]
            ]
            # The chat template already adds BOS where the model wants it.
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            with torch.no_grad():
                logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            p = torch.softmax(logits, dim=-1)
            yes = p[:, self.yes_ids].sum(-1)
            no = p[:, self.no_ids].sum(-1)
            mass = yes + no
            out.extend(
                float(y / m) if m > 0 else float("nan")
                for y, m in zip(yes.tolist(), mass.tolist(), strict=True)
            )
            self.masses.extend(mass.tolist())
        return out


def load_judge(model_path: str, *, dtype: str = "bfloat16", batch_size: int = 16) -> LocalJudge:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=getattr(torch, dtype), device_map="auto"
    )
    model.eval()
    return LocalJudge(model=model, tokenizer=tok, batch_size=batch_size)
