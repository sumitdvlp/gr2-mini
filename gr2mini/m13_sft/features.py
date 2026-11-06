"""SFT features with separate reasoning / output weights (Eq. 13).  (README "Character spans")"""
from __future__ import annotations

import json  # noqa: F401
from typing import Sequence

from gr2mini.m07_llm.generate import chat_prompt  # noqa: F401


def explanation_char_span(content: str) -> tuple[int, int]:
    """(start, end) character span of the explanation VALUE inside an assistant_json() string,
    including the surrounding double quotes, so content[start:end] is a JSON string literal.

    Find it via json.loads(content)["explanation"] -> json.dumps(value, ensure_ascii=False), located
    right after the "explanation" key. ValueError if the content isn't such JSON.
    """
    raise NotImplementedError("TODO(m13): implement explanation_char_span")


def sft_feature(
    tokenizer, messages: Sequence[dict], lambda_r: float, lambda_o: float, max_len: int
) -> dict | None:
    """{"input_ids": list[int], "loss_weights": list[float]} for a trace conversation
    (system, user, assistant=assistant_json(...)), weighted per README. None if longer than max_len.

    * prompt = chat_prompt(messages[:-1], add_generation_prompt=True);
      full = chat_prompt(messages, add_generation_prompt=False); ValueError unless full startswith prompt
    * the assistant content starts at full.index(content, len(prompt)); shift explanation_char_span by that
    * tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    * weight 0.0 if token end <= len(prompt); lambda_r if it overlaps the explanation span; else lambda_o
    """
    raise NotImplementedError("TODO(m13): implement sft_feature")
