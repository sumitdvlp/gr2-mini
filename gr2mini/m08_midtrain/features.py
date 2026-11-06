"""Chat example -> token ids + per-token loss weights, and batching.  (README "Loss masking")"""
from __future__ import annotations

from typing import Sequence

import torch

from gr2mini.m07_llm.generate import chat_prompt  # noqa: F401


def tokenize_chat(tokenizer, messages: Sequence[dict], max_len: int) -> dict | None:
    """{"input_ids": list[int], "loss_weights": list[float]} for a chat whose LAST message is the
    assistant answer to learn.

        prompt = chat_prompt(tokenizer, messages[:-1], add_generation_prompt=True)
        full   = chat_prompt(tokenizer, messages, add_generation_prompt=False)
        * ValueError if full doesn't start with prompt (as strings), or if the prompt's token ids
          (tokenize prompt alone, add_special_tokens=False) aren't a prefix of full's token ids.
        * weights: 0.0 for the prompt tokens, 1.0 for every token after them.
        * Return None if len(input_ids) > max_len (we drop, not truncate: truncating would cut answers).
    """
    raise NotImplementedError("TODO(m08): implement tokenize_chat")


def collate(features: Sequence[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """RIGHT-pad a list of features to the longest one.

    Returns {"input_ids": long [B, T] (pad_id), "attention_mask": long [B, T] (1 real / 0 pad),
             "loss_weights": float32 [B, T] (0.0 on padding)}
    """
    raise NotImplementedError("TODO(m08): implement collate")
