"""Sampling G answers for one prompt, and turning them into training tensors.  (README)"""
from __future__ import annotations

from typing import Sequence

import torch

from gr2mini.m07_llm.generate import chat_prompt  # noqa: F401


@torch.no_grad()
def rollout_group(
    model, tokenizer, messages: Sequence[dict], group_size: int, max_new_tokens: int, temperature: float = 1.0
) -> dict:
    """Sample `group_size` answers to ONE prompt.

    * prompt_ids = tokenizer(chat_prompt(tokenizer, messages), add_special_tokens=False) as a [1, Tp] tensor
      on the model's device (one prompt: no padding needed)
    * model.eval(), model.config.use_cache = True; model.generate(prompt, do_sample=True, temperature,
      max_new_tokens, num_return_sequences=group_size, pad_token_id=tokenizer.pad_token_id);
      restore training mode and use_cache afterwards
    Returns {"prompt_ids": LongTensor [Tp] (CPU), "completion_ids": LongTensor [G, Tc] (CPU, the new tokens),
             "texts": [decoded completions with skip_special_tokens=True, stripped]}
    """
    raise NotImplementedError("TODO(m14): implement rollout_group")


def sequence_batch(
    prompt_ids: torch.Tensor, completion_ids: torch.Tensor, comp_mask: torch.Tensor, pad_id: int
) -> dict[str, torch.Tensor]:
    """Concatenate one prompt with its G completions for token_logprobs.

        input_ids      [G, Tp + Tc]  prompt repeated, then the completion; masked-out completion tokens
                                     are replaced by pad_id
        attention_mask [G, Tp + Tc]  1 for the prompt and for completion tokens where comp_mask, else 0
        target_mask    [G, Tp + Tc]  bool: True exactly at completion tokens where comp_mask
    """
    raise NotImplementedError("TODO(m14): implement sequence_batch")
