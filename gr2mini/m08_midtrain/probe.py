"""Probing what mid-training taught the model.  (README "Probes")"""
from __future__ import annotations

import re  # noqa: F401
from collections import Counter  # noqa: F401
from typing import Callable, Sequence

from gr2mini.m02_data.sequences import SeqExample
from gr2mini.m05_semantic_ids.sid_index import SIDIndex

# A generate_fn takes a list of conversations (message lists) and returns one string per conversation.
# run.py passes: lambda convs: [o[0] for o in generate_chat(model, tok, convs, max_new_tokens=...)]
GenerateFn = Callable[[list[list[dict]]], list[str]]


def word_f1(prediction: str, reference: str) -> float:
    """Bag-of-words F1 between lowercased word lists (words = re.findall(r"[a-z0-9]+", text.lower())).

    precision = overlap / #pred words, recall = overlap / #ref words, overlap counts duplicates
    (Counter intersection). Returns 0.0 if either side has no words or overlap is 0.
    """
    raise NotImplementedError("TODO(m08): implement word_f1")


def probe_next_item(
    generate_fn: GenerateFn, index: SIDIndex, examples: Sequence[SeqExample], max_history: int = 10
) -> dict[str, float]:
    """Ask for the next item with sequential_prompt (histories truncated to max_history, as SID strings).

    Returns {"hit@1": fraction whose parsed item (index.item_from_text) == target,
             "valid_sid_rate": fraction whose answer contains a known SID}.
    Call generate_fn ONCE with all conversations.
    """
    raise NotImplementedError("TODO(m08): implement probe_next_item")


def probe_caption(
    generate_fn: GenerateFn, items: Sequence[dict], index: SIDIndex, item_ids: Sequence[int]
) -> dict[str, float]:
    """Ask caption_example's question for each item id; {"title_f1": mean word_f1(answer, item["title"])}.
    Call generate_fn ONCE."""
    raise NotImplementedError("TODO(m08): implement probe_caption")
