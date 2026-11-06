"""Item-alignment tasks: chat examples that interleave SIDs with natural language.  (README "The tasks")

A chat example is a list of messages: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
"""
from __future__ import annotations

from typing import Sequence

from gr2mini.m05_semantic_ids.sid_index import SIDIndex


def caption_example(item: dict, sid: str) -> list[dict]:
    """user: asks what product `sid` is (must contain `sid`);  assistant: exactly item["text"]."""
    raise NotImplementedError("TODO(m08): implement caption_example")


def reverse_example(item: dict, sid: str) -> list[dict]:
    """user: shows item["text"] and asks for its Semantic ID;  assistant: exactly `sid`."""
    raise NotImplementedError("TODO(m08): implement reverse_example")


def sequential_prompt(history_sids: Sequence[str]) -> str:
    """The user turn for next-item prediction: the history as a numbered list, oldest first
    ("1. <sid>\\n2. <sid>..."), then a request to answer with the Semantic ID of the next product.
    m08's probe and the tests reuse this, so keep it a separate function."""
    raise NotImplementedError("TODO(m08): implement sequential_prompt")


def sequential_example(history_sids: Sequence[str], target_sid: str) -> list[dict]:
    """user: sequential_prompt(history_sids);  assistant: exactly target_sid."""
    raise NotImplementedError("TODO(m08): implement sequential_example")


def build_midtrain_examples(
    items: Sequence[dict], index: SIDIndex, sequences: Sequence[Sequence[int]], cfg: dict, seed: int = 0
) -> list[list[dict]]:
    """All mid-training examples, shuffled with random.Random(seed), capped at cfg["max_examples"].

    cfg = the `midtrain:` section. Keys: caption (bool), reverse (bool), sequential_per_user (int),
    max_history (int), max_examples (int | None).
        * caption / reverse: one example per item (when enabled)
        * sequential: for each user, s = seq[:-2] (NEVER the valid/test targets). For the last
          `sequential_per_user` positions t of s with t >= 1:
              history = s[:t][-max_history:], target = s[t]
          (use SIDIndex.sid_str for every item)
    """
    raise NotImplementedError("TODO(m08): implement build_midtrain_examples")
