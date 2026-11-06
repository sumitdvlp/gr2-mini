"""Top-k candidate lists and the paper's 'target must be in the list' filter.  (README §3)"""
from __future__ import annotations

from typing import Sequence

import torch

from gr2mini.m02_data.sequences import SeqExample
from gr2mini.m06_retriever.model import SASRec, encode_histories  # noqa: F401


@torch.no_grad()
def topk_candidates(
    model: SASRec,
    histories: Sequence[Sequence[int]],
    k: int,
    exclude_history: bool = True,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> list[list[int]]:
    """For each history: the k highest-scoring item ids (best first) as python ints.

    * Use model.user_scores(encode_histories(batch, model.max_len)) in eval mode, batch by batch.
    * exclude_history: set the scores of every item in that history (the FULL history, not only
      the last max_len) to -inf before taking the top-k.
    * Restore the model's train/eval mode afterwards.
    """
    raise NotImplementedError("TODO(m06): implement topk_candidates")


def build_candidate_examples(
    users: Sequence[int], examples: Sequence[SeqExample], candidates: Sequence[Sequence[int]]
) -> list[dict]:
    """Keep only the examples whose target is in their candidate list.

    Returns, in input order, dicts:
        {"user": users[i], "history": list(history), "candidates": list(candidates[i]),
         "target": target, "target_rank": 1-based position of target in candidates[i]}
    All values plain python ints/lists (they get written to JSON).
    """
    raise NotImplementedError("TODO(m06): implement build_candidate_examples")
