"""Co-engagement contrastive loss.  (paper §2.2, Eqs. 2-3; README §5)"""
from __future__ import annotations

from typing import Sequence

import torch


def info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.1,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE with in-batch negatives (Eq. 2 with |P(i)| = 1).

    anchor, positive: [B, D]. Row i of `positive` is the positive for row i of `anchor`; all
    other rows of `positive` are its negatives.

        a = anchor / ||anchor||,  p = positive / ||positive||      (Eq. 3; guard zero norms)
        logits = a @ p.T / temperature                            # [B, B]
        loss_i = cross_entropy(logits[i], target=i)
        return mean of loss_i over rows where valid[i] is True (all rows if valid is None)

    If no row is valid, return a zero scalar that is still connected to the graph
    (e.g. anchor.sum() * 0.0) so .backward() doesn't fail.
    """
    raise NotImplementedError("TODO(m04): implement info_nce")


def build_coengagement_neighbors(sequences: Sequence[Sequence[int]], num_items: int, window: int = 1) -> list[list[int]]:
    """For every item id in 0..num_items-1, the sorted list of distinct OTHER items that appear
    within `window` positions of it in any sequence (both directions).

    Example, window=1: sequences [[0, 1, 2], [2, 3]] -> [[1], [0, 2], [1, 3], [2]] (+ [] for unseen items)
    An item never counts as its own neighbour (even if it is repeated in a sequence).
    """
    raise NotImplementedError("TODO(m04): implement build_coengagement_neighbors")


def sample_positives(
    item_ids: torch.Tensor, neighbors: Sequence[Sequence[int]], generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each id in item_ids [B] (a CPU or GPU long tensor), pick one neighbour uniformly at random.

    Returns (pos_ids [B] long, valid [B] bool), both on the same device as item_ids.
    Items without neighbours: pos_ids = the item itself, valid = False.
    All randomness from `generator` (CPU). A python loop is fine. Bonus: vectorize it.
    """
    raise NotImplementedError("TODO(m04): implement sample_positives")
