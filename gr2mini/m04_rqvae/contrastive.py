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
    # Normalize to unit vectors, guard against zero norms
    anchor_norm = torch.norm(anchor, dim=1, keepdim=True).clamp_min(1e-8)
    positive_norm = torch.norm(positive, dim=1, keepdim=True).clamp_min(1e-8)
    a = anchor / anchor_norm
    p = positive / positive_norm

    # Compute logits: [B, B]
    logits = a @ p.T / temperature

    # Target labels: row i should match column i
    targets = torch.arange(logits.shape[0], device=logits.device)

    # Compute cross-entropy loss for each row
    loss_per_row = torch.nn.functional.cross_entropy(logits, targets, reduction="none")

    if valid is not None:
        # Only consider valid rows
        loss_per_row = loss_per_row[valid]

    if loss_per_row.numel() == 0:
        # No valid rows; return a zero scalar connected to the graph
        return anchor.sum() * 0.0

    return loss_per_row.mean()


def build_coengagement_neighbors(sequences: Sequence[Sequence[int]], num_items: int, window: int = 1) -> list[list[int]]:
    """For every item id in 0..num_items-1, the sorted list of distinct OTHER items that appear
    within `window` positions of it in any sequence (both directions).

    Example, window=1: sequences [[0, 1, 2], [2, 3]] -> [[1], [0, 2], [1, 3], [2]] (+ [] for unseen items)
    An item never counts as its own neighbour (even if it is repeated in a sequence).
    """
    neighbors = [[] for _ in range(num_items)]
    for seq in sequences:
        for i, item in enumerate(seq):
            start = max(0, i - window)
            end = min(len(seq), i + window + 1)
            for j in range(start, end):
                if j != i:
                    neighbor_item = seq[j]
                    if neighbor_item not in neighbors[item]:
                        neighbors[item].append(neighbor_item)
    return neighbors


def sample_positives(
    item_ids: torch.Tensor, neighbors: Sequence[Sequence[int]], generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each id in item_ids [B] (a CPU or GPU long tensor), pick one neighbour uniformly at random.

    Returns (pos_ids [B] long, valid [B] bool), both on the same device as item_ids.
    Items without neighbours: pos_ids = the item itself, valid = False.
    All randomness from `generator` (CPU). A python loop is fine. Bonus: vectorize it.
    """
    pos_ids = torch.empty_like(item_ids)
    valid = torch.empty(item_ids.shape[0], dtype=torch.bool, device=item_ids.device)

    for i, item_id in enumerate(item_ids.tolist()):
        neighbor_list = neighbors[item_id]
        if neighbor_list:
            # Sample a random neighbor
            pos_ids[i] = neighbor_list[torch.randint(0, len(neighbor_list), (1,), generator=generator).item()]
            valid[i] = True
        else:
            # No neighbors; use the item itself and mark as invalid
            pos_ids[i] = item_id
            valid[i] = False

    return pos_ids, valid
