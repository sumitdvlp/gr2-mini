"""From a trained RQ-VAE to unique Semantic IDs.  (paper Eq. 10; README §1-4)

`codes` everywhere is a LongTensor [N_items, L]; row i is the SID of item i.
"""
from __future__ import annotations

import torch

from gr2mini.m04_rqvae.model import RQVAE


def compute_semantic_codes(
    model: RQVAE, embeddings: torch.Tensor, batch_size: int = 4096, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Deterministic codes [N, L] for all items (model.get_codes), returned as a CPU LongTensor.
    Put the model in eval mode first."""
    raise NotImplementedError("TODO(m05): implement compute_semantic_codes")


def randomize_last_levels(
    codes: torch.Tensor, num_random_levels: int, codebook_size: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Eq. 10: return a NEW tensor where the last `num_random_levels` columns are replaced by
    uniform random ints in [0, codebook_size). num_random_levels == 0 -> a plain copy.
    ValueError if num_random_levels < 0 or > L. Must not modify `codes`."""
    raise NotImplementedError("TODO(m05): implement randomize_last_levels")


def find_collisions(codes: torch.Tensor) -> dict[tuple[int, ...], list[int]]:
    """{sid_tuple: [item ids sharing it, ascending]} for every SID used by MORE than one item."""
    raise NotImplementedError("TODO(m05): implement find_collisions")


def resolve_collisions(
    codes: torch.Tensor,
    num_random_levels: int,
    codebook_size: int,
    generator: torch.Generator | None = None,
    max_rounds: int = 100,
) -> torch.Tensor:
    """Return a NEW tensor with all SIDs unique (README §3).

    Each round: for every collision group, keep the lowest item id as is and re-draw the last
    `num_random_levels` codes of the other items. Repeat until there are no collisions.
        * Items that are not in any collision group must never change.
        * No collisions -> return an unchanged copy.
        * Collisions but num_random_levels == 0 -> ValueError (nothing we're allowed to re-draw).
        * Still colliding after max_rounds -> RuntimeError.
    """
    raise NotImplementedError("TODO(m05): implement resolve_collisions")


def codebook_utilization(codes: torch.Tensor, codebook_size: int) -> list[float]:
    """Per level: (number of distinct codes used) / codebook_size."""
    raise NotImplementedError("TODO(m05): implement codebook_utilization")


def prefix_stats(codes: torch.Tensor) -> list[dict]:
    """For each prefix length k = 1..L:
        {"level": k, "distinct": #distinct length-k prefixes,
         "largest_group": max #items sharing one prefix, "mean_group": N / distinct}
    """
    raise NotImplementedError("TODO(m05): implement prefix_stats")


def assign_semantic_ids(
    model: RQVAE,
    embeddings: torch.Tensor,
    num_random_levels: int,
    codebook_size: int,
    generator: torch.Generator | None = None,
    max_rounds: int = 100,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, dict]:
    """The full pipeline: compute_semantic_codes -> randomize_last_levels -> resolve_collisions.

    Returns (final_codes, report) with report =
        {"uniqueness_semantic": m01 uniqueness of the deterministic codes,
         "uniqueness_random_last": uniqueness after randomize_last_levels,
         "uniqueness_final": uniqueness of final_codes (1.0),
         "utilization": codebook_utilization of the DETERMINISTIC codes,
         "num_redrawn": number of items whose codes differ between the randomized and final codes}
    """
    raise NotImplementedError("TODO(m05): implement assign_semantic_ids")
