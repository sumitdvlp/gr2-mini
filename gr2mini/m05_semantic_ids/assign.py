"""From a trained RQ-VAE to unique Semantic IDs.  (paper Eq. 10; README §1-4)

`codes` everywhere is a LongTensor [N_items, L]; row i is the SID of item i.
"""
from __future__ import annotations

import torch

from gr2mini.m01_metrics.metrics import uniqueness
from gr2mini.m04_rqvae.model import RQVAE


def compute_semantic_codes(
    model: RQVAE, embeddings: torch.Tensor, batch_size: int = 4096, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Deterministic codes [N, L] for all items (model.get_codes), returned as a CPU LongTensor.
    Put the model in eval mode first."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        codes = model.get_codes(embeddings, batch_size=batch_size)
    return codes.to(device="cpu", dtype=torch.long)


def randomize_last_levels(
    codes: torch.Tensor, num_random_levels: int, codebook_size: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Eq. 10: return a NEW tensor where the last `num_random_levels` columns are replaced by
    uniform random ints in [0, codebook_size). num_random_levels == 0 -> a plain copy.
    ValueError if num_random_levels < 0 or > L. Must not modify `codes`."""
    num_levels = codes.shape[1]
    if num_random_levels < 0 or num_random_levels > num_levels:
        raise ValueError(f"num_random_levels must be in [0, {num_levels}], got {num_random_levels}")
    out = codes.clone()
    if num_random_levels == 0:
        return out
    rand = torch.randint(
        0, codebook_size, (codes.shape[0], num_random_levels), generator=generator, dtype=torch.long
    )
    out[:, num_levels - num_random_levels :] = rand.to(device=out.device, dtype=out.dtype)
    return out


def find_collisions(codes: torch.Tensor) -> dict[tuple[int, ...], list[int]]:
    """{sid_tuple: [item ids sharing it, ascending]} for every SID used by MORE than one item."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, row in enumerate(codes.tolist()):
        groups.setdefault(tuple(row), []).append(i)
    return {sid: items for sid, items in groups.items() if len(items) > 1}


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
    out = codes.clone()
    num_levels = out.shape[1]
    collisions = find_collisions(out)
    if not collisions:
        return out
    if num_random_levels <= 0 or num_random_levels > num_levels:
        raise ValueError(
            f"SIDs collide; num_random_levels must be in [1, {num_levels}] to re-draw, got {num_random_levels}"
        )
    start = num_levels - num_random_levels
    for _ in range(max_rounds):
        # Keep the lowest id of each group; re-draw the tails of the others.
        redraw = [i for items in collisions.values() for i in items[1:]]
        idx = torch.tensor(redraw, dtype=torch.long, device=out.device)
        rand = torch.randint(
            0, codebook_size, (len(redraw), num_random_levels), generator=generator, dtype=torch.long
        )
        out[idx, start:] = rand.to(device=out.device, dtype=out.dtype)
        collisions = find_collisions(out)
        if not collisions:
            return out
    n_left = sum(len(v) for v in collisions.values())
    raise RuntimeError(f"{n_left} items still collide after {max_rounds} rounds")


def codebook_utilization(codes: torch.Tensor, codebook_size: int) -> list[float]:
    """Per level: (number of distinct codes used) / codebook_size."""
    return [codes[:, l].unique().numel() / codebook_size for l in range(codes.shape[1])]


def prefix_stats(codes: torch.Tensor) -> list[dict]:
    """For each prefix length k = 1..L:
        {"level": k, "distinct": #distinct length-k prefixes,
         "largest_group": max #items sharing one prefix, "mean_group": N / distinct}
    """
    n, num_levels = codes.shape
    rows = codes.tolist()
    stats = []
    for k in range(1, num_levels + 1):
        counts: dict[tuple[int, ...], int] = {}
        for row in rows:
            key = tuple(row[:k])
            counts[key] = counts.get(key, 0) + 1
        distinct = len(counts)
        stats.append(
            {
                "level": k,
                "distinct": distinct,
                "largest_group": max(counts.values()) if counts else 0,
                "mean_group": n / distinct if distinct else 0.0,
            }
        )
    return stats


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
    semantic = compute_semantic_codes(model, embeddings, device=device)
    randomized = randomize_last_levels(semantic, num_random_levels, codebook_size, generator)
    final = resolve_collisions(randomized, num_random_levels, codebook_size, generator, max_rounds)
    report = {
        "uniqueness_semantic": uniqueness(semantic.tolist()),
        "uniqueness_random_last": uniqueness(randomized.tolist()),
        "uniqueness_final": uniqueness(final.tolist()),
        "utilization": codebook_utilization(semantic, codebook_size),
        "num_redrawn": int((randomized != final).any(dim=1).sum().item()),
    }
    return final, report
