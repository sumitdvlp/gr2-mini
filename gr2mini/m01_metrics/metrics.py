"""Ranking metrics (paper App. G, Eqs. 30-32) and SID collision metrics (Eqs. 33-35).

Conventions used everywhere in this repo:
    * a "ranked list" is a sequence of item ids, best first
    * ranks are 1-based (the top item has rank 1)
    * there is exactly ONE relevant (target) item per user
"""
from __future__ import annotations

from typing import Iterable, Sequence
import math

def rank_of(target: int, ranked: Sequence[int]) -> int | None:
    """1-based position of `target` in `ranked`, or None if it is absent.

    rank_of(3, [5, 3, 9]) == 2
    """
    try:
        return ranked.index(target) + 1
    except ValueError:
        return None


def recall_at_k(ranked_lists: Sequence[Sequence[int]], targets: Sequence[int], k: int) -> float:
    """Eq. 30: fraction of users whose target appears in the first `k` items of their list.

    Raises ValueError if: the two sequences have different lengths, they are empty, or k < 1.
    Lists shorter than k are fine (just look at what's there).
    """
    if len(ranked_lists) != len(targets):
        raise ValueError(f"ranked_lists and targets must have the same length, got {len(ranked_lists)} vs {len(targets)}")
    if len(ranked_lists) == 0:
        raise ValueError("ranked_lists and targets must not be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    hits = sum(1 for ranked, target in zip(ranked_lists, targets) if rank_of(target, ranked[:k]) is not None)
    return hits / len(ranked_lists)


def ndcg_at_k(ranked_lists: Sequence[Sequence[int]], targets: Sequence[int], k: int) -> float:
    """Eqs. 31-32 with one relevant item: mean over users of 1/log2(rank+1) if rank <= k else 0.

    Same validation rules as `recall_at_k`.
    """
    if len(ranked_lists) != len(targets):
        raise ValueError(f"ranked_lists and targets must have the same length, got {len(ranked_lists)} vs {len(targets)}")
    if len(ranked_lists) == 0:
        raise ValueError("ranked_lists and targets must not be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    total = 0.0
    for ranked, target in zip(ranked_lists, targets):
        rank = rank_of(target, ranked)
        if rank is not None and rank <= k:
            total += 1 / (math.log2(rank + 1))
    return total / len(ranked_lists)


def ranking_report(
    ranked_lists: Sequence[Sequence[int]],
    targets: Sequence[int],
    recall_ks: Iterable[int] = (1, 5, 9),
    ndcg_ks: Iterable[int] = (5, 10),
) -> dict[str, float]:
    """All metrics at once, keyed like the paper's tables:

    {"R@1": ..., "R@5": ..., "R@9": ..., "N@5": ..., "N@10": ...}
    (recall keys first, in the given order, then ndcg keys)
    """
    metrics = {}
    ranked_lists = list(ranked_lists)
    targets = list(targets)
    for k in recall_ks:
        metrics[f"R@{k}"] = recall_at_k(ranked_lists, targets, k)
    for k in ndcg_ks:
        metrics[f"N@{k}"] = ndcg_at_k(ranked_lists, targets, k)
    return metrics


def collision_rate(sids: Iterable[Iterable[int]]) -> float:
    """Eqs. 33-34: |items whose SID is shared with at least one other item| / |items|.

    `sids` is anything iterable of rows: list of tuples, list of lists, a 2-D numpy array or
    a 2-D torch LongTensor. Convert each row to a tuple of python ints before comparing.
    Raises ValueError if there are no items.
    """
    seen: set[tuple[int, ...]] = set()
    collisions: set[tuple[int, ...]] = set()
    n_items = 0
    for row in sids:
        sid = tuple(int(x) for x in row)
        if sid in seen:
            collisions.add(sid)
        else:
            seen.add(sid)
        n_items += 1
    if n_items == 0:
        raise ValueError("collision_rate: no items")
    return sum(1 for row in sids if tuple(int(x) for x in row) in collisions) / n_items


def uniqueness(sids: Iterable[Iterable[int]]) -> float:
    """Eq. 35: 1 - collision_rate(sids)."""
    return 1.0 - collision_rate(sids)
