"""Ranking metrics (paper App. G, Eqs. 30-32) and SID collision metrics (Eqs. 33-35).

Conventions used everywhere in this repo:
    * a "ranked list" is a sequence of item ids, best first
    * ranks are 1-based (the top item has rank 1)
    * there is exactly ONE relevant (target) item per user
"""
from __future__ import annotations

from typing import Iterable, Sequence


def rank_of(target: int, ranked: Sequence[int]) -> int | None:
    """1-based position of `target` in `ranked`, or None if it is absent.

    rank_of(3, [5, 3, 9]) == 2
    """
    raise NotImplementedError("TODO(m01): implement rank_of")


def recall_at_k(ranked_lists: Sequence[Sequence[int]], targets: Sequence[int], k: int) -> float:
    """Eq. 30: fraction of users whose target appears in the first `k` items of their list.

    Raises ValueError if: the two sequences have different lengths, they are empty, or k < 1.
    Lists shorter than k are fine (just look at what's there).
    """
    raise NotImplementedError("TODO(m01): implement recall_at_k")


def ndcg_at_k(ranked_lists: Sequence[Sequence[int]], targets: Sequence[int], k: int) -> float:
    """Eqs. 31-32 with one relevant item: mean over users of 1/log2(rank+1) if rank <= k else 0.

    Same validation rules as `recall_at_k`.
    """
    raise NotImplementedError("TODO(m01): implement ndcg_at_k")


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
    raise NotImplementedError("TODO(m01): implement ranking_report")


def collision_rate(sids: Iterable[Iterable[int]]) -> float:
    """Eqs. 33-34: |items whose SID is shared with at least one other item| / |items|.

    `sids` is anything iterable of rows: list of tuples, list of lists, a 2-D numpy array or
    a 2-D torch LongTensor. Convert each row to a tuple of python ints before comparing.
    Raises ValueError if there are no items.
    """
    raise NotImplementedError("TODO(m01): implement collision_rate")


def uniqueness(sids: Iterable[Iterable[int]]) -> float:
    """Eq. 35: 1 - collision_rate(sids)."""
    raise NotImplementedError("TODO(m01): implement uniqueness")
