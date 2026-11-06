"""Interactions -> filtered, chronological, integer-id user sequences and splits.  (README §1-3)

An "interaction" is a tuple (user: str, item: str, timestamp: int).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

Interaction = tuple[str, str, int]


def k_core_filter(interactions: Sequence[Interaction], k: int) -> list[Interaction]:
    """Iteratively remove users and items with fewer than `k` interactions until stable.

    * Counts are numbers of interaction ROWS (a repeated (user, item) pair counts twice).
    * The surviving interactions keep their ORIGINAL relative order.
    * k <= 1 returns a copy of the input.
    """
    raise NotImplementedError("TODO(m02): implement k_core_filter")


def subsample_users(interactions: Sequence[Interaction], max_users: int | None, seed: int) -> list[Interaction]:
    """Keep only the interactions of a random subset of `max_users` users.

    * max_users None, or >= the number of users -> return a copy of the input.
    * Choose users with random.Random(seed).sample(sorted(unique_users), max_users) so the
      choice is deterministic for a given seed.
    * Keep the original order of the surviving interactions.
    """
    raise NotImplementedError("TODO(m02): implement subsample_users")


def build_sequences(interactions: Sequence[Interaction]) -> dict[str, list[str]]:
    """Group interactions by user into chronologically sorted item lists.

    Ties in timestamp keep their original file order (use a STABLE sort).
    """
    raise NotImplementedError("TODO(m02): implement build_sequences")


def remap_ids(sequences: dict[str, list[str]]) -> tuple[list[list[int]], list[str], list[str]]:
    """Map string ids to dense ints, deterministically.

    Returns (int_sequences, user_ids, item_asins) where
        * user_ids   = sorted list of user strings; int_sequences[u] belongs to user_ids[u]
        * item_asins = sorted list of all asins that appear; item id i <-> item_asins[i]
        * int_sequences[u] is sequences[user_ids[u]] with every asin replaced by its int id
    """
    raise NotImplementedError("TODO(m02): implement remap_ids")


class SeqExample(NamedTuple):
    history: list[int]  # items before the target, oldest first
    target: int         # the item to predict


@dataclass
class UserSplit:
    retriever_train: list[int]  # seq[:-3]           -> SASRec trains on this (m06)
    rerank_train: SeqExample    # (seq[:-3], seq[-3]) -> re-ranker training example
    valid: SeqExample           # (seq[:-2], seq[-2])
    test: SeqExample            # (seq[:-1], seq[-1])


def make_splits(seq: Sequence[int]) -> UserSplit:
    """The 4-way split from README §3. Raises ValueError if len(seq) < 4.

    Return fresh lists (not views/aliases of `seq`).
    """
    raise NotImplementedError("TODO(m02): implement make_splits")


def dataset_stats(sequences: Sequence[Sequence[int]], num_items: int) -> dict[str, float]:
    """Summary numbers comparable to the paper's Table 1:

    {"users": int, "items": int, "interactions": int,
     "avg_seq_len": float, "density": interactions / (users * items)}
    """
    raise NotImplementedError("TODO(m02): implement dataset_stats")
