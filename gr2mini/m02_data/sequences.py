"""Interactions -> filtered, chronological, integer-id user sequences and splits.  (README §1-3)

An "interaction" is a tuple (user: str, item: str, timestamp: int).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence
from collections import Counter
Interaction = tuple[str, str, int]


def k_core_filter(interactions: Sequence[Interaction], k: int) -> list[Interaction]:
    """Iteratively remove users and items with fewer than `k` interactions until stable.

    * Counts are numbers of interaction ROWS (a repeated (user, item) pair counts twice).
    * The surviving interactions keep their ORIGINAL relative order.
    * k <= 1 returns a copy of the input.
    """
    if k <= 1:
        return list(interactions)

    current = list(interactions)

    while True:
        # Count occurrences of each user and item in the current set
        user_counts = Counter(i[0] for i in current)
        item_counts = Counter(i[1] for i in current)

        # Identify users and items that meet the threshold k
        valid_users = {user for user, count in user_counts.items() if count >= k}
        valid_items = {item for item, count in item_counts.items() if count >= k}

        # Filter interactions keeping only those where both user and item are valid
        # (List comprehension preserves the original relative order)
        next_current = [
            i for i in current 
            if i[0] in valid_users and i[1] in valid_items
        ]

        # If no interactions were removed in this pass, we have reached stability
        if len(next_current) == len(current):
            break

        current = next_current

    return current


def subsample_users(interactions: Sequence[Interaction], max_users: int | None, seed: int) -> list[Interaction]:
    """Keep only the interactions of a random subset of `max_users` users.

    * max_users None, or >= the number of users -> return a copy of the input.
    * Choose users with random.Random(seed).sample(sorted(unique_users), max_users) so the
      choice is deterministic for a given seed.
    * Keep the original order of the surviving interactions.
    """
    if max_users is None:
        return list(interactions)

    # Extract unique users and sort them for deterministic sampling
    unique_users = sorted({i[0] for i in interactions})

    # If max_users is greater than or equal to the number of unique users, return a copy
    if max_users >= len(unique_users):
        return list(interactions)

    # Sample a subset of users using the provided seed for reproducibility
    import random
    rng = random.Random(seed)
    sampled_users = set(rng.sample(unique_users, max_users))

    # Filter interactions to keep only those with sampled users, preserving original order
    filtered_interactions = [i for i in interactions if i[0] in sampled_users]

    return filtered_interactions


def build_sequences(interactions: Sequence[Interaction]) -> dict[str, list[str]]:
    """Group interactions by user into chronologically sorted item lists.

    Ties in timestamp keep their original file order (use a STABLE sort).
    """
    from collections import defaultdict

    # Create a dictionary to hold user sequences
    user_sequences = defaultdict(list)

    # Sort interactions by timestamp while preserving original order for ties
    sorted_interactions = sorted(interactions, key=lambda x: x[2])  # Sort by timestamp

    # Group items by user
    for user, item, _ in sorted_interactions:
        user_sequences[user].append(item)

    return dict(user_sequences)


def remap_ids(sequences: dict[str, list[str]]) -> tuple[list[list[int]], list[str], list[str]]:
    """Map string ids to dense ints, deterministically.

    Returns (int_sequences, user_ids, item_asins) where
        * user_ids   = sorted list of user strings; int_sequences[u] belongs to user_ids[u]
        * item_asins = sorted list of all asins that appear; item id i <-> item_asins[i]
        * int_sequences[u] is sequences[user_ids[u]] with every asin replaced by its int id
    """
    # Create sorted lists of unique users and items
    user_ids = sorted(sequences.keys())
    item_set = {item for items in sequences.values() for item in items}
    item_asins = sorted(item_set)

    # Create mappings from string IDs to integer IDs
    user_to_int = {user: idx for idx, user in enumerate(user_ids)}
    item_to_int = {item: idx for idx, item in enumerate(item_asins)}

    # Build the list of integer sequences
    int_sequences = []
    for user in user_ids:
        int_sequence = [item_to_int[item] for item in sequences[user]]
        int_sequences.append(int_sequence)

    return int_sequences, user_ids, item_asins


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
    if len(seq) < 4:
        raise ValueError("Sequence must have at least 4 items to create splits.")

    retriever_train = list(seq[:-3])  # All but the last 3 items
    rerank_train = SeqExample(history=list(seq[:-3]), target=seq[-3])  # Third-to-last item
    valid = SeqExample(history=list(seq[:-2]), target=seq[-2])  # Second-to-last item
    test = SeqExample(history=list(seq[:-1]), target=seq[-1])  # Last item

    return UserSplit(
        retriever_train=retriever_train,
        rerank_train=rerank_train,
        valid=valid,
        test=test
    )


def dataset_stats(sequences: Sequence[Sequence[int]], num_items: int) -> dict[str, float]:
    """Summary numbers comparable to the paper's Table 1:

    {"users": int, "items": int, "interactions": int,
     "avg_seq_len": float, "density": interactions / (users * items)}
    """
    num_users = len(sequences)
    total_interactions = sum(len(seq) for seq in sequences)
    avg_seq_len = total_interactions / num_users if num_users > 0 else 0.0
    density = total_interactions / (num_users * num_items) if num_users > 0 and num_items > 0 else 0.0

    return {
        "users": num_users,
        "items": num_items,
        "interactions": total_interactions,
        "avg_seq_len": avg_seq_len,
        "density": density
    }
