"""GR2 reward: rank improvement (Eq. 14) + conditional format reward (Eq. 15).  (README)"""
from __future__ import annotations

from dataclasses import dataclass

from gr2mini.m10_parsing.parsing import apply_ranking, parse_rerank_output  # noqa: F401


@dataclass
class RewardInfo:
    total: float  # R (Eq. 15)
    rank: float  # R_rank (Eq. 14)
    fmt: float  # R_fmt in {0.0, 1.0}
    parsed: bool  # did m10 return a ranking?
    new_rank: int  # 1-based rank of the target after re-ranking (pre-rank if unparsed)
    identity: bool  # parsed AND equal to the input order (reward-hacking alarm)


def rank_reward(pre_rank: int, new_rank: int, list_size: int) -> float:
    """Eq. 14: (pre_rank - new_rank) / list_size."""
    raise NotImplementedError("TODO(m11): implement rank_reward")


def compute_reward(text: str, example: dict, alpha: float = 0.1, conditional: bool = True) -> RewardInfo:
    """Score one model output for one candidate record (m06 format).

    1. parsed = parse_rerank_output(text, len(example["candidates"]))
    2. new order = apply_ranking(candidates, parsed.ranking)   (pre-rank order if unparsed)
       new_rank  = 1-based position of example["target"] in the new order
    3. R_rank = rank_reward(example["target_rank"], new_rank, |D|);  R_fmt = 1.0 if parsed else 0.0
    4. conditional: R = R_rank + alpha * R_fmt if (R_rank > 0 or target_rank == 1) else R_rank
       unconditional (the ablation): R = R_rank + alpha * R_fmt always
    5. identity = parsed and ranking == [0, 1, ..., |D|-1]
    """
    raise NotImplementedError("TODO(m11): implement compute_reward")
