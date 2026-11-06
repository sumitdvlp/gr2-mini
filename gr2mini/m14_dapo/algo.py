"""DAPO math: advantages, dynamic sampling, masks, clipped token-level loss.  (Eqs. 16-17; README)"""
from __future__ import annotations

from typing import Iterable

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. 17. rewards [P, G] (P prompts, G samples each) -> advantages [P, G]:
    (R - mean over the group) / (population std over the group (unbiased=False) + eps)."""
    raise NotImplementedError("TODO(m14): implement group_advantages")


def informative_groups(rewards: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    """Dynamic sampling: bool [P], True where the group's rewards are NOT all equal (max - min > tol)."""
    raise NotImplementedError("TODO(m14): implement informative_groups")


def completion_mask(completion_ids: torch.Tensor, stop_ids: Iterable[int]) -> torch.Tensor:
    """Bool [N, T]: True for every token up to AND INCLUDING the first stop token of each row,
    False after it. A row without any stop token is all True (truncated answer)."""
    raise NotImplementedError("TODO(m14): implement completion_mask")


def dapo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    num_tokens: int | float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq. 16 as a loss to MINIMIZE.

    logp, old_logp: [N, T] token log-probs under pi_theta (with grad) and pi_old (no grad)
    advantages:     [N] one per sequence, broadcast over its tokens
    mask:           [N, T] bool, which tokens count
        ratio = exp(logp - old_logp)
        obj   = min(ratio * A, clamp(ratio, 1 - eps_low, 1 + eps_high) * A)
        loss  = -(obj * mask).sum() / (num_tokens if given else mask.sum())
    stats: {"clip_frac": fraction of masked tokens where the clamp changed the ratio,
            "ratio_mean": mean ratio over masked tokens}   (python floats)
    """
    raise NotImplementedError("TODO(m14): implement dapo_loss")
