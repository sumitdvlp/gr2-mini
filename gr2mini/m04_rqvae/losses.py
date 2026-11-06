"""Reconstruction and diversity losses.  (Eq. 26; Eqs. 6-7; README §3, §4.4)"""
from __future__ import annotations

import torch


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Eq. 26: mean over the batch of the squared L2 error summed over dimensions. Scalar.

    (Note: this is D times F.mse_loss, which averages over dimensions too.)
    """
    return torch.mean(torch.sum((x_hat - x) ** 2, dim=1))


def soft_assignments(dists: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Soft code assignment probabilities: softmax(-dists / temperature) over the last dim.

    dists: [B, C] squared distances. Returns [B, C], rows sum to 1.
    """
    return torch.softmax(-dists / temperature, dim=-1)


def diversity_loss(dists_per_level: list[torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
    """Eqs. 6-7 WITHOUT the lambda_div factor (it's applied by the caller):

        for each level k:  p_k = soft_assignments(dists_k, T).mean(dim=0)     # [C_k], usage distribution
        L_div = sum_k  C_k * sum_j p_k[j]^2

    Equals (#levels) when usage is uniform at every level, and sum_k C_k under total collapse.
    Must be differentiable w.r.t. the distances.
    """
    loss = torch.tensor(0.0, device=dists_per_level[0].device)
    for dists in dists_per_level:
        p = soft_assignments(dists, temperature).mean(dim=0)  # [C_k]
        loss += dists.shape[1] * torch.sum(p**2)
    return loss
