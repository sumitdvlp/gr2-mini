"""Distances, k-means++ seeding and Lloyd's k-means.  (README §4.2)

Used to initialize each RQ codebook level on the residuals of that level ("RQ-K-means init").
All functions work on whatever device `x` lives on.
"""
from __future__ import annotations

import torch


def pairwise_sq_dists(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between every row of x [N, D] and every row of c [K, D].

    Returns [N, K], all entries >= 0 (clamp tiny negatives caused by float error).
    Do NOT materialize an [N, K, D] tensor: use ||x||^2 - 2 x.c^T + ||c||^2.
    Must be differentiable w.r.t. both x and c (no in-place ops on inputs).
    """
    raise NotImplementedError("TODO(m04): implement pairwise_sq_dists")


def kmeans_plusplus_init(x: torch.Tensor, k: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """k-means++ seeding. Returns k rows of x (copies), shape [k, D].

    1. first centre: a uniformly random row of x
    2. each next centre: sample a row with probability proportional to D(x)^2, its squared
       distance to the NEAREST centre chosen so far
    3. if all D(x)^2 are 0 (fewer distinct points than k), sample uniformly instead

    * Raise ValueError if x has fewer than k rows.
    * All randomness must come from `generator` (a CPU generator; sample probabilities on CPU
      if x is on GPU, e.g. torch.multinomial(p.cpu(), 1, generator=generator)).
    """
    raise NotImplementedError("TODO(m04): implement kmeans_plusplus_init")


def kmeans(
    x: torch.Tensor, k: int, n_iters: int = 20, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's algorithm starting from kmeans_plusplus_init. Returns (centroids [k, D], assign [N] long).

    Each iteration:
        assign  = argmin over centroids of pairwise_sq_dists
        update  = each centroid becomes the mean of its assigned points
        empty clusters: move them onto the points currently farthest from their assigned
                        centroid (a different point for each empty cluster)
        stop early when assignments no longer change

    The returned `assign` must be consistent with the returned centroids, i.e.
    assign == pairwise_sq_dists(x, centroids).argmin(1).
    Must not modify x. Runs under torch.no_grad() (decorate it, or use a `with` block).
    """
    raise NotImplementedError("TODO(m04): implement kmeans")
