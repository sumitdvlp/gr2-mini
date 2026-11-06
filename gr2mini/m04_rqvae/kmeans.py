"""Distances, k-means++ seeding and Lloyd's k-means.  (README §4.2)

Used to initialize each RQ codebook level on the residuals of that level ("RQ-K-means init").
All functions work on whatever device `x` lives on.
"""
from __future__ import annotations

import torch


import torch

def pairwise_sq_dists(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between every row of x [N, D] and every row of c [K, D].

    Returns [N, K], all entries >= 0 (clamp tiny negatives caused by float error).
    Do NOT materialize an [N, K, D] tensor: use ||x||^2 - 2 x.c^T + ||c||^2.
    Must be differentiable w.r.t. both x and c (no in-place ops on inputs).
    """
    # ||x||^2: sum along features, shape [N, 1]
    x_sq = (x ** 2).sum(dim=1, keepdim=True)
    
    # ||c||^2: sum along features, shape [1, K]
    c_sq = (c ** 2).sum(dim=1).unsqueeze(0)
    
    # 2 * x * c^T: matrix multiplication, shape [N, K]
    cross = x @ c.T
    
    # Compute squared distances using broadcasting: [N, 1] - 2*[N, K] + [1, K] -> [N, K]
    dists = x_sq - 2.0 * cross + c_sq
    
    # Clamp tiny negative floating-point errors to 0.0 while preserving gradients
    return torch.clamp(dists, min=0.0)


import torch

def kmeans_plusplus_init(x: torch.Tensor, k: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """k-means++ seeding. Returns k rows of x (copies), shape [k, D].

    * 1. first centre: a uniformly random row of x
    * 2. each next centre: sample a row with probability proportional to D(x)^2, its squared
         distance to the NEAREST centre chosen so far
    * 3. if all D(x)^2 are 0 (fewer distinct points than k), sample uniformly instead

    * Raise ValueError if x has fewer than k rows.
    * All randomness must come from `generator` (a CPU generator; sample probabilities on CPU
      if x is on GPU, e.g. torch.multinomial(p.cpu(), 1, generator=generator)).
    """
    N, D = x.shape
    if N < k:
        raise ValueError(f"Number of rows in x ({N}) is less than k ({k}).")

    centers = []

    # 1. First centre: uniformly random row of x
    first_idx = int(torch.randint(0, N, (1,), generator=generator).item())
    centers.append(x[first_idx].clone())

    # 2. Each subsequent centre
    for _ in range(1, k):
        c_tensor = torch.stack(centers, dim=0)  # Shape [num_centers, D]

        # Compute squared Euclidean distances to all current centres using ||x||^2 - 2 x c^T + ||c||^2
        x_sq = (x ** 2).sum(dim=1, keepdim=True)             # [N, 1]
        c_sq = (c_tensor ** 2).sum(dim=1).unsqueeze(0)       # [1, num_centers]
        cross = x @ c_tensor.T                               # [N, num_centers]
        dists = torch.clamp(x_sq - 2.0 * cross + c_sq, min=0.0) # [N, num_centers]

        # Distance to the NEAREST centre chosen so far
        min_dists = dists.min(dim=1).values                  # [N]

        sum_dists = min_dists.sum()
        if sum_dists == 0 or torch.isnan(sum_dists):
            # Fallback: if all squared distances are 0, sample uniformly
            next_idx = int(torch.randint(0, N, (1,), generator=generator).item())
        else:
            # Sample with probability proportional to D(x)^2
            probs = min_dists / sum_dists
            probs_cpu = probs.cpu()
            next_idx = int(torch.multinomial(probs_cpu, 1, generator=generator).item())

        centers.append(x[next_idx].clone())

    return torch.stack(centers, dim=0)


import torch

@torch.no_grad()
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
    N, D = x.shape
    if N < k:
        raise ValueError(f"Number of rows in x ({N}) is less than k ({k}).")

    # Initialize centroids using k-means++ seeding
    centroids = kmeans_plusplus_init(x, k, generator=generator)

    prev_assign = None

    for _ in range(n_iters):
        # 1. Assignment step: compute squared Euclidean distances and find nearest centroids
        dists = pairwise_sq_dists(x, centroids)  # [N, k]
        assign = dists.argmin(dim=1)             # [N]

        # Early stopping if assignments no longer change
        if prev_assign is not None and torch.equal(assign, prev_assign):
            break
        prev_assign = assign.clone()

        # 2. Update step: compute new centroids as the mean of assigned points
        new_centroids = []
        empty_clusters = []

        for j in range(k):
            mask = (assign == j)
            if mask.any():
                cluster_points = x[mask]
                new_centroids.append(cluster_points.mean(dim=0))
            else:
                empty_clusters.append(j)
                new_centroids.append(centroids[j].clone())

        centroids = torch.stack(new_centroids, dim=0)

        # 3. Handle empty clusters: move them to points currently farthest from their assigned centroids
        if empty_clusters:
            # Get the distance of each point to its currently assigned centroid
            assigned_dists = dists[torch.arange(N, device=x.device), assign]
            farthest_indices = torch.argsort(assigned_dists, descending=True)

            used_points = set()
            for j in empty_clusters:
                for idx in farthest_indices:
                    idx_val = int(idx.item())
                    if idx_val not in used_points:
                        used_points.add(idx_val)
                        centroids[j] = x[idx_val].clone()
                        break

    # Final guarantee: ensure returned `assign` is strictly consistent with final centroids
    final_dists = pairwise_sq_dists(x, centroids)
    assign = final_dists.argmin(dim=1)

    return centroids, assign
