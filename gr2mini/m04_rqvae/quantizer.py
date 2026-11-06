"""Residual vector quantizer with EMA codebooks and dead-code reset.

Paper: Eqs. 19-23 (residual quantization), 27-28 (codebook/commitment losses), §2.3 EMA (Eqs. 4-5,
implemented the standard VQ-VAE way, see README §4.1), dead-code reset (Eqs. 8-9).

Index convention in code: levels are 0-based, so residuals[:, 0] is the paper's r_1 (= z),
quantized[:, 0] is q_1, codes[:, 0] is the first SID token, etc.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gr2mini.m04_rqvae.kmeans import kmeans, pairwise_sq_dists  # noqa: F401  (you'll need these)


@dataclass
class RQOutput:
    z_q: torch.Tensor  # [B, D] straight-through output: value == z_hat, d z_q / d z == identity
    z_hat: torch.Tensor  # [B, D] sum over levels of quantized vectors (no gradient path to z)
    codes: torch.Tensor  # [B, L] long, chosen code index per level
    residuals: torch.Tensor  # [B, L, D] the residual entering each level (residuals[:, 0] == z)
    quantized: torch.Tensor  # [B, L, D] the codeword chosen at each level
    dists: list[torch.Tensor]  # L tensors [B, C]: squared distances residual -> codebook (differentiable wrt z)
    commitment_loss: torch.Tensor  # scalar: sum_k mean_b ||r_k - sg[q_k]||^2          (Eq. 28)
    codebook_loss: torch.Tensor  # scalar: sum_k mean_b ||sg[r_k] - q_k||^2 (0 if EMA) (Eq. 27)


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        num_levels: int,
        codebook_size: int,
        dim: int,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: int | None = None,
    ):
        """Create the quantizer state.

        Attributes the tests rely on:
            self.num_levels, self.codebook_size, self.dim, self.use_ema, self.ema_decay, self.eps,
            self.dead_code_threshold
            self.codebooks        [L, C, D] float. nn.Parameter if not use_ema, else a BUFFER.
                                  Initialize with small random values (e.g. randn * 0.1);
                                  init_codebooks() overwrites them anyway.
            self.ema_cluster_size [L, C] buffer, initialized to ones      (N_j in README §4.1)
            self.ema_embed_sum    [L, C, D] buffer, initialized to a copy of codebooks   (M_j)
            self.unused_steps     [L, C] long buffer of zeros
            self.initialized      0-dim bool buffer, False until init_codebooks() runs
        """
        super().__init__()
        raise NotImplementedError("TODO(m04): implement ResidualQuantizer.__init__")

    # ------------------------------------------------------------------ pure computation
    def forward(self, z: torch.Tensor) -> RQOutput:
        """Quantize z [B, D] level by level (Eqs. 19-23). MUST NOT modify any module state.

        For each level k (0-based):
            dists_k = pairwise_sq_dists(r_k, codebooks[k])       # [B, C]
            codes_k = dists_k.argmin(-1)
            q_k     = codebooks[k][codes_k]
            r_{k+1} = r_k - q_k.detach()                         # detach the residual chain!
        z_hat = sum_k q_k ;  z_q = z + (z_hat - z).detach()       # straight-through
        commitment_loss = sum_k mean_b ||r_k - q_k.detach()||^2  (sum over D, mean over batch)
        codebook_loss   = sum_k mean_b ||r_k.detach() - q_k||^2  if not use_ema
                          else torch.zeros((), device=z.device)
        """
        raise NotImplementedError("TODO(m04): implement ResidualQuantizer.forward")

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Codes [B, L] only (same as forward(z).codes, without building the rest)."""
        raise NotImplementedError("TODO(m04): implement ResidualQuantizer.encode")

    # ------------------------------------------------------------------ state updates
    @torch.no_grad()
    def init_codebooks(self, z: torch.Tensor, kmeans_iters: int = 20, generator: torch.Generator | None = None) -> None:
        """RQ-K-means init (README §4.2): for each level, run kmeans(residual, C) and use the
        centroids as that level's codebook; then subtract the assigned centroids to get the next
        level's residual.

        Also set: ema_embed_sum[k] = centroids, ema_cluster_size[k] = 1, unused_steps = 0,
        initialized = True. Copy values INTO the existing tensors (e.g. self.codebooks.data[k] = ...),
        don't replace the Parameter/buffer objects.
        """
        raise NotImplementedError("TODO(m04): implement init_codebooks")

    @torch.no_grad()
    def ema_update(self, level: int, residual: torch.Tensor, codes: torch.Tensor) -> None:
        """One EMA step for codebook `level` (README §4.1), given residual [B, D] and codes [B]:

            n = counts per code [C];  s = per-code sums of residuals [C, D]
            N = decay * N + (1 - decay) * n
            M = decay * M + (1 - decay) * s
            total = N.sum();  N_smooth = (N + eps) / (total + C * eps) * total
            codebooks[level] = M / N_smooth[:, None]
        """
        raise NotImplementedError("TODO(m04): implement ema_update")

    @torch.no_grad()
    def reset_dead_codes(self, level: int, residual: torch.Tensor, codes: torch.Tensor) -> int:
        """Dead-code bookkeeping + reset for one level (Eqs. 8-9, README §4.3). Returns #codes reset.

        1. unused_steps[level][used codes] = 0 ; all other codes of this level += 1
        2. dead = codes with unused_steps >= dead_code_threshold   (nothing to do if threshold is None)
        3. error_i = ||residual_i - codebooks[level][codes_i]||  using the CURRENT codebook
           take the n_dead residuals with the LARGEST error (distinct rows; if n_dead > B, only
           reset the first B dead codes)
        4. for each reset code j: codebooks[level][j] = that residual,
           ema_embed_sum[level][j] = that residual, ema_cluster_size[level][j] = 1,
           unused_steps[level][j] = 0
        """
        raise NotImplementedError("TODO(m04): implement reset_dead_codes")

    @torch.no_grad()
    def update_codebooks(self, out: RQOutput) -> dict[str, list[int]]:
        """Call AFTER optimizer.step(). For every level k, in order:
            if use_ema:                    ema_update(k, out.residuals[:, k], out.codes[:, k])
            if dead_code_threshold set:    n_k = reset_dead_codes(k, out.residuals[:, k], out.codes[:, k])
        Detach inputs defensively. Returns {"dead_codes_reset": [n_0, n_1, ...]} (zeros if disabled).
        """
        raise NotImplementedError("TODO(m04): implement update_codebooks")
