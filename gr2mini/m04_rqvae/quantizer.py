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
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.dim = dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        init = torch.randn(num_levels, codebook_size, dim) * 0.1
        if use_ema:
            self.register_buffer("codebooks", init)
        else:
            self.codebooks = nn.Parameter(init)
        self.register_buffer("ema_cluster_size", torch.ones(num_levels, codebook_size))
        self.register_buffer("ema_embed_sum", init.clone())
        self.register_buffer("unused_steps", torch.zeros(num_levels, codebook_size, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))  

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
        residual = z
        residuals, quantized, codes, dists = [], [], [], []
        commitment_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        codebook_loss = torch.zeros((), device=z.device, dtype=z.dtype)

        for k in range(self.num_levels):
            codebook = self.codebooks[k]  # [C, D]
            d_k = pairwise_sq_dists(residual, codebook)  # [B, C]
            c_k = d_k.argmin(dim=-1)  # [B]
            q_k = codebook[c_k]  # [B, D]

            commitment_loss = commitment_loss + ((residual - q_k.detach()) ** 2).sum(-1).mean()
            if not self.use_ema:
                codebook_loss = codebook_loss + ((residual.detach() - q_k) ** 2).sum(-1).mean()

            residuals.append(residual)
            quantized.append(q_k)
            codes.append(c_k)
            dists.append(d_k)

            residual = residual - q_k.detach()

        quantized_t = torch.stack(quantized, dim=1)  # [B, L, D]
        z_hat = quantized_t.sum(dim=1)  # [B, D]
        z_q = z + (z_hat - z).detach()

        return RQOutput(
            z_q=z_q,
            z_hat=z_hat,
            codes=torch.stack(codes, dim=1),
            residuals=torch.stack(residuals, dim=1),
            quantized=quantized_t,
            dists=dists,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Codes [B, L] only (same as forward(z).codes, without building the rest)."""
        residual = z
        codes = []
        for k in range(self.num_levels):
            codebook = self.codebooks[k]
            c_k = pairwise_sq_dists(residual, codebook).argmin(dim=-1)
            codes.append(c_k)
            residual = residual - codebook[c_k]
        return torch.stack(codes, dim=1)

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
        residual = z.detach().to(device=self.codebooks.device, dtype=self.codebooks.dtype)
        for k in range(self.num_levels):
            centroids, assign = kmeans(residual, self.codebook_size, n_iters=kmeans_iters, generator=generator)
            self.codebooks.data[k].copy_(centroids)
            self.ema_embed_sum[k].copy_(centroids)
            self.ema_cluster_size[k].fill_(1.0)
            # assign is consistent with centroids (argmin of pairwise_sq_dists), so this is the
            # same residual the forward pass would produce at the next level.
            residual = residual - centroids[assign]
        self.unused_steps.zero_()
        self.initialized.fill_(True)

    @torch.no_grad()
    def ema_update(self, level: int, residual: torch.Tensor, codes: torch.Tensor) -> None:
        """One EMA step for codebook `level` (README §4.1), given residual [B, D] and codes [B]:

            n = counts per code [C];  s = per-code sums of residuals [C, D]
            N = decay * N + (1 - decay) * n
            M = decay * M + (1 - decay) * s
            total = N.sum();  N_smooth = (N + eps) / (total + C * eps) * total
            codebooks[level] = M / N_smooth[:, None]
        """
        C = self.codebook_size
        codebook = self.codebooks.data[level]
        N = self.ema_cluster_size[level]
        M = self.ema_embed_sum[level]
        residual = residual.detach().to(dtype=M.dtype)
        codes = codes.detach().long()

        n = torch.bincount(codes, minlength=C).to(N.dtype)  # [C]
        s = torch.zeros_like(M).index_add_(0, codes, residual)  # [C, D]

        N.mul_(self.ema_decay).add_(n, alpha=1 - self.ema_decay)
        M.mul_(self.ema_decay).add_(s, alpha=1 - self.ema_decay)

        total = N.sum()
        N_smooth = (N + self.eps) / (total + C * self.eps) * total
        codebook.copy_(M / N_smooth[:, None])

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
        residual = residual.detach()
        codes = codes.detach().long()
        unused = self.unused_steps[level]

        # 1. bookkeeping: every code ages by one step, codes used in this batch go back to 0
        used = torch.zeros(self.codebook_size, dtype=torch.bool, device=unused.device)
        used[codes] = True
        unused += 1
        unused[used] = 0

        # 2. find dead codes
        if self.dead_code_threshold is None:
            return 0
        dead = torch.nonzero(unused >= self.dead_code_threshold, as_tuple=False).squeeze(1)  # ascending
        n_reset = min(int(dead.numel()), residual.shape[0])
        if n_reset == 0:
            return 0
        dead = dead[:n_reset]

        # 3. the n_reset worst-quantized residuals under the current codebook (distinct rows)
        codebook = self.codebooks.data[level]
        error = (residual - codebook[codes]).norm(dim=-1)  # [B]
        worst = torch.topk(error, n_reset).indices
        new_vecs = residual[worst].to(codebook.dtype)

        # 4. reseed the dead codes
        codebook[dead] = new_vecs
        self.ema_embed_sum[level][dead] = new_vecs
        self.ema_cluster_size[level][dead] = 1.0
        unused[dead] = 0
        return n_reset

    @torch.no_grad()
    def update_codebooks(self, out: RQOutput) -> dict[str, list[int]]:
        """Call AFTER optimizer.step(). For every level k, in order:
            if use_ema:                    ema_update(k, out.residuals[:, k], out.codes[:, k])
            if dead_code_threshold set:    n_k = reset_dead_codes(k, out.residuals[:, k], out.codes[:, k])
        Detach inputs defensively. Returns {"dead_codes_reset": [n_0, n_1, ...]} (zeros if disabled).
        """
        residuals = out.residuals.detach()
        codes = out.codes.detach()
        dead_codes_reset: list[int] = []
        for k in range(self.num_levels):
            r_k, c_k = residuals[:, k], codes[:, k]
            if self.use_ema:
                self.ema_update(k, r_k, c_k)
            n_k = self.reset_dead_codes(k, r_k, c_k) if self.dead_code_threshold is not None else 0
            dead_codes_reset.append(int(n_k))
        return {"dead_codes_reset": dead_codes_reset}
