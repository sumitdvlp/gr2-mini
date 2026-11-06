"""The RQ-VAE: MLP encoder -> ResidualQuantizer -> MLP decoder.  (README §2)"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from gr2mini.m04_rqvae.quantizer import ResidualQuantizer, RQOutput  # noqa: F401


def mlp(dims: Sequence[int]) -> nn.Sequential:
    """Linear layers dims[0]->dims[1]->...->dims[-1] with nn.ReLU between them, NO activation
    after the last layer. mlp([4, 8, 2]) == Sequential(Linear(4, 8), ReLU(), Linear(8, 2)).
    """
    raise NotImplementedError("TODO(m04): implement mlp")


@dataclass
class RQVAEOutput:
    x_hat: torch.Tensor  # [B, in_dim] reconstruction
    z: torch.Tensor  # [B, latent_dim] encoder output (pre-quantization)
    rq: RQOutput  # everything the quantizer returned


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        num_levels: int,
        codebook_size: int,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        dead_code_threshold: int | None = None,
    ):
        """Build:
            self.encoder   = mlp([in_dim, *hidden_dims, latent_dim])
            self.decoder   = mlp([latent_dim, *reversed(hidden_dims), in_dim])
            self.quantizer = ResidualQuantizer(num_levels, codebook_size, latent_dim, use_ema=...,
                                               ema_decay=..., dead_code_threshold=...)
            self.init_kwargs = dict of ALL constructor arguments (hidden_dims as a list),
                               so save_rqvae/load_rqvae can rebuild the model.
        """
        super().__init__()
        raise NotImplementedError("TODO(m04): implement RQVAE.__init__")

    @classmethod
    def from_config(cls, in_dim: int, rq_cfg: dict) -> "RQVAE":
        """Build from the `rqvae:` config section (keys: latent_dim, hidden_dims, num_levels,
        codebook_size, use_ema, ema_decay, dead_code_threshold)."""
        raise NotImplementedError("TODO(m04): implement RQVAE.from_config")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, in_dim] -> z [B, latent_dim]."""
        raise NotImplementedError("TODO(m04): implement RQVAE.encode")

    def forward(self, x: torch.Tensor) -> RQVAEOutput:
        """z = encode(x); rq = quantizer(z); x_hat = decoder(rq.z_q)."""
        raise NotImplementedError("TODO(m04): implement RQVAE.forward")

    @torch.no_grad()
    def get_codes(self, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        """Codes [N, L] for all rows of x, processed in chunks of batch_size. Does not change
        the train/eval mode of the model or any state."""
        raise NotImplementedError("TODO(m04): implement RQVAE.get_codes")


def save_rqvae(model: RQVAE, path: str | Path, extra: dict | None = None) -> None:
    """torch.save({"init_kwargs": ..., "state_dict": ..., "extra": extra or {}}, path). Create parent dirs."""
    raise NotImplementedError("TODO(m04): implement save_rqvae")


def load_rqvae(path: str | Path, map_location: str | torch.device = "cpu") -> RQVAE:
    """Rebuild RQVAE(**init_kwargs), load the state dict (buffers included), return it in eval mode.

    Note: torch.load(..., weights_only=True) is the safe default in recent PyTorch. Plain dicts,
    lists, numbers and tensors load fine with it.
    """
    raise NotImplementedError("TODO(m04): implement load_rqvae")
