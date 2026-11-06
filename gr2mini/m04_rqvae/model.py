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
    dims = list(dims)
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


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
        hidden_dims = list(hidden_dims)
        self.encoder = mlp([in_dim, *hidden_dims, latent_dim])
        self.decoder = mlp([latent_dim, *reversed(hidden_dims), in_dim])
        self.quantizer = ResidualQuantizer(
            num_levels,
            codebook_size,
            latent_dim,
            use_ema=use_ema,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
        )
        self.init_kwargs = dict(
            in_dim=in_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            num_levels=num_levels,
            codebook_size=codebook_size,
            use_ema=use_ema,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
        )

    @classmethod
    def from_config(cls, in_dim: int, rq_cfg: dict) -> "RQVAE":
        """Build from the `rqvae:` config section (keys: latent_dim, hidden_dims, num_levels,
        codebook_size, use_ema, ema_decay, dead_code_threshold)."""
        return cls(
            in_dim=in_dim,
            latent_dim=int(rq_cfg["latent_dim"]),
            hidden_dims=list(rq_cfg["hidden_dims"]),
            num_levels=int(rq_cfg["num_levels"]),
            codebook_size=int(rq_cfg["codebook_size"]),
            use_ema=bool(rq_cfg.get("use_ema", True)),
            ema_decay=float(rq_cfg.get("ema_decay", 0.99)),
            dead_code_threshold=rq_cfg.get("dead_code_threshold"),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, in_dim] -> z [B, latent_dim]."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> RQVAEOutput:
        """z = encode(x); rq = quantizer(z); x_hat = decoder(rq.z_q)."""
        z = self.encode(x)
        rq = self.quantizer(z)
        x_hat = self.decoder(rq.z_q)
        return RQVAEOutput(x_hat=x_hat, z=z, rq=rq)

    @torch.no_grad()
    def get_codes(self, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        """Codes [N, L] for all rows of x, processed in chunks of batch_size. Does not change
        the train/eval mode of the model or any state."""
        # Encoder is a plain Linear/ReLU MLP and quantizer.encode is pure, so no mode switch needed.
        device = self.quantizer.codebooks.device
        dtype = self.quantizer.codebooks.dtype
        if x.shape[0] == 0:
            return torch.empty(0, self.quantizer.num_levels, dtype=torch.long, device=device)
        out = []
        for start in range(0, x.shape[0], batch_size):
            chunk = x[start : start + batch_size].to(device=device, dtype=dtype)
            out.append(self.quantizer.encode(self.encode(chunk)))
        return torch.cat(out, dim=0)


def save_rqvae(model: RQVAE, path: str | Path, extra: dict | None = None) -> None:
    """torch.save({"init_kwargs": ..., "state_dict": ..., "extra": extra or {}}, path). Create parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"init_kwargs": dict(model.init_kwargs), "state_dict": model.state_dict(), "extra": extra or {}},
        path,
    )


def load_rqvae(path: str | Path, map_location: str | torch.device = "cpu") -> RQVAE:
    """Rebuild RQVAE(**init_kwargs), load the state dict (buffers included), return it in eval mode.

    Note: torch.load(..., weights_only=True) is the safe default in recent PyTorch. Plain dicts,
    lists, numbers and tensors load fine with it.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = RQVAE(**ckpt["init_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(map_location)
    model.eval()
    return model
