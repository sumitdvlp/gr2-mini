"""RQ-VAE objective and training loop.  (Eq. 25 + Eqs. 2, 7; README §3, §6)"""
from __future__ import annotations

from typing import Callable, Sequence

import torch

from gr2mini.m01_metrics.metrics import uniqueness
from gr2mini.m04_rqvae.contrastive import info_nce, sample_positives
from gr2mini.m04_rqvae.losses import diversity_loss, reconstruction_loss
from gr2mini.m04_rqvae.model import RQVAE, RQVAEOutput  # noqa: F401


def rqvae_loss(
    model: RQVAE,
    x: torch.Tensor,
    pos_x: torch.Tensor | None = None,
    pos_valid: torch.Tensor | None = None,
    *,
    commitment_weight: float = 0.25,
    codebook_weight: float = 1.0,
    diversity_weight: float = 0.0,
    diversity_temperature: float = 1.0,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float], RQVAEOutput]:
    """Full objective for one batch.

        out   = model(x)
        rec   = reconstruction_loss(out.x_hat, x)
        cb    = out.rq.codebook_loss              com = out.rq.commitment_loss
        div   = diversity_loss(out.rq.dists, diversity_temperature)          only if diversity_weight > 0
        ctr   = info_nce(out.z, model.encode(pos_x), contrastive_temperature, pos_valid)
                                                  only if contrastive_weight > 0 (pos_x required then)
        total = rec + codebook_weight*cb + commitment_weight*com + diversity_weight*div + contrastive_weight*ctr

    Returns (total, terms, out) where terms = {"loss", "rec", "codebook", "commitment",
    "diversity", "contrastive"} as python floats (0.0 for disabled terms).
    Raises ValueError if contrastive_weight > 0 and pos_x is None.
    """
    if contrastive_weight > 0 and pos_x is None:
        raise ValueError("contrastive_weight > 0 requires pos_x")

    out = model(x)
    rec = reconstruction_loss(out.x_hat, x)
    cb = out.rq.codebook_loss
    com = out.rq.commitment_loss
    zero = torch.zeros((), device=x.device, dtype=rec.dtype)

    div = diversity_loss(out.rq.dists, diversity_temperature) if diversity_weight > 0 else zero
    if contrastive_weight > 0:
        ctr = info_nce(out.z, model.encode(pos_x), contrastive_temperature, pos_valid)
    else:
        ctr = zero

    total = (
        rec
        + codebook_weight * cb
        + commitment_weight * com
        + diversity_weight * div
        + contrastive_weight * ctr
    )
    terms = {
        "loss": float(total.item()),
        "rec": float(rec.item()),
        "codebook": float(cb.item()),
        "commitment": float(com.item()),
        "diversity": float(div.item()),
        "contrastive": float(ctr.item()),
    }
    return total, terms, out


def train_rqvae(
    model: RQVAE,
    embeddings: torch.Tensor,
    cfg: dict,
    neighbors: Sequence[Sequence[int]] | None = None,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    log: Callable[[str], None] | None = print,
) -> list[dict]:
    """Train `model` on all item embeddings [N, in_dim] (README §6). Returns the history.

    `cfg` is the `rqvae:` config section. Keys used: epochs, batch_size, lr, kmeans_iters,
    eval_every, commitment_weight, codebook_weight, diversity_weight, diversity_temperature,
    contrastive_weight, contrastive_temperature.

    Steps:
        1. model.to(device); x = embeddings.float().to(device); model.train()
        2. with no_grad: model.quantizer.init_codebooks(model.encode(x), cfg["kmeans_iters"], generator)
        3. optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        4. each epoch: ids = torch.randperm(N, generator=generator); iterate in batch_size chunks:
               if contrastive_weight > 0: pos, valid = sample_positives(ids_batch, neighbors, generator)
               loss, terms, out = rqvae_loss(...)
               zero_grad / backward / step;  model.quantizer.update_codebooks(out.rq)
        5. Record a history entry at epoch 1, at every multiple of eval_every, and at the last epoch:
               {"epoch": int, <mean of each term over the epoch's batches>,
                "uniqueness": m01 uniqueness of model.get_codes(x),
                "used_codes": [number of distinct codes used at each level]}
           and log a one-line summary if `log` is not None.
    Leave the model in train mode at the end. If contrastive_weight > 0 and neighbors is None,
    raise ValueError.
    """
    contrastive_weight = float(cfg.get("contrastive_weight", 0.0))
    if contrastive_weight > 0 and neighbors is None:
        raise ValueError("contrastive_weight > 0 requires neighbors")

    loss_kwargs = dict(
        commitment_weight=float(cfg.get("commitment_weight", 0.25)),
        codebook_weight=float(cfg.get("codebook_weight", 1.0)),
        diversity_weight=float(cfg.get("diversity_weight", 0.0)),
        diversity_temperature=float(cfg.get("diversity_temperature", 1.0)),
        contrastive_weight=contrastive_weight,
        contrastive_temperature=float(cfg.get("contrastive_temperature", 0.1)),
    )
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    eval_every = int(cfg["eval_every"])

    model.to(device)
    x = embeddings.float().to(device)
    n = x.shape[0]
    model.train()

    with torch.no_grad():
        model.quantizer.init_codebooks(model.encode(x), cfg["kmeans_iters"], generator)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        ids = torch.randperm(n, generator=generator)
        sums: dict[str, float] = {}
        n_batches = 0
        for start in range(0, n, batch_size):
            ids_batch = ids[start : start + batch_size]
            xb = x[ids_batch.to(device)]
            pos_x = pos_valid = None
            if contrastive_weight > 0:
                pos_ids, pos_valid = sample_positives(ids_batch, neighbors, generator)
                pos_x = x[pos_ids.to(device)]
                pos_valid = pos_valid.to(device)

            loss, terms, out = rqvae_loss(model, xb, pos_x, pos_valid, **loss_kwargs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.quantizer.update_codebooks(out.rq)

            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v
            n_batches += 1

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            codes = model.get_codes(x)
            entry = {"epoch": epoch, **{k: v / n_batches for k, v in sums.items()}}
            entry["uniqueness"] = float(uniqueness(codes.tolist()))
            entry["used_codes"] = [int(codes[:, k].unique().numel()) for k in range(codes.shape[1])]
            history.append(entry)
            if log is not None:
                log(
                    f"[rqvae] epoch {epoch:>4d}  loss {entry['loss']:.4f}  rec {entry['rec']:.4f}  "
                    f"com {entry['commitment']:.4f}  cb {entry['codebook']:.4f}  "
                    f"div {entry['diversity']:.4f}  ctr {entry['contrastive']:.4f}  "
                    f"uniq {entry['uniqueness']:.4f}  used {entry['used_codes']}"
                )

    model.train()
    return history
