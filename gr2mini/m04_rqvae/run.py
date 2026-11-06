"""Try it:  python -m gr2mini.m04_rqvae.run --config configs/tiny.yaml [--set rqvae.use_ema=false ...]

Trains the RQ-VAE on the item embeddings from m03 and saves <artifacts_dir>/rqvae.pt.
"""
from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np
import torch

from gr2mini.m00_setup.config import apply_overrides, load_config
from gr2mini.m00_setup.device import get_device
from gr2mini.m00_setup.seed import make_generator, seed_everything
from gr2mini.m02_data.sequences import make_splits
from gr2mini.m02_data.storage import load_processed
from gr2mini.m04_rqvae.contrastive import build_coengagement_neighbors
from gr2mini.m04_rqvae.model import RQVAE, save_rqvae
from gr2mini.m04_rqvae.train import train_rqvae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RQ-VAE semantic-ID tokenizer.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="config override, e.g. --set rqvae.use_ema=false (repeatable)")
    parser.add_argument("--out", default=None, help="checkpoint path (default <artifacts_dir>/rqvae.pt)")
    args = parser.parse_args()

    # TODO(m04):
    #   1. cfg = apply_overrides(load_config(args.config), args.set); seed_everything(cfg["seed"])
    #   2. emb = np.load(<artifacts_dir>/item_emb.npy)  -> torch tensor
    #   3. _, _, seqs = load_processed(...);  train_seqs = [make_splits(s).retriever_train for s in seqs]
    #      neighbors = build_coengagement_neighbors(train_seqs, num_items, cfg["rqvae"]["coengagement_window"])
    #      (only retriever_train positions: no leakage from valid/test!)
    #   4. model = RQVAE.from_config(emb.shape[1], cfg["rqvae"])
    #   5. history = train_rqvae(model, emb, cfg["rqvae"], neighbors, device=get_device(cfg["device"]),
    #                            generator=make_generator(cfg["seed"]))
    #   6. save_rqvae(model, out_path, extra={"config": cfg, "history": history})
    #   7. Print a small table: epoch | rec | commitment | uniqueness | used codes per level
    #      and the final uniqueness. (m05 then adds the random last level.)
    cfg = apply_overrides(load_config(args.config), args.set)
    seed_everything(cfg["seed"])
    rq_cfg = cfg["rqvae"]
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    out_path = Path(args.out) if args.out else artifacts_dir / "rqvae.pt"

    emb = torch.from_numpy(np.load(artifacts_dir / "item_emb.npy"))
    items, _, seqs = load_processed(cfg["paths"]["processed_dir"])
    num_items = len(items)
    if emb.shape[0] != num_items:
        raise ValueError(f"item_emb.npy has {emb.shape[0]} rows but there are {num_items} items; re-run m03")

    # Only retriever_train positions: no leakage from valid/test.
    train_seqs = [make_splits(s).retriever_train for s in seqs]
    neighbors = build_coengagement_neighbors(train_seqs, num_items, rq_cfg["coengagement_window"])
    print(f"items: {num_items}  emb dim: {emb.shape[1]}  "
          f"items with co-engaged neighbours: {sum(1 for n in neighbors if n)}")

    model = RQVAE.from_config(emb.shape[1], rq_cfg)
    history = train_rqvae(model, emb, rq_cfg, neighbors, device=get_device(cfg["device"]),
                          generator=make_generator(cfg["seed"]), log=None)
    save_rqvae(model, out_path, extra={"config": cfg, "history": history})

    print(f"\n{'epoch':>6} | {'rec':>8} | {'commitment':>10} | {'uniqueness':>10} | used codes per level")
    print("-" * 70)
    for h in history:
        print(f"{h['epoch']:>6} | {h['rec']:>8.4f} | {h['commitment']:>10.4f} | "
              f"{h['uniqueness']:>10.4f} | {h['used_codes']}")
    print(f"\nfinal uniqueness: {history[-1]['uniqueness']:.4f}  (m05 adds the random last level)")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
