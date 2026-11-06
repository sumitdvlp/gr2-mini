"""Try it:  python -m gr2mini.m04_rqvae.run --config configs/tiny.yaml [--set rqvae.use_ema=false ...]

Trains the RQ-VAE on the item embeddings from m03 and saves <artifacts_dir>/rqvae.pt.
"""
from __future__ import annotations

import argparse


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
    c


if __name__ == "__main__":
    main()
