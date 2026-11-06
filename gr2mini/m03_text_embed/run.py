"""Try it:  python -m gr2mini.m03_text_embed.run --config configs/tiny.yaml

Embeds every item's text and shows nearest neighbours of a few random items.
"""
from __future__ import annotations

import argparse

from gr2mini.m00_setup.config import load_config
from gr2mini.m02_data.storage import load_processed
from gr2mini.m03_text_embed.encoder import embed_items_cached, load_sentence_encoder
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed item texts.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--show", type=int, default=3, help="number of random query items to display")
    args = parser.parse_args()

    # TODO(m03):
    #   1. cfg = load_config(...); items, _, _ = load_processed(cfg["paths"]["processed_dir"])
    #   2. texts = [it["text"] for it in items]
    #   3. emb = embed_items_cached(texts, <artifacts_dir>/item_emb.npy,
    #                               load_encoder=lambda: load_sentence_encoder(name, device), ...)
    #   4. Print emb.shape, dtype, and the mean row norm (should be 1.0).
    #   5. For `args.show` random items (seeded by cfg["seed"]): print the query text and its
    #      top-5 neighbours from cosine_top_k with their similarity scores.
    #   6. Run it twice. The second run should be instant (cache hit).
    cfg = load_config(args.config)
    items, _, _ = load_processed(cfg["paths"]["processed_dir"])
    texts = [it["text"] for it in items]
    emb = embed_items_cached(
        texts,
        cfg["paths"]["artifacts_dir"] + "/item_emb.npy",
        lambda: load_sentence_encoder(cfg["text_embed"]["model_name"], cfg["device"]),
        cfg["text_embed"]["model_name"],
        batch_size=cfg["text_embed"]["batch_size"],
        normalize=cfg["text_embed"]["normalize"]
    )
    
    print(f"emb.shape: {emb.shape}, dtype: {emb.dtype}, mean row norm: {np.mean(np.linalg.norm(emb, axis=1)):.4f}")

if __name__ == "__main__":
    main()
