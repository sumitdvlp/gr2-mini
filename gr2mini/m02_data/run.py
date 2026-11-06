"""Try it:  python -m gr2mini.m02_data.run --config configs/tiny.yaml

Builds data/processed*/items.jsonl and sequences.json from the raw Amazon files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gr2mini.m00_setup.config import load_config
from gr2mini.m02_data.download import download_file
from gr2mini.m02_data.parse import read_reviews
from gr2mini.m02_data.sequences import (
    k_core_filter,
    subsample_users,
    build_sequences,
    remap_ids,
    dataset_stats,
)
from gr2mini.m02_data.item_text import build_item_record
from gr2mini.m02_data.parse import read_meta
from gr2mini.m02_data.storage import save_processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess Amazon Beauty.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    args = parser.parse_args()

    # TODO(m02):
    #   1. cfg = load_config(args.config); raw_dir = Path(cfg["paths"]["raw_dir"])
    #   2. download_file(...) both URLs into raw_dir (file name = last part of the URL)
    #   3. interactions = read_reviews(...)
    #   4. interactions = k_core_filter(interactions, cfg["data"]["kcore"])
    #   5. interactions = subsample_users(interactions, cfg["data"]["max_users"], cfg["seed"])
    #   6. seqs, user_ids, asins = remap_ids(build_sequences(interactions))
    #   7. meta = read_meta(..., keep=set(asins))
    #      items = [build_item_record(i, a, meta.get(a)) for i, a in enumerate(asins)]
    #   8. save_processed(cfg["paths"]["processed_dir"], items, user_ids, seqs)
    #   9. Print dataset_stats(seqs, len(items)), how many items have no title, and one
    #      example user's sequence as item texts (oldest first). Compare with Table 1.
    config = load_config(args.config)
    raw_dir = Path(config["paths"]["raw_dir"])
    download_reviews_path = download_file(config["data"]["reviews_url"], raw_dir / "reviews.json.gz")
    print(f"Downloaded reviews to {download_reviews_path}")
    download_meta_path = download_file(config["data"]["meta_url"], raw_dir / "meta.json.gz")
    print(f"Downloaded meta to {download_meta_path}")

    interactions = read_reviews(download_reviews_path)
    interactions = k_core_filter(interactions, config["data"]["kcore"])
    interactions = subsample_users(interactions, config["data"]["max_users"], config["seed"])
    
    seqs, user_ids, asins = remap_ids(build_sequences(interactions))
    
    meta = read_meta(download_meta_path, keep=set(asins))
    items = [build_item_record(i, a, meta.get(a)) for i, a in enumerate(asins)]
    
    save_processed(config["paths"]["processed_dir"], items, user_ids, seqs)
    
    print(dataset_stats(seqs, len(items)))

if __name__ == "__main__":
    main()
