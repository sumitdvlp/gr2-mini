"""Try it:  python -m gr2mini.m02_data.run --config configs/tiny.yaml

Builds data/processed*/items.jsonl and sequences.json from the raw Amazon files.
"""
from __future__ import annotations

import argparse


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
    raise NotImplementedError("TODO(m02): implement run.main")


if __name__ == "__main__":
    main()
