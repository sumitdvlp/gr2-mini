"""Try it:  python -m gr2mini.m05_semantic_ids.run --config configs/tiny.yaml

Assigns final Semantic IDs to every item, saves <artifacts_dir>/sids.json, and shows SID siblings.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign Semantic IDs.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--show", type=int, default=3, help="random items to display with their siblings")
    args = parser.parse_args()

    # TODO(m05):
    #   1. cfg (with overrides); model = load_rqvae(<artifacts_dir>/rqvae.pt); emb = np.load(item_emb.npy)
    #   2. codes, report = assign_semantic_ids(model, emb, cfg["sids"]["random_last_levels"],
    #                                          cfg["rqvae"]["codebook_size"], make_generator(cfg["seed"]), ...)
    #   3. Print the report (README §4) and prefix_stats of the deterministic codes
    #      (hint: compute_semantic_codes again, or return them from your pipeline).
    #   4. index = SIDIndex(codes, codebook_size); index.save(<artifacts_dir>/sids.json)
    #   5. For `args.show` random items: print item id, index.sid_str(i), the item text, then
    #      up to 5 siblings sharing the first TWO codes (index.items_with_prefix(sid[:2]), excluding
    #      the item) and up to 5 sharing only the FIRST code, with their texts. Level-2 groups are
    #      tiny (~1.6 items), so the level-1 siblings are where you'll see the semantics.
    raise NotImplementedError("TODO(m05): implement run.main")


if __name__ == "__main__":
    main()
