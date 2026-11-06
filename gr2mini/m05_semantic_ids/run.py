"""Try it:  python -m gr2mini.m05_semantic_ids.run --config configs/tiny.yaml

Assigns final Semantic IDs to every item, saves <artifacts_dir>/sids.json, and shows SID siblings.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from gr2mini.m00_setup.config import apply_overrides, load_config
from gr2mini.m00_setup.device import get_device
from gr2mini.m00_setup.seed import make_generator, seed_everything
from gr2mini.m02_data.storage import load_processed
from gr2mini.m04_rqvae.model import load_rqvae
from gr2mini.m05_semantic_ids.assign import assign_semantic_ids, compute_semantic_codes, prefix_stats
from gr2mini.m05_semantic_ids.sid_index import SIDIndex


def _short(text: str, n: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign Semantic IDs.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--show", type=int, default=3, help="random items to display with their siblings")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    seed_everything(cfg["seed"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    codebook_size = cfg["rqvae"]["codebook_size"]
    sid_cfg = cfg["sids"]
    device = get_device(cfg["device"])

    # 1. Model, embeddings, item texts.
    model = load_rqvae(artifacts_dir / "rqvae.pt", map_location="cpu")
    emb = torch.from_numpy(np.load(artifacts_dir / "item_emb.npy"))
    items, _, _ = load_processed(cfg["paths"]["processed_dir"])
    if emb.shape[0] != len(items):
        raise ValueError(f"item_emb.npy has {emb.shape[0]} rows but there are {len(items)} items; re-run m03")

    # 2. Full pipeline: deterministic codes -> random last levels -> collision resolution.
    codes, report = assign_semantic_ids(
        model, emb, sid_cfg["random_last_levels"], codebook_size,
        generator=make_generator(cfg["seed"]),
        max_rounds=sid_cfg.get("max_collision_rounds", 100),
        device=device,
    )

    # 3. Report (README §4) + prefix stats of the deterministic codes.
    print(f"items: {codes.shape[0]}  levels: {codes.shape[1]}  codebook size: {codebook_size}  "
          f"random last levels: {sid_cfg['random_last_levels']}")
    print(f"\nuniqueness_semantic    : {report['uniqueness_semantic']:.4f}  (deterministic codes)")
    print(f"uniqueness_random_last : {report['uniqueness_random_last']:.4f}  (after Eq. 10)")
    print(f"uniqueness_final       : {report['uniqueness_final']:.4f}  (after collision resolution)")
    print(f"items re-drawn         : {report['num_redrawn']}")
    print("utilization per level  : " + "  ".join(f"{u:.2%}" for u in report["utilization"]))

    semantic = compute_semantic_codes(model, emb, device=device)
    print("\nprefix stats (deterministic codes)")
    print(f"{'level':>6} | {'distinct':>8} | {'largest':>8} | {'mean':>8}")
    print("-" * 40)
    for st in prefix_stats(semantic):
        print(f"{st['level']:>6} | {st['distinct']:>8} | {st['largest_group']:>8} | {st['mean_group']:>8.2f}")

    # 4. Save the index.
    index = SIDIndex(codes, codebook_size)
    out_path = artifacts_dir / "sids.json"
    index.save(out_path)
    print(f"\nsaved -> {out_path}")

    # 5. A few random items with their SID siblings.
    rng = random.Random(cfg["seed"])
    for i in rng.sample(range(index.num_items), min(args.show, index.num_items)):
        sid = index.sid(i)
        print(f"\n=== item {i}  {index.sid_str(i)}")
        print(f"    {_short(items[i]['text'])}")
        lvl2 = [j for j in index.items_with_prefix(sid[:2]) if j != i]
        lvl2_set = set(lvl2)
        lvl1 = [j for j in index.items_with_prefix(sid[:1]) if j != i and j not in lvl2_set]
        print(f"  siblings sharing first 2 codes {sid[:2]} ({len(lvl2)}):")
        for j in lvl2[:5]:
            print(f"    {j:>6}  {_short(items[j]['text'])}")
        print(f"  siblings sharing only first code {sid[:1]} ({len(lvl1)}):")
        for j in lvl1[:5]:
            print(f"    {j:>6}  {_short(items[j]['text'])}")


if __name__ == "__main__":
    main()
