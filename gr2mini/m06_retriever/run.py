"""Try it:  python -m gr2mini.m06_retriever.run --config configs/tiny.yaml

Trains SASRec on retriever_train sequences and writes the candidate lists every later module uses:
    <artifacts_dir>/sasrec.pt
    <artifacts_dir>/candidates_train.jsonl   (from rerank_train examples)
    <artifacts_dir>/candidates_valid.jsonl
    <artifacts_dir>/candidates_test.jsonl
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SASRec and build candidate lists.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    # TODO(m06):
    #   1. cfg + overrides, seed_everything; items, user_ids, seqs = load_processed(...)
    #   2. splits = [make_splits(s) for s in seqs]
    #   3. model = SASRec(len(items), **retriever hyper-parameters); device = get_device(cfg["device"])
    #   4. history = train_sasrec(model, [sp.retriever_train for sp in splits],
    #                             [sp.valid for sp in splits], cfg["retriever"], device, make_generator(seed))
    #      torch.save({"init_kwargs": model.init_kwargs, "state_dict": model.state_dict()}, sasrec.pt)
    #   5. For split name, attribute in [("train", "rerank_train"), ("valid", "valid"), ("test", "test")]:
    #        exs = [getattr(sp, attribute) for sp in splits]
    #        cands = topk_candidates(model, [e.history for e in exs], k=num_candidates, device=device)
    #        kept = build_candidate_examples(list(range(len(exs))), exs, cands)
    #        save_jsonl(kept, artifacts_dir / f"candidates_{name}.jsonl")
    #        print: kept / total (the hit rate = Recall@10 on that split)
    #   6. Print the test PRE-RANK baseline: ranking_report([c["candidates"] for c in kept_test],
    #      [c["target"] for c in kept_test]). This is your "Pre-rank" row for the final table.
    #   7. Show one kept test example: history titles, then the 10 candidates with the target marked.
    raise NotImplementedError("TODO(m06): implement run.main")


if __name__ == "__main__":
    main()
