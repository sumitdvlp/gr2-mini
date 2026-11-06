"""Try it:  python -m gr2mini.m15_eval_demo.run --config configs/tiny.yaml

Evaluates every available checkpoint on the kept TEST examples and writes <artifacts_dir>/results.md.
"""
from __future__ import annotations

import argparse

ADAPTERS = [
    "midtrain_lora",
    "sft_targeted_lora",
    "sft_rejection_lora",
    "rl_from_midtrain_lora",
    "rl_from_sft_targeted_lora",
    "rl_from_sft_rejection_lora",
]
COLUMNS = ["R@1", "R@5", "R@9", "N@5", "N@10", "parse_rate", "identity_rate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation table (like Tables 5-6).")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    # TODO(m15):
    #   1. cfg; items; index; test = load_jsonl(candidates_test.jsonl)[: eval.max_examples]
    #   2. rows = {"Pre-rank": pre_rank_report(test)}
    #   3. for each name in ADAPTERS whose folder exists: load_student(..., adapter_dir, trainable=False),
    #      evaluate_reranker with greedy generate_chat (return_outputs=True), store the row;
    #      R@1 hits per example -> bootstrap_ci -> print "R@1 0.291 [0.268, 0.314]"
    #      free the model (del model; torch.cuda.empty_cache()) before loading the next one
    #   4. table = markdown_table(rows, COLUMNS); print it; write it to <artifacts_dir>/results.md
    raise NotImplementedError("TODO(m15): implement run.main")


if __name__ == "__main__":
    main()
