"""Try it:  python -m gr2mini.m14_dapo.run --config configs/tiny.yaml --init sft_rejection_lora

Stage 3 of GR2. --init names the adapter folder in <artifacts_dir> to start from:
    sft_rejection_lora -> "RL-rejection",  sft_targeted_lora -> "RL-targeted",  midtrain_lora -> "RL-zeroshot"
Saves <artifacts_dir>/rl_from_<init>/ and <artifacts_dir>/rl_from_<init>_history.json.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="DAPO reinforcement learning.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--init", required=True, help="adapter dir name inside artifacts_dir")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    # TODO(m14):
    #   1. cfg; items; index; train = load_jsonl(candidates_train.jsonl); valid = candidates_valid[: rl.eval_examples]
    #   2. model, tok, _ = load_student(cfg["llm"], L, C, device, adapter_dir=<artifacts>/<init>, trainable=True)
    #   3. gen = greedy generate_chat with eval.max_new_tokens
    #      eval_fn = lambda m: {f"valid_{k}": v for k, v in evaluate_reranker(valid, items, index,
    #                                                       cfg["prompts"], gen).items() if k != "n"}
    #      print eval_fn(model) BEFORE training
    #   4. history = train_dapo(model, tok, train, items, index, cfg["rl"], cfg["prompts"], seed, eval_fn)
    #   5. model.save_pretrained(<artifacts>/rl_from_<init>); dump the history to JSON
    #   6. Print a compact table: step | reward | parse | identity | kept/drawn | clip | valid R@1
    raise NotImplementedError("TODO(m14): implement run.main")


if __name__ == "__main__":
    main()
