"""The live demo:
    python -m gr2mini.m15_eval_demo.demo --config configs/tiny.yaml --adapter rl_from_sft_rejection_lora --random 3
    python -m gr2mini.m15_eval_demo.demo --config configs/tiny.yaml --adapter rl_from_sft_rejection_lora --user 17
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch GR2 re-rank a real user's candidates.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--adapter", default="rl_from_sft_rejection_lora")
    parser.add_argument("--user", type=int, default=None, help="user index (must have a kept test example)")
    parser.add_argument("--random", type=int, default=1, help="how many random test users to show")
    parser.add_argument("--sample", action="store_true", help="sample (temperature 1.0) instead of greedy")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    # TODO(m15):
    #   1. cfg; items; index; test = load_jsonl(candidates_test.jsonl)
    #   2. pick the example(s): the one with example["user"] == args.user, or args.random random ones
    #   3. model, tok, _ = load_student(..., adapter_dir=<artifacts>/<adapter>, trainable=False)
    #   4. for each: text = generate_chat(model, tok, [policy messages], max_new_tokens, do_sample=args.sample)[0][0]
    #      parsed = parse_rerank_output(text, 10); reordered = apply_ranking(candidates, parsed.ranking)
    #      reward = compute_reward(text, example)
    #      print(render_case(example, items, index, reordered, parsed, reward))
    #   (remember $env:PYTHONUTF8 = "1" for the arrows and product titles)
    raise NotImplementedError("TODO(m15): implement demo.main")


if __name__ == "__main__":
    main()
