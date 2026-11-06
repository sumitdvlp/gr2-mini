"""Try it:  python -m gr2mini.m09_prompts.run --config configs/tiny.yaml

Prints the three prompt variants for one real candidate record, and token statistics.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Show re-ranking prompts.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    # TODO(m09):
    #   1. cfg; items; index; train = load_jsonl(<artifacts_dir>/candidates_train.jsonl)
    #   2. print build_rerank_messages(train[0], ..., "policy") and the TARGETED user turn
    #   3. tokenizer = AutoTokenizer.from_pretrained(cfg["llm"]["model_name"]); add the SID tokens
    #      (hint: add_sid_tokens needs a model; tokenizer.add_tokens(all_sid_tokens(...)) is enough here)
    #   4. token lengths of chat_prompt(policy messages) over the first 200 records: mean / max
    #   5. an example assistant_json(...) answer
    raise NotImplementedError("TODO(m09): implement run.main")


if __name__ == "__main__":
    main()
