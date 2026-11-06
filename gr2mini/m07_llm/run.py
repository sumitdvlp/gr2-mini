"""Try it:  python -m gr2mini.m07_llm.run --config configs/tiny.yaml

Loads the student with SID tokens and shows that it works (before any training).
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the student LLM and poke at it.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    # TODO(m07):
    #   1. cfg; index = SIDIndex.load(<artifacts_dir>/sids.json)
    #   2. model, tok, ids = load_student(cfg["llm"], index.num_levels, index.codebook_size, device)
    #   3. print trainable_summary(model), check_sid_tokenization(...), the first/last new id,
    #      and vram_info(device)
    #   4. print tok.convert_ids_to_tokens(tok.encode("I like " + index.sid_str(0), add_special_tokens=False))
    #   5. generate_chat on [{"role": "user", "content": f"What is the product {index.sid_str(0)}?"}]
    #      -> nonsense is EXPECTED: the SID rows are still just "the mean token". m08 fixes this.
    raise NotImplementedError("TODO(m07): implement run.main")


if __name__ == "__main__":
    main()
