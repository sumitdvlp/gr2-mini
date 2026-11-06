"""Try it:  python -m gr2mini.m13_sft.run --config configs/tiny.yaml --traces rejection

Stage 2 of GR2: SFT on accepted teacher traces, starting from the mid-trained adapter.
Saves <artifacts_dir>/sft_<traces>_lora/ and prints valid metrics before/after.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT on reasoning traces (Eq. 13).")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--traces", choices=["targeted", "rejection"], required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    # TODO(m13):
    #   1. cfg; items; index; traces = [r for r in load_jsonl(traces_<mode>.jsonl) if r["accepted"]]
    #      valid = load_jsonl(candidates_valid.jsonl)[: eval.max_examples]
    #   2. model, tok, _ = load_student(cfg["llm"], L, C, device, adapter_dir=<artifacts>/midtrain_lora)
    #   3. features = [sft_feature(tok, r["messages"], lambda_r, lambda_o, max_len) ...] (drop None)
    #   4. gen = lambda convs: [o[0] for o in generate_chat(model, tok, convs, eval.max_new_tokens,
    #                                                     batch_size=eval.batch_size)]
    #      print pre_rank_report(valid) and evaluate_reranker(valid, ..., gen) BEFORE training
    #   5. train_lm(model, features, cfg["sft"], tok.pad_token_id, device, make_generator(seed))
    #   6. model.save_pretrained(<artifacts>/sft_<mode>_lora); evaluate AFTER; print both rows
    #      and one generated explanation.
    raise NotImplementedError("TODO(m13): implement run.main")


if __name__ == "__main__":
    main()
