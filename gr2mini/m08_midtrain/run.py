"""Try it:  python -m gr2mini.m08_midtrain.run --config configs/tiny.yaml

Stage 1 of GR2: teaches the student what SIDs mean. Saves <artifacts_dir>/midtrain_lora/.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenized mid-training (item alignment).")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    # TODO(m08):
    #   1. cfg; items, _, seqs = load_processed(...); index = SIDIndex.load(sids.json)
    #   2. model, tok, _ = load_student(cfg["llm"], index.num_levels, index.codebook_size, device)
    #   3. examples = build_midtrain_examples(items, index, seqs, cfg["midtrain"], seed)
    #      features = [f for f in (tokenize_chat(tok, ex, max_len) for ex in examples) if f]
    #      print counts per task and the mean / max token length
    #   4. gen = lambda convs: [o[0] for o in generate_chat(model, tok, convs, max_new_tokens=48, batch_size=8)]
    #      probe BEFORE training: probe_next_item on probe_examples valid examples
    #      (make_splits(s).valid), probe_caption on probe_examples random items
    #   5. history = train_lm(model, features, cfg["midtrain"], tok.pad_token_id, device, make_generator(seed))
    #   6. model.save_pretrained(<artifacts_dir>/midtrain_lora)
    #   7. probe AFTER training and print before/after side by side, plus 3 example captions.
    raise NotImplementedError("TODO(m08): implement run.main")


if __name__ == "__main__":
    main()
