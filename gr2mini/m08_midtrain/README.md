# m08 — Stage 1: tokenized mid-training (item alignment)

**Paper:** §2.4, Table 4, Appendix B. Follows OneRec-Think's "item alignment" tasks.
**Time:** ~1 day of coding + 1–3 h of training. **Files you implement:** `tasks.py`, `features.py`,
`trainer.py`, `probe.py`, `run.py`.

## Why

After m07, `<s_a_12>` is just "the mean token". Mid-training **grounds** the SID tokens: this ID is a
moisturizing shampoo, and people who buy X then buy Y. Without this stage, the re-ranker would reason
about strings it doesn't understand.

## The tasks (SIDs interleaved with natural language)

| Task (`tasks.py`) | user turn | assistant turn |
|---|---|---|
| `caption_example`: itemic dense captioning | "What is the product `<SID>`?" | item text (title, brand, category) |
| `reverse_example`: reverse grounding | "Which Semantic ID belongs to this product? *text*" | `<SID>` |
| `sequential_example`: sequential preference | "A customer bought, in order: 1. `<SID>` … Which product will they buy next? …" | `<SID>` |

`build_midtrain_examples` makes one caption + one reverse example per item, plus
`sequential_per_user` next-item examples per user. Those come from the **latest positions of
`seq[:-2]`** (= `valid.history`), so the valid and test targets are never seen. The examples are shuffled
and optionally capped at `max_examples`. The paper's Table 4 compares single-task vs multi-task. Our
flags let you rerun that (`--set midtrain.caption=false ...`).

## Loss masking (`features.py`)

```
prompt  = chat_prompt(messages[:-1], add_generation_prompt=True)    ends with "...assistant\n<think>\n\n</think>\n\n"
full    = chat_prompt(messages,      add_generation_prompt=False)   = prompt + answer + "<|im_end|>\n"
input_ids    = tokenize(full)
loss_weights = 0 for the prompt tokens, 1 for everything after (answer + <|im_end|> + "\n")
```

Learning to emit `<|im_end|>` is how the model learns to *stop*. Check that the prompt's token ids are
a prefix of the full ids. If a tokenizer merged characters across that boundary, the mask would be
off by one. The tests check it.

`loss_weights` is a float, not a bool, on purpose: m13 reuses everything here with weights λ_r and λ_o.

## The training loop (`trainer.py`)

```
weighted_lm_loss = − Σ_t w_t · log p(token_t) / #(tokens with w_t > 0)        (m07 token_logprobs)
```

`train_lm`: AdamW over trainable params only, linear warmup then linear decay, gradient accumulation,
fp16 autocast + `GradScaler` on CUDA (the trainable params are fp32, see m07), and gradient clipping at 1.0.
**`model.train()` first** (the m07 eval-mode trap).

How to normalize with accumulation: each micro-batch loss is already a mean, so divide it by
`grad_accum` before `backward()`. Then an accumulated step is the mean of its micro-batches.

## Probes (`probe.py`): did it learn anything?

* `probe_next_item`: greedy-generate the sequential prompt for held-out *valid* examples, parse the SID
  with `SIDIndex.item_from_text`, and report hit@1 plus the fraction of answers that are valid known SIDs.
* `probe_caption`: generate captions for items and compute word-level F1 against the true title.

Before training, both are ~0. After a full run, expect valid-SID rates near 100 %, caption F1 well above 0,
and next-item hit@1 a few percent. Generation alone retrieves poorly; the paper's Table 4 has hit@5 ≈ 5 %
with an 8B model. That's fine: we only need the re-ranker to *understand* SIDs.

## Budget

Full Beauty ≈ 24k caption/reverse examples + 67k sequential examples ≈ 90k short sequences. One epoch
with batch 8 × accum 2 takes ~2–3 h on your GPU. Use `--set midtrain.max_examples=20000` for a ~40 min
version. `tiny`: 3,000 examples, ~5 min.

## Implementation order & test

`tasks.py` → `features.py` → `trainer.py` → `probe.py` → `run.py`

```powershell
python -m pytest gr2mini/m08_midtrain -v
python -m gr2mini.m08_midtrain.run --config configs/tiny.yaml     # → <artifacts_dir>/midtrain_lora/
```
