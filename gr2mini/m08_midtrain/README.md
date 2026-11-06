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
and gradient clipping at 1.0. fp16 autocast + `GradScaler` only if the model holds fp16 weights. On
your card we use fp32 (m00 correction), so it's plain fp32 training.
**`model.train()` first** (the m07 eval-mode trap).

How to normalize with accumulation: each micro-batch loss is already a mean, so divide it by
`grad_accum` before `backward()`. Then an accumulated step is the mean of its micro-batches.

## Learning rates: what I found by experiment

The new SID rows start as identical copies of "the mean token", and they have to become 1,026
*distinguishable* vectors. At typical LoRA settings that barely happens. My reference runs taught
200 items (caption + reverse tasks) and then asked for each item's SID back:

| LoRA `lr` | SID-row `token_lr` | epochs | generated SIDs that are real items | exact item recall | caption F1 |
|---|---|---|---|---|---|
| 2e-4 | (same) | 5 | 0 % | 0 % | 0.01 |
| 2e-4 | 2e-3 | 5 | 98 % | 0 % | 0.02 |
| **1e-3** | **5e-3** | **10** | **100 %** | **100 %** | **0.66** |

Lessons: (1) **a separate, larger lr for the new embedding rows matters.** That's why `train_lm` takes
`token_lr` (two AdamW param groups). (2) Grounding is memorization, so every item needs **repeated
exposure**. One pass over the data is not enough. Hence `lr: 0.001, token_lr: 0.005, epochs: 2` in the
config. If your probes stay at ~0 after training, this is the first thing to check.

## Probes (`probe.py`): did it learn anything?

* `probe_next_item`: greedy-generate the sequential prompt for held-out *valid* examples, parse the SID
  with `SIDIndex.item_from_text`, and report hit@1 plus the fraction of answers that are valid known SIDs.
* `probe_caption`: generate captions for items and compute word-level F1 against the true title.

Before training, both are ~0. After a full run, expect valid-SID rates near 100 %, caption F1 well above 0,
and next-item hit@1 a few percent. Generation alone retrieves poorly; the paper's Table 4 has hit@5 ≈ 5 %
with an 8B model. That's fine: we only need the re-ranker to *understand* SIDs.

## Budget

Measured with the reference on your GPU (fp32): **2.2 s per optimizer step** (batch 8 × accum 2 of
~73-token examples), peak 4.25 GB. Full Beauty ≈ 24k caption/reverse + 67k sequential ≈ 90k examples
≈ 5.6k steps ≈ **3.5 h per epoch**, so 2 epochs ≈ 7 h (an overnight run).
`--set midtrain.max_examples=30000` gives a ~2.3 h version (2 epochs). `tiny` (3,000 examples × 3 epochs): ~20 min.

**Don't judge mid-training on `tiny`.** In my smoke run the model learned the SID *format* (it emitted
well-formed `<|sid_begin|><s_a_41>…<|sid_end|>`), but the SIDs it produced belonged to no real item.
`tiny` has 7.4k items and only 3k examples, so most items are never seen even once. Grounding needs
several exposures per item.

## Implementation order & test

`tasks.py` → `features.py` → `trainer.py` → `probe.py` → `run.py`

```powershell
python -m pytest gr2mini/m08_midtrain -v
python -m gr2mini.m08_midtrain.run --config configs/tiny.yaml     # → <artifacts_dir>/midtrain_lora/
```
