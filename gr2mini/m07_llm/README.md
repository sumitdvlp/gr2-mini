# m07 — The student LLM: Qwen3-0.6B + Semantic-ID tokens  🟡 preview

**Paper:** §2.4 ("optimize the SID embedding table via next-token prediction"), §5.4 (Qwen3-8B student).
**Status:** lesson preview. Stubs and tests come when you get here.

## What you'll build

1. **Load Qwen3-0.6B** in fp16 (1.2 GB) with SDPA attention and gradient checkpointing. Your 1660 Ti
   has no native bf16 (m00).
2. **Extend the vocabulary** with the `2 + L·C` SID tokens from `m05.all_sid_tokens` (1,026 for
   L = 4, C = 256), so `<s_a_12>` is a single token.
3. **Make only the right things trainable:** LoRA adapters on attention + MLP, plus *only the new rows*
   of the embedding table. Training the full 151k × 1024 embedding with Adam would cost ~2.5 GB.
4. **Memory-efficient log-probs** `token_logprobs(model, input_ids, attention_mask, target_mask)`. SFT
   (m13) and RL (m14) both build on it.

## Key ideas you'll learn

* **Vocabulary padding gotcha:** Qwen3's embedding matrix has 151,936 rows, but its tokenizer has only
  151,669 tokens. New tokens get ids starting at 151,669, so the first 267 land in *existing padding
  rows*, and `resize_token_embeddings` must account for that. We'll write a test that checks every
  new id has its own trainable row.
* **Initializing new embeddings:** random vs. the mean of existing embeddings vs. the mean of the
  embeddings of the item's title tokens. The init matters a lot for how fast mid-training converges.
* **Tied embeddings:** in Qwen3-0.6B the input embedding and the output `lm_head` share weights. A new
  SID row is both "how the model reads `<s_a_12>`" and "how it scores writing it".
* **Computing logits only where needed:** a 1,500-token sequence × 152k vocab in fp32 is ~0.9 GB of
  logits. If the loss only covers the ~300 assistant tokens, run the final `lm_head` on those hidden
  states only: 5× less memory.
* **Chat templates:** Qwen3 has a "thinking" mode. We disable it (`enable_thinking=False`) because GR2's
  reasoning lives in the JSON `explanation` field. ⚠️ Even then, the template inserts an *empty*
  `<think>\n\n</think>\n\n` block at the start of every assistant turn. Your loss masks must handle it.

**Verified on your machine** (transformers 5.22): config vocab 151,936, tokenizer 151,669, tied
embeddings, and the 1,026 SID tokens get ids 151,669–152,694. Each SID token is a single token, but
`"Buy <|sid_begin|>"` tokenizes the space before a SID as its own `Ġ` token: `['Buy', 'Ġ', '<|sid_begin|>', ...]`.
That's harmless, but it's why prompt formatting should be consistent. PEFT 0.21's `LoraConfig` supports
`trainable_token_indices`, which trains only the new rows.

## Planned interface (may be refined)

`load.py: load_student(cfg) -> (model, tokenizer)` · `sid_tokens.py: add_sid_tokens, check_sid_tokenization` ·
`lora.py: apply_lora_with_trainable_tokens` · `logprobs.py: token_logprobs` · tests use a *tiny randomly
initialized* Qwen3 config (2 layers, hidden 64), so they run in seconds without a download.
