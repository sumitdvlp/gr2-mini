# m07 — The student LLM: Qwen3-0.6B + Semantic-ID tokens

**Paper:** §2.4 ("optimize the SID embedding table via next-token prediction"), §5.4 (Qwen3-8B student).
**Time:** ~1 day. **Files you implement:** `sid_tokens.py`, `lora.py`, `logprobs.py`, `generate.py`, `load.py`.
(`testing.py` is mine: a tiny random Qwen3 for fast tests, used by m07/m08/m13/m14.)

## What you'll build

```
Qwen3-0.6B (fp16, 1.2 GB)
  + 2 + L·C new tokens  <|sid_begin|> <|sid_end|> <s_a_0> ... <s_d_255>      sid_tokens.py
  + LoRA on all attention/MLP projections  +  trainable rows for ONLY the new tokens   lora.py
  → token_logprobs(...)     memory-efficient log p(token | prefix)          logprobs.py  (SFT + RL use this)
  → generate_chat(...)      batched chat generation for evaluation           generate.py
  → load_student(...)       everything above, wired together                  load.py
```

---

## 1. Adding the SID tokens: the padding gotcha (verified on your machine)

| | value |
|---|---|
| `model.config.vocab_size` = embedding rows | **151,936** |
| `len(tokenizer)` | **151,669** |
| ids given to the 1,026 new SID tokens | **151,669 … 152,694** |

The embedding matrix has 267 spare rows beyond the tokenizer's vocabulary (padded for GPU efficiency).
The first 267 SID tokens land in those padding rows, which hold *untrained junk*, and the rest need new
rows. So `add_sid_tokens` must:

1. `tokenizer.add_tokens(all_sid_tokens(L, C))` (regular added tokens: each becomes a single token, never split)
2. resize the embedding **only if** `len(tokenizer) > rows`
3. initialize **every** new token's row (including the ones that landed in padding rows!) to the mean of
   the real rows `0 … len_before − 1`

Why the mean? Random rows have the wrong scale. Junk rows have unknown content. The mean embedding is a
"generic token" the model already handles reasonably. A smarter init (e.g. the mean of the item title's
token embeddings) is a nice bonus experiment for m08.

**Tied embeddings:** Qwen3-0.6B shares the input embedding with `lm_head`. One row is both "how the
model reads `<s_a_12>`" and "how it scores writing it". The tests check both views change.

## 2. Training only what we must (`lora.py`)

Full fine-tuning: ~16 bytes × 0.6 B params ≈ 10 GB, impossible. Our trainable set:
* **LoRA** (r = 16) on `q,k,v,o,gate,up,down` projections, ~10 M params
* **only the 1,026 new embedding rows**, via PEFT's `trainable_token_indices={"embed_tokens": ids}`.
  PEFT keeps a small delta for those rows and handles the tied `lm_head`.

Measured on your GPU: 11.1 M trainable params (1.8 %), all fp32 (good: fp16 *trainable* weights would
underflow), and a 1,500-token training step peaks at **2.4 GB**.

⚠️ **The eval-mode trap (I hit this when measuring):** `from_pretrained` returns the model in
**eval mode**, and HF silently skips gradient checkpointing in eval mode. The same step then peaked at
**5.9 GB** instead of 2.4. Always call `model.train()` before training steps.

## 3. Log-probs only where you need them (`logprobs.py`)

A 1,500-token sequence × 152k vocab in fp32 = **0.9 GB of logits**, but SFT and RL only need the ~300
assistant tokens. So instead of `model(...).logits`:

```
hidden = backbone(input_ids, attention_mask).last_hidden_state      [B, T, H]  (cheap: H = 1024)
pick positions t-1 for every target token t                          [n, H]
logits = lm_head(those) → float32 → log_softmax → gather token ids   [n]
```

Shift alignment: the token at position `t` is predicted from the hidden state at position `t − 1`.
Position 0 can never be a target. `get_backbone_and_head` unwraps a PEFT model
(`model.get_base_model()`) and returns `(base.model, base.lm_head)`. With trainable tokens, both may
be PEFT wrappers, and that's fine: call them like modules.

## 4. Generation (`generate.py`)

Batched generation needs **left padding** (`tokenizer.padding_side = "left"`), so all prompts end at
the same position and new tokens line up. Decode only the new tokens, with `skip_special_tokens=True`.
SID tokens are *added*, not *special*, so they survive. `<|im_end|>` is special, so it's removed.

Use the chat template with `enable_thinking=False`. Even then, Qwen3 puts an *empty*
`<think>\n\n</think>\n\n` at the start of the assistant turn. The generation prompt already contains
it, so completions start directly with the answer.

Speed on your GPU: ~8–10 new tokens/s per step (batched sequences share steps). LoRA layers add Python
overhead per token. That's slow, and it's why the RL budget is small.

---

## Implementation order

`sid_tokens.py` → `logprobs.py` → `lora.py` → `generate.py` → `load.py`

```powershell
python -m pytest gr2mini/m07_llm -v                       # tiny random model, ~30 s
python -m pytest gr2mini/m07_llm -m integration -v        # real Qwen3-0.6B on your GPU (downloads 1.2 GB once)
python -m gr2mini.m07_llm.run --config configs/tiny.yaml  # loads the student, chats, shows SID tokenization
```

The tests download only the Qwen3 *tokenizer* (a few MB, already cached on your machine).
