# m08 — Stage 1: tokenized mid-training (item alignment)  🟡 preview

**Paper:** §2.4, Table 4, Appendix B (prompt examples). Follows OneRec-Think's "item alignment" tasks.
**Status:** lesson preview. Stubs and tests come when you get here.

## Why

Right after m07, the SID tokens are meaningless vectors: the LLM has never seen `<s_a_12>`.
Mid-training **grounds** them, by teaching the model that this SID is "moisturizing shampoo", and that
people who buy X then buy Y. Without this stage, the re-ranker would reason about strings it doesn't
understand.

## The tasks (interleaving SIDs with natural language)

| Task | Input (user turn) | Target (assistant turn) |
|---|---|---|
| Itemic dense captioning | "What is `<SID>`?" | "Title: … Category: …" |
| Reverse grounding | "Which item is *Title: …*?" | `<SID>` |
| Sequential preference | "The user bought `<SID1>`, `<SID2>`, … What next?" | `<SID_next>` |
| General language (small mix) | ordinary instruction data | ordinary answers, against catastrophic forgetting |

The paper's Table 4 compares **single-task** vs **multi-task** mixtures. We'll make the mixture configurable.

## Key ideas you'll learn

* **Loss masking:** compute the loss only on assistant tokens (labels = −100 elsewhere).
* **Leakage discipline:** the sequential task only uses `seq[:-2]` (never valid/test targets).
* **Packing vs padding,** gradient accumulation, and fp16 training with `GradScaler` on a 6 GB card.
* **Probing what the model learned:** caption accuracy for unseen SIDs, and next-SID hit rate with
  *constrained decoding* (only valid SID continuations allowed, via a prefix trie).

## Planned interface (may be refined)

`tasks.py: caption_examples, reverse_examples, sequential_examples` · `collate.py: build_chat_batch`
(masks!) · `train.py: train_midtrain` · `probe.py` · `run.py` → `artifacts/midtrain_lora/`

## Budget

Full Beauty, 1 epoch, LoRA on Qwen3-0.6B: roughly 1.5–3 h on your GPU. `tiny`: ~10 min.
