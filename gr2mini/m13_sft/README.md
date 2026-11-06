# m13 — Stage 2: supervised fine-tuning on reasoning traces  🟡 preview

**Paper:** §4.2, Eq. 13, Table 5 "SFT-*" rows.
**Status:** lesson preview. Stubs and tests come when you get here.

## The loss

```
L_SFT = − λ_r · Σ_i log P(r_i | prompt, r_<i)          reasoning tokens (the "explanation")
        − λ_o · Σ_j log P(o_j | prompt, τ, o_<j)       output tokens (the "recommendations")
        with λ_r < λ_o                                                                   (13)
```

**Why two weights?** The ranking is what gets evaluated. The reasoning is a means to an end, and it
comes from a teacher that may be imperfect. Down-weighting it (we'll start at λ_r = 0.5, λ_o = 1.0)
keeps the student from over-fitting the teacher's prose.

## Key ideas you'll learn

* **Token-span weights:** find which tokens belong to the explanation vs the recommendations, using the
  tokenizer's `offset_mapping` (character spans → token spans). This is a classic source of
  off-by-one bugs, and the tests use hand-made examples.
* **Normalization choices:** sum vs. mean, and per-sequence vs. per-batch. We'll pin one down and
  test it.
* **The paper's surprising finding:** SFT alone can make Recall@1 *worse* than the pre-rank
  (0.2006 vs 0.2892 for SFT-targeted-KP). Imitating explanations doesn't teach the model to rank.
  That's why Stage 3 exists. You'll measure this yourself.
