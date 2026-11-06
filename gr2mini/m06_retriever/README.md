# m06 — Retriever: SASRec → top-10 candidate lists  🟡 preview

**Paper:** §5.1 ("Pre-rank" rows of Tables 5–6). Background: SASRec (Kang & McAuley, 2018).
**Status:** lesson preview. Stubs and tests come when you get here.

## Why this module exists

A re-ranker needs something to re-rank. In the paper, the mid-trained Qwen3-8B generates the top-10
list ("Pre-rank (MTL)": R@1 = 0.289 on Beauty). An 8B retriever can't run on 6 GB, so we substitute
**SASRec**, a 2-layer causal transformer over item IDs. It trains in minutes and has a similar
Recall@10 on Beauty.

## What you'll build

```
retriever_train sequences ──► SASRec (d=64, 2 blocks, max_len=50, causal self-attention)
                               trained with full-softmax cross-entropy over all 12k items
  for each split example (history → target):
      scores = SASRec(history)       mask items already in history
      candidates = top-10
      keep the example only if target ∈ candidates   ← the paper's setting (Eq. 14 needs r^D)
  ──► candidates_{train,valid,test}.jsonl  {user, history, candidates, target, target_rank}
```

## Key ideas you'll learn

* **Causal self-attention for sequences:** each position predicts the next item.
  `nn.TransformerEncoder` with a causal mask, plus learned positional embeddings and a padding index.
* **Full softmax vs. sampled negatives:** the original SASRec uses BCE with one negative. For 12k items a
  full softmax is cheap and works better.
* **Why the 4-way split from m02 matters now:** SASRec never saw the `rerank_train` targets, so the
  candidate lists for re-ranker training look like the test lists.
* **Selection bias:** only ~7–9 % of examples have the target in the top-10, so the re-ranker is trained
  and evaluated on the "retriever succeeded" subset. The paper's Recall@9 ≈ 0.95 reflects this filtering.

## Planned interface (may be refined)

`model.py: SASRec` · `train.py: train_sasrec, evaluate_full_ranking` ·
`candidates.py: topk_candidates, build_candidate_split` · `run.py`

## Expected

Full-ranking Recall@10 ≈ 0.06–0.09 on valid, so about 1.5–2k kept examples per split. The pre-rank
metrics on the kept test examples are your "Pre-rank" row: R@1 ≈ 0.2–0.3.
