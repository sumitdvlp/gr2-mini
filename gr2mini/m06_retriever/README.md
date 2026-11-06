# m06 — Retriever: SASRec → top-10 candidate lists

**Paper:** §5.1 ("Pre-rank" rows of Tables 5–6). Background: SASRec (Kang & McAuley, 2018).
**Time:** ~1 day. **Files you implement:** `model.py`, `train.py`, `candidates.py`, `run.py`.

## Why this module exists

A re-ranker needs something to re-rank. In the paper, the mid-trained Qwen3-8B generates a top-10
list ("Pre-rank (MTL)": R@1 = 0.289 on Beauty). An 8B retriever can't run on 6 GB, so we substitute
**SASRec**: a 2-layer causal transformer over item IDs. It trains in minutes and has a similar Recall@10.

```
retriever_train sequences ─► SASRec ─► for each rerank_train / valid / test example (m02 splits):
                                           scores over all items, items in the history masked out
                                           candidates = top-10
                                           keep ONLY if target ∈ candidates      ← paper's setting
                              ─► artifacts/candidates_{train,valid,test}.jsonl
     {"user": 17, "history": [...], "candidates": [10 item ids], "target": 523, "target_rank": 3}
```

This record format is the input of **every later module**.

---

## 1. The model

```
ids  (left-padded, item id + 1, 0 = pad)        [B, T]          T = max_len = 50
x  = item_emb(ids) · √H  +  pos_emb(0..T-1)     [B, T, H]       H = 64
x  = dropout(x)
for each block: x = TransformerEncoderLayer(x, mask=attention_mask(ids))   (pre-norm, batch_first)
h  = LayerNorm(x)
scores(h_t) = h_t · item_emb.weight[1:]ᵀ        [B, N_items]    (tied input/output embeddings)
```

### Initialization: the 7× bug I hit while checking this module

`nn.Embedding` initializes weights with **N(0, 1)**. Because the *same* table also scores the output
(`h · item_embᵀ` over 12k items), the initial logits are huge and the softmax is saturated. Training
then crawls. My reference run on full Beauty, everything else equal:

| item/pos embedding init | dropout | best valid R@10 | N@10 |
|---|---|---|---|
| default N(0, 1) | 0.5 | **0.014** | 0.007 |
| N(0, 0.02) | 0.2 | 0.096 | 0.058 |
| N(0, 0.02) | 0.5 | **0.102** | 0.062 |

So: `nn.init.normal_(std=0.02)` for both embeddings, then re-zero the pad row. This is the BERT/GPT
convention, and it's why most SASRec code has an explicit init. The tests check it.

**Left padding** puts the most recent item at the last position, `h[:, -1]`, for every sequence. That
position's output is "the user's state now".

### The attention mask: a NaN trap

`attention_mask(ids, num_heads)` returns a bool tensor `[B·heads, T, T]`. `True` means "may NOT attend":

* causal: query `i` may not look at key `j > i`
* padding: real tokens may not look at pad keys
* **but a pad query may always look at itself.**

Why that last rule? A pad position at the start of a left-padded sequence has no allowed keys at all
(everything before it is pad, everything after it is the future). A softmax over an all-masked row is
`0/0 = NaN`, and in the next layer that NaN leaks into real tokens through `0 · NaN = NaN`. Letting pad
queries see themselves keeps every row non-empty. Their outputs are garbage, but nobody reads them.

`nn.TransformerEncoderLayer` wants a 3-D mask of shape `[B·num_heads, T, T]`, ordered batch-major:
rows `b·heads .. b·heads + heads − 1` belong to sequence `b`. Hint: `repeat_interleave`.

## 2. Training: next-item prediction at every position

From `[i1, i2, i3, i4]`: inputs `[i1, i2, i3]`, targets `[i2, i3, i4]`. Each position predicts its next item.
The loss is a **full softmax over all 12k items** (cross-entropy), ignoring padded target positions
(`-100`). The original SASRec samples one negative and uses BCE. With only 12k items, a full softmax
is cheap and usually better.

Early stopping: evaluate on `valid` every `eval_every` epochs (full ranking, NDCG@10), keep the best
weights, and stop after `patience` evaluations without improvement.

## 3. Candidates and the selection effect

`topk_candidates` masks every item already in the history (scores → −inf), so we never recommend
something the user already bought, then takes the top 10. `build_candidate_examples` keeps an example
**only if the target is in its list**, and records `target_rank` (1-based), which is the paper's
`r^D` in Eq. 14.

Only ~6–9 % of examples survive, so the re-ranker works on the "retriever already roughly succeeded"
subset. That's why the paper's pre-rank R@9 is ~0.95: within the kept lists the target is almost always
in the top 9. The re-ranker's job is to move it from, say, position 4 to position 1.

**Why the m02 4-way split pays off now:** SASRec trained on `retriever_train` = `seq[:-3]`, so it has
never seen any of the three targets. The `rerank_train` lists are as hard as the `test` lists.

---

## Implementation order

| Step | File | Functions |
|---|---|---|
| 1 | `model.py` | `encode_histories`, `attention_mask`, `SASRec` |
| 2 | `train.py` | `make_training_batch`, `sasrec_loss` |
| 3 | `candidates.py` | `topk_candidates`, `build_candidate_examples` |
| 4 | `train.py` | `evaluate_full_ranking`, `train_sasrec` |
| 5 | `run.py` | the pipeline |

```powershell
python -m pytest gr2mini/m06_retriever -v
python -m gr2mini.m06_retriever.run --config configs/tiny.yaml
```

## Targets from my reference run

| | full Beauty | tiny |
|---|---|---|
| training time (GPU) | ~6–7 min (early stop ≈ epoch 45–60) | ~1 min |
| valid full-ranking R@10 / N@10 | ≈ 0.10 / 0.06 | much lower (tiny data) |
| kept examples train / valid / test | ≈ 2–2.3k each (~10 %) | a few dozen each |
| test pre-rank R@1 (kept lists) | ≈ 0.25–0.35 | noisy |

The test pre-rank `ranking_report` is your **"Pre-rank" row**, the baseline every later stage must beat.
