# m01 — Metrics: Recall@K, NDCG@K, SID uniqueness

**Paper:** Appendix G (Eqs. 30–35). Used in Tables 2–6.
**Time:** ~1 hour. **File you implement:** `metrics.py` (pure Python, no torch needed).

We build the measuring tools before anything else, because every later module reports them:
SID uniqueness in m05, retriever quality in m06, re-ranking quality in m13–m15.

---

## 1. The evaluation setting: leave-one-out, one relevant item

For each user we hide the **one** next item `i*` and ask the model for a ranked list `R_u`.
There is exactly one relevant item per user, which makes the formulas simple.

### Recall@K (Eq. 30), a.k.a. Hit Rate@K

```
Recall@K = mean over users of  1[ i* is among the first K items of R_u ]
```

### NDCG@K (Eqs. 31–32)

DCG rewards hits near the top more: a hit at (1-based) rank `j` is worth `1 / log2(j + 1)`.

```
rank:            1      2      3      4      5     ...
gain 1/log2(j+1): 1.000  0.631  0.500  0.431  0.387
```

The ideal DCG with one relevant item is 1 (it's at rank 1), so **NDCG@K = 1/log2(rank+1) if rank ≤ K
else 0**, averaged over users. (The general formula uses `2^rel − 1` with `rel ∈ {0,1}`, which is the same thing.)

**Worked example:**

| user | ranked list | target | rank | R@1 | R@3 | N@3 |
|---|---|---|---|---|---|---|
| u1 | [5, 3, 9] | 3 | 2 | 0 | 1 | 0.631 |
| u2 | [1, 2, 3] | 4 | — | 0 | 0 | 0 |
| u3 | [7, 8, 9] | 7 | 1 | 1 | 1 | 1.000 |
| **mean** | | | | **0.333** | **0.667** | **0.544** |

### Reading the paper's re-ranking tables

In Tables 5–6 the candidate list has K = 10 and **the target is always in it** (that's how the
examples are filtered). So Recall@10 is trivially 1, which is why the paper reports R@1, R@5, R@9, N@5,
and N@10. **R@1 is the headline number:** did the re-ranker put the right item on top?

---

## 2. SID collisions (Eqs. 33–35)

Two items *collide* if they get exactly the same Semantic ID. The LLM can then never tell them
apart, so we want ≈ 0 collisions.

```
C              = { items i  |  there exists another item j ≠ i with SID(i) == SID(j) }   (Eq. 33)
CollisionRate  = |C| / |I|                                                                (Eq. 34)
Uniqueness     = 1 − CollisionRate                                                        (Eq. 35)
```

⚠️ **Common mistake:** `1 − (#distinct SIDs / #items)` is a *different* number. Example: SIDs
`[(1,2), (1,2), (3,4), (5,6)]`:

* paper's definition: items 0 and 1 collide → |C| = 2 → CollisionRate = 2/4 = **0.5**
* the "distinct" formula: 3 distinct / 4 items → 1 − 0.75 = **0.25**

The paper's version counts *every* item involved in a collision. Use the paper's.

Hint: `collections.Counter` over `tuple(sid)`.

---

## Implementation order

`rank_of` → `recall_at_k` → `ndcg_at_k` → `ranking_report` → `collision_rate` → `uniqueness`.

## Test

```powershell
python -m pytest gr2mini/m01_metrics -v
```

(No live example. You'll see these numbers printed by every later `run.py`.)
