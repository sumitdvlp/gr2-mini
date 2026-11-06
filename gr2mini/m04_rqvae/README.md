# m04 — RQ-VAE: turning item embeddings into discrete codes

**Paper:** §2.1 (Eq. 1), §2.2 (Eqs. 2–3), §2.3 (Eqs. 4–9), Appendix A (Eqs. 18–28, Algorithm 1).
Background: TIGER (Rajput et al., 2023), RQ-VAE (Lee et al., 2022), VQ-VAE (van den Oord et al., 2017).
**Time:** ~1–2 days, the most math-heavy module so far. **Files you implement:** `kmeans.py`,
`quantizer.py`, `losses.py`, `contrastive.py`, `model.py`, `train.py`, `run.py`.

---

## 0. The goal

An LLM can't read a 384-d float vector, but it can read tokens. We want each item to become a short
tuple of discrete codes, its **Semantic ID (SID)**:

```
"Moisturizing Shampoo ..."  ──►  (12, 3, 250, 7)  ──►  <s_a_12><s_b_3><s_c_250><s_d_7>
```

with two properties:
1. **Semantic:** similar items share *prefixes*. All shampoos might start with `12`, and moisturizing
   shampoos with `12, 3`. The LLM can then generalize from items it has seen to items it hasn't.
2. **Unique:** no two items get the same full SID (m05 handles the last bit of this).

## 1. Residual quantization: coarse to fine (Eqs. 19–23)

Vector quantization (VQ) replaces a vector with its nearest **codeword** from a learned codebook.
One codebook of 256 entries can only name 256 things, which is far fewer than 12k items. *Residual*
quantization repeats VQ on what's left over:

```
r_1 = z                                   (z = latent of the item)          (19)
for k = 1..L:
    c_k = argmin_j ‖ r_k − e_j^(k) ‖²     (nearest codeword in codebook k)  (20)
    q_k = e^(k)_{c_k}                                                         (21)
    r_{k+1} = r_k − q_k                   (what level k couldn't explain)    (22)
ẑ = q_1 + q_2 + ... + q_L                 (the reconstruction)              (23)
```

Picture it in 2-D: level 1 says "you're in the north-east region", level 2 says "...the lower-left part
of it", and level 3 refines further. With L = 4 levels of C = 256 codes, there are 256⁴ ≈ 4·10⁹ possible SIDs.

**Why the prefixes are semantic:** level 1 captures the largest directions of variation, which
correspond to coarse categories. Items that are close in latent space pick the same early codes.

### The straight-through estimator
`argmin` has no gradient. To train the encoder anyway, we pretend quantization is the identity in
the backward pass:

```python
z_q = z + (z_hat - z).detach()     # forward value: z_hat.   d z_q / d z = I
```

The decoder sees `z_hat`, and gradients pass straight to the encoder.

### Detaching the residual chain
Compute `r_{k+1} = r_k − q_k.detach()`. Each codebook should then be trained by its *own* level's loss
only, not through later levels. Most RQ implementations do this.

## 2. The model: encoder → quantizer → decoder

The paper's Appendix A compares `ĥ = Σ q_k` with the text embedding `h` directly. We use the standard
TIGER-style RQ-VAE, which wraps the quantizer in small MLPs (see the top-level README §8):

```
x (384, unit-norm text embedding)
  └─ encoder MLP 384 → 256 → 128 → 32 ──► z (32-d latent)
        └─ ResidualQuantizer (L=4, C=256) ──► z_q (straight-through), codes (B, 4)
              └─ decoder MLP 32 → 128 → 256 → 384 ──► x̂
```

The quantizer works in a small 32-d latent space, which makes nearest-neighbour search cheap and
k-means well-behaved.

## 3. The loss (Eqs. 25–28) and its weights

```
L = L_rec  +  w_cb · Σ_k L_cb^(k)  +  w_com · Σ_k L_com^(k)  [+ λ_div · L_div]  [+ λ_ctr · L_ctr]

L_rec      = ‖x − x̂‖²                       reconstruction                  (26)
L_cb^(k)   = ‖sg[r_k] − q_k‖²               move codewords to the data      (27)
L_com^(k)  = ‖r_k − sg[q_k]‖²               keep the encoder near codewords (28)
```

`sg[·]` = stop-gradient = `.detach()`. Each squared norm is **summed over dimensions and averaged over
the batch** (consistent everywhere, and the tests check it). Defaults: `w_com = 0.25` (as in VQ-VAE),
`w_cb = 1.0`. **With EMA, L_cb isn't used**, because EMA replaces it (next section).

## 4. Codebook collapse and the paper's five fixes (§2.3)

**Collapse** = only a handful of codewords get used. Every item then maps to the same few SIDs, and
uniqueness is terrible. The paper's Table 2 shows uniqueness ranges from **3 % to 99.98 %** depending
on which fixes are on. These are the fixes, ordered by importance:

### 4.1 EMA codebook updates (biggest effect: avg. uniqueness 31 % → 97 %)
Instead of gradient descent on `L_cb`, each codeword becomes a running average of the residuals
assigned to it:

```
n_j = #residuals assigned to code j in this batch          s_j = Σ of those residuals
N_j ← γ·N_j + (1−γ)·n_j                                     (EMA cluster size)
M_j ← γ·M_j + (1−γ)·s_j                                     (EMA sum)
Ñ_j = (N_j + ε) / (Σ_i N_i + C·ε) · Σ_i N_i                (Laplace smoothing: no divide-by-zero)
e_j = M_j / Ñ_j
```

with decay γ ∈ [0.95, 0.99]. This is k-means running *online*. It's stable because it doesn't depend on
learning rates or gradient magnitudes. Initialize `N_j = 1` and `M_j = e_j`, so an unused code
**stays where it is** instead of collapsing to 0.

⚠️ The paper's Eq. 4–5 write EMA as "smooth the optimizer-updated weights". We implement the standard
VQ-VAE version above, which is what the method the paper cites does.

### 4.2 RQ-K-means initialization
Random codewords start far from the data. Many are never the nearest to anything, so they're dead
from step 1. Instead, **initialize level k's codebook by running k-means on the level-k residuals of
the whole dataset** (k-means++ seeding, then Lloyd iterations). Level 1 on z, level 2 on z − q_1, and so on.

k-means++ seeding: pick the first centre uniformly at random, then pick each next centre with
probability ∝ D(x)², the squared distance to the nearest centre chosen so far. This spreads the
centres out. Edge case: if all D(x)² are 0 (fewer distinct points than k), fall back to uniform sampling.

### 4.3 Dead-code reset (Eqs. 8–9)
Track for each code how many **consecutive batches** it went unused. When the count reaches τ, move
the code onto **the residual that the codebook currently explains worst**, i.e. the largest
`‖r_i − e_{c_i}‖`. That puts the code where it's needed. With several dead codes in one batch, use the
top-n worst residuals (distinct ones), not the same point n times. Reset that code's EMA stats as well
(`N_j = 1`, `M_j = new vector`).

### 4.4 Diversity loss (Eqs. 6–7)
Soft-assign each residual to codes, `p(j | r) = softmax(−‖r − e_j‖² / T)`, then average over the batch
to get the code usage distribution `p_{k,j}`:

```
L_div = Σ_k C_k · Σ_j p_{k,j}²
```

`Σ_j p_j²` is smallest (= 1/C) when usage is uniform and largest (= 1) when everything uses one code.
Multiplying by C_k makes each level's term equal to **1 at uniform** and **C_k at full collapse**.
The gradient reaches the *encoder* through the soft assignments. The paper finds it helps only slightly
(+0.6 %) and its best config turns it off (`diversity_weight: 0.0`), but implement it anyway.

### 4.5 Random last level (Eq. 10), in m05
At SID assignment time (not training), the last M levels are drawn uniformly at random. We'll see in
m05 why that nearly eliminates collisions.

## 5. Collaborative signal: the co-engagement contrastive loss (Eqs. 2–3)

Text alone says "shampoo ≈ shampoo". Purchases also say "shampoo ≈ conditioner", because people buy
them together. The paper adds an InfoNCE loss that pulls **co-engaged** items together:

```
s(i, j) = (h_i/‖h_i‖) · (h_j/‖h_j‖) / T                                         (3)
L_ctr(i) = − log  exp(s(i, p)) / ( exp(s(i, p)) + Σ_{n ∈ N(i)} exp(s(i, n)) )   (2)
```

Our choices (the paper doesn't pin these down):
* **representation:** the RQ-VAE latent `z` (the text encoder is frozen, so its output can't move)
* **positive p:** a random item that appeared within `coengagement_window` positions of item i in some
  user's **retriever_train** sequence. Never use valid/test positions: that would be leakage!
* **negatives N(i):** the positives of the *other* anchors in the batch ("in-batch negatives").
  Then Eq. 2 is exactly cross-entropy over a [B, B] similarity matrix with the diagonal as labels:

```python
logits = normalize(anchor) @ normalize(positive).T / T     # [B, B]
loss   = F.cross_entropy(logits, arange(B))                # over valid rows only
```

Items with no co-engaged neighbour get `valid = False` and are left out of the loss.

**Trade-off in the paper (Tables 2–3):** the contrastive loss *lowers* uniqueness (−13 %, since similar
items get pulled into the same codes) but *improves* the downstream recommender. Their best tokenizer
uses it.

## 6. Training loop (`train.py`)

```
1. x = all item embeddings on device
2. init: quantizer.init_codebooks(encoder(x))          ← RQ-K-means on the INITIAL latents
3. for epoch:  for batch of item ids (random permutation):
       pos, valid = sample_positives(ids, neighbors)   (if contrastive_weight > 0)
       loss, terms, out = rqvae_loss(model, x[ids], x[pos], valid, ...)
       opt.zero_grad(); loss.backward(); opt.step()
       quantizer.update_codebooks(out.rq)              ← EMA + dead-code reset, AFTER the step
   every eval_every epochs: codes for all items → uniqueness, #used codes per level
```

**Why is `update_codebooks` a separate call instead of happening inside `forward`?** If forward
modified the codebook tensors in place, autograd would complain at `backward()`: a tensor saved for
the backward pass was modified ("one of the variables needed for gradient computation has been modified
by an inplace operation"). Keeping `forward` pure also makes it testable.

## 7. What to expect: targets from my reference runs

Full Beauty (12,101 items, 300 epochs) takes **~2–3 min** on your GTX 1660 Ti (`tiny`: ~1 min).
I ran a reference implementation, and these are the numbers to aim for. Yours will differ a little
(seeds, small implementation choices) but should be in the same ballpark:

| config (`--set ...`) | uniqueness, deterministic 4-level codes | level-1 codes used | note |
|---|---|---|---|
| default (EMA + dead-code reset + contrastive) | **0.968** | 256 / 256 | |
| `rqvae.use_ema=false` | 0.964 | 256 / 256 | dead-code reset rescues it |
| `rqvae.dead_code_threshold=null` | 0.966 | **69 / 256** | ⚠️ level-1 collapse |
| `rqvae.use_ema=false rqvae.dead_code_threshold=null` | 0.972 | 162 / 256 | |
| `rqvae.contrastive_weight=0.0` | 0.950 | 256 / 256 | trains ~35 % faster |

(`tiny` default: ≈ 0.96, with all 64 level-1 codes used.) With M = 1 random last level (m05), every
config reaches ≥ 99.9 %.

### Things to notice (and think about)

1. **A single uniqueness number hides collapse.** Without dead-code reset, level 1 uses only 69 codes,
   so ~73 % of the coarsest "semantic categories" are wasted. Uniqueness still looks fine because levels
   2–4 compensate. **Always look at per-level usage.**
2. **Why does level 1 collapse in epoch 1?** Look at `used_codes` at epoch 1 with
   `dead_code_threshold=null`. Only a handful of codes are used, and uniqueness is ~3 %. The codebooks
   were fit by k-means to the latents of the *untrained* encoder, then the encoder moved a lot in its
   first epoch, and most codewords ended up nearest to nothing. Dead-code reset fixes exactly this.
   *Bonus experiment:* re-run `init_codebooks` after one warm-up epoch instead.
3. **We don't reproduce the paper's Table 2 gaps.** The paper reports that EMA raises average uniqueness
   from 31 % to 97 %, and that the contrastive loss *lowers* it by 13 %. In our setup the differences are
   small, and contrastive slightly *raises* it. The paper doesn't give its learning rate, latent size,
   loss weights or number of levels used for the metric, and any of these can change the picture. This
   is normal when reproducing papers. Write down your numbers. If you want, try to find a setting
   where no-EMA collapses (higher `lr`? smaller `latent_dim`?).
4. Try `--set rqvae.diversity_weight=0.1` too. The paper found +0.6 %. What do you get?

---

## Implementation order (each step has its own tests)

| Step | File | Functions | Tests |
|---|---|---|---|
| 1 | `kmeans.py` | `pairwise_sq_dists`, `kmeans_plusplus_init`, `kmeans` | `test_m04_kmeans.py` |
| 2 | `quantizer.py` | `ResidualQuantizer.__init__`, `forward`, `encode` | `test_m04_quantizer.py` (forward tests) |
| 3 | `quantizer.py` | `init_codebooks`, `ema_update`, `reset_dead_codes`, `update_codebooks` | `test_m04_quantizer.py` (state tests) |
| 4 | `losses.py` | `reconstruction_loss`, `soft_assignments`, `diversity_loss` | `test_m04_losses.py` |
| 5 | `contrastive.py` | `info_nce`, `build_coengagement_neighbors`, `sample_positives` | `test_m04_losses.py` |
| 6 | `model.py` | `mlp`, `RQVAE`, `save_rqvae`, `load_rqvae` | `test_m04_model_train.py` |
| 7 | `train.py` | `rqvae_loss`, `train_rqvae` | `test_m04_model_train.py` |
| 8 | `run.py` | the live example | (run it) |

```powershell
python -m pytest gr2mini/m04_rqvae -v
python -m pytest gr2mini/m04_rqvae/tests/test_m04_kmeans.py -v      # one step at a time
python -m gr2mini.m04_rqvae.run --config configs/tiny.yaml
```

### Hints
* Pairwise squared distances without a [B, C, D] tensor: `‖a‖² − 2 a·bᵀ + ‖b‖²`, then `.clamp_min(0)`
  (float error can make it slightly negative).
* `torch.Tensor.index_add_` / `torch.bincount(codes, minlength=C)` compute per-code sums and counts.
* Register non-trainable state with `self.register_buffer(...)` so it moves with `.to(device)` and is
  saved in `state_dict()`, but isn't returned by `.parameters()`.
* `torch.randperm(n, generator=g)` and `torch.multinomial(p, 1, generator=g)` take generators.
