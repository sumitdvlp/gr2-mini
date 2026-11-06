# m03 — Text embeddings: item text → vector h

**Paper:** Appendix A, Eq. 18: `h = f_enc(x), h ∈ R^d`. The paper doesn't name its text encoder.
**Time:** ~1 hour. **Files you implement:** `encoder.py`, `run.py`.

## Where this fits

Semantic IDs should give **similar items similar IDs**. "Similar" has to come from somewhere, and
for GR2 it's the item's *text*. A pretrained sentence encoder maps each item's text line from m02
to a vector, and the RQ-VAE (m04) turns those vectors into discrete codes.

```
"Title: Moisturizing Shampoo | Brand: X | Category: Beauty > Hair Care > Shampoos"
                    │  frozen sentence encoder (all-MiniLM-L6-v2, 384-d)
                    ▼
              h ∈ R^384, ‖h‖ = 1        ── m04 ──►  (12, 3, 250, 7)
```

We use `sentence-transformers/all-MiniLM-L6-v2`: 22 M parameters, 384 dimensions, and ~20 s for
all 12k items on your GPU, including the model download and load the first time. TIGER, the original SID paper, used Sentence-T5 (768-d). Any decent encoder
works, since the RQ-VAE learns its own latent space on top.

## Concepts

### Why L2-normalize?
Sentence encoders are trained with cosine similarity. With unit vectors:
* cosine similarity = dot product (cheap, one matmul)
* squared distance = 2 − 2·cos, so k-means and nearest-code search in m04 behave like cosine search.

**Pitfall:** an all-zero vector has norm 0, and dividing by it gives NaN. Clamp the norm: `v / max(‖v‖, 1e-12)`.

### Caching with a fingerprint
Embedding is deterministic, so compute it once and save it. But a cache that silently survives a
change in the data or the model will poison every later module. We store a **fingerprint** (a SHA-256
hash of model name + normalize flag + all texts) next to the `.npy`, and recompute on mismatch.

```
artifacts/item_emb.npy         float32 [N_items, 384]
artifacts/item_emb.meta.json   {"fingerprint": "ab12...", "shape": [12101, 384], "model_name": "..."}
```

### Lazy loading
Loading the encoder takes a few seconds and ~100 MB of VRAM. `embed_items_cached` takes a
*factory* (`load_encoder: Callable[[], encoder]`) and only calls it on a cache miss. The tests check this.

### Duck typing for testability
`encode_texts` only needs an object with `.encode(list_of_texts) -> array [B, D]`. A real
`SentenceTransformer` has that method, and so does the tiny fake encoder in the tests. That's why
these tests run in milliseconds without downloading anything.

## Implementation order

`text_fingerprint` → `encode_texts` → `embed_items_cached` → `cosine_top_k` → `load_sentence_encoder` → `run.py`.

## Test & try

```powershell
python -m pytest gr2mini/m03_text_embed -v
python -m pytest gr2mini/m03_text_embed -m integration -v   # downloads MiniLM, checks real semantics
python -m gr2mini.m03_text_embed.run --config configs/tiny.yaml
```

The live example prints the nearest neighbours of a few random items, something like:

```
query: Title: Moisturizing Shampoo for Dry Hair | Brand: ... | Category: Beauty > Hair Care > Shampoos
  0.83  Title: Hydrating Conditioner ... | Category: Beauty > Hair Care > Conditioners
  0.80  Title: Argan Oil Shampoo ...
```

If the neighbours look related, the RQ-VAE has something meaningful to quantize.
