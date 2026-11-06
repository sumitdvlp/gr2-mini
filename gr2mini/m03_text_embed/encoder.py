"""Item text -> unit-norm embedding vectors, with a fingerprinted cache.  (paper Eq. 18)"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np


class TextEncoder(Protocol):
    """Anything with .encode(list[str]) -> array-like [B, D]. SentenceTransformer satisfies this."""

    def encode(self, texts: list[str], /) -> Any: ...


def load_sentence_encoder(model_name: str, device: str = "cuda") -> TextEncoder:
    """Return sentence_transformers.SentenceTransformer(model_name, device=device).

    Import sentence_transformers INSIDE this function, so importing this module stays fast and
    the unit tests don't need the library.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def text_fingerprint(texts: Sequence[str], model_name: str, normalize: bool) -> str:
    """SHA-256 hex digest identifying (model_name, normalize, texts).

    * Must change if ANY text changes, the order changes, the model name changes, or `normalize`
      changes. Hint: hashlib.sha256(); feed it model_name, str(normalize) and every text,
      with a separator that can't appear inside a text (e.g. "\\x00").
    """
    hasher = hashlib.sha256()
    
    # Update hasher with model_name and separator
    hasher.update(model_name.encode('utf-8'))
    hasher.update(b'\x00')
    
    # Update hasher with normalize state and separator
    hasher.update(str(normalize).encode('utf-8'))
    hasher.update(b'\x00')
    
    # Update hasher with each text in the sequence, separated by null bytes
    for text in texts:
        hasher.update(text.encode('utf-8'))
        hasher.update(b'\x00')
        
    return hasher.hexdigest()
    


def encode_texts(encoder: TextEncoder, texts: Sequence[str], batch_size: int = 256, normalize: bool = True) -> np.ndarray:
    """Encode texts in batches. Returns float32 array [len(texts), D].

    * Call encoder.encode(batch) with a python list of at most `batch_size` texts each time.
    * Convert each result with np.asarray(..., dtype=np.float32) and concatenate.
    * If normalize: divide each row by max(L2 norm, 1e-12)  (zero rows stay zero, never NaN).
    * The result must not depend on batch_size.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    encoded_chunks = []
    
    # Process texts in batches sequentially to preserve order and independence from batch_size
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        batch_embeddings = encoder.encode(batch)
        arr = np.asarray(batch_embeddings, dtype=np.float32)
        encoded_chunks.append(arr)

    # Concatenate all batches along the row axis
    embeddings = np.concatenate(encoded_chunks, axis=0)

    if normalize:
        # Calculate L2 norms along each row (axis=1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Prevent division by zero while ensuring zero rows remain zero
        denom = np.maximum(norms, 1e-12)
        embeddings = embeddings / denom

    return embeddings


def embed_items_cached(
    texts: Sequence[str],
    cache_path: str | Path,
    load_encoder: Callable[[], TextEncoder],
    model_name: str,
    batch_size: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """Return embeddings for `texts`, using/refreshing a cache.

    cache_path is the .npy file, e.g. artifacts/item_emb.npy. Its metadata lives next to it in
    the same name with suffix ".meta.json" (artifacts/item_emb.meta.json), containing at least
    {"fingerprint": ..., "shape": [N, D], "model_name": ...}.

    * Cache hit  (both files exist AND the fingerprint matches): np.load and return.
      `load_encoder` must NOT be called.
    * Cache miss: encoder = load_encoder(); emb = encode_texts(...); save both files
      (create parent dirs); return emb.
    """
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(".meta.json")

    # Compute current fingerprint to validate cache
    current_fingerprint = text_fingerprint(texts, model_name, normalize)

    # Check for cache hit
    cache_hit = False
    if cache_path.is_file() and meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") == current_fingerprint:
                cache_hit = True
        except (json.JSONDecodeError, OSError):
            cache_hit = False

    if cache_hit:
        return np.load(cache_path)

    # Cache miss: load encoder, compute embeddings, and save to cache
    encoder = load_encoder()
    emb = encode_texts(encoder, texts, batch_size=batch_size, normalize=normalize)

    # Ensure parent directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Save embeddings to .npy
    np.save(cache_path, emb)

    # Save metadata to .meta.json
    meta_data = {
        "fingerprint": current_fingerprint,
        "shape": list(emb.shape),
        "model_name": model_name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    return emb




def cosine_top_k(emb: np.ndarray, query_idx: int, k: int, exclude_self: bool = True) -> list[tuple[int, float]]:
    """The k rows of `emb` most cosine-similar to row `query_idx`, as [(index, similarity), ...]
    sorted by similarity descending.

    * Must be correct even if `emb` is NOT normalized (normalize inside; guard zero norms).
    * exclude_self: drop query_idx itself from the results.
    * If fewer than k candidates exist, return all of them.
    """
    if emb.size == 0 or k <= 0:
        return []

    # 1. Normalize rows of emb with zero-norm guarding
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    denom = np.maximum(norms, 1e-12)
    emb_norm = emb / denom

    # 2. Compute cosine similarities (dot product of normalized vectors)
    query_vec = emb_norm[query_idx]
    similarities = emb_norm @ query_vec

    # 3. Sort indices by similarity in descending order
    sorted_indices = np.argsort(similarities)[::-1]

    # 4. Collect top k results while respecting exclude_self
    results = []
    for idx in sorted_indices:
        idx_int = int(idx)
        if exclude_self and idx_int == query_idx:
            continue
        
        results.append((idx_int, float(similarities[idx_int])))
        if len(results) >= k:
            break

    return results
