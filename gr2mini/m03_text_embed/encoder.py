"""Item text -> unit-norm embedding vectors, with a fingerprinted cache.  (paper Eq. 18)"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np


class TextEncoder(Protocol):
    """Anything with .encode(list[str]) -> array-like [B, D]. SentenceTransformer satisfies this."""

    def encode(self, texts: list[str]) -> Any: ...


def load_sentence_encoder(model_name: str, device: str = "cuda") -> TextEncoder:
    """Return sentence_transformers.SentenceTransformer(model_name, device=device).

    Import sentence_transformers INSIDE this function, so importing this module stays fast and
    the unit tests don't need the library.
    """
    raise NotImplementedError("TODO(m03): implement load_sentence_encoder")


def text_fingerprint(texts: Sequence[str], model_name: str, normalize: bool) -> str:
    """SHA-256 hex digest identifying (model_name, normalize, texts).

    Must change if ANY text changes, the order changes, the model name changes, or `normalize`
    changes. Hint: hashlib.sha256(); feed it model_name, str(normalize) and every text,
    with a separator that can't appear inside a text (e.g. "\\x00").
    """
    raise NotImplementedError("TODO(m03): implement text_fingerprint")


def encode_texts(encoder: TextEncoder, texts: Sequence[str], batch_size: int = 256, normalize: bool = True) -> np.ndarray:
    """Encode texts in batches. Returns float32 array [len(texts), D].

    * Call encoder.encode(batch) with a python list of at most `batch_size` texts each time.
    * Convert each result with np.asarray(..., dtype=np.float32) and concatenate.
    * If normalize: divide each row by max(L2 norm, 1e-12)  (zero rows stay zero, never NaN).
    * The result must not depend on batch_size.
    """
    raise NotImplementedError("TODO(m03): implement encode_texts")


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
    raise NotImplementedError("TODO(m03): implement embed_items_cached")


def cosine_top_k(emb: np.ndarray, query_idx: int, k: int, exclude_self: bool = True) -> list[tuple[int, float]]:
    """The k rows of `emb` most cosine-similar to row `query_idx`, as [(index, similarity), ...]
    sorted by similarity descending.

    * Must be correct even if `emb` is NOT normalized (normalize inside; guard zero norms).
    * exclude_self: drop query_idx itself from the results.
    * If fewer than k candidates exist, return all of them.
    """
    raise NotImplementedError("TODO(m03): implement cosine_top_k")
