import json

import numpy as np
import pytest

from gr2mini.m03_text_embed.encoder import (
    cosine_top_k,
    embed_items_cached,
    encode_texts,
    text_fingerprint,
)


class FakeEncoder:
    """Deterministic toy 'sentence encoder': character histogram into 8 buckets.

    Returns float64 lists on purpose: your code must convert to float32.
    """

    dim = 8

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def encode(self, texts):
        assert isinstance(texts, list), "encode() should receive a python list"
        self.calls += 1
        self.batch_sizes.append(len(texts))
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for j, ch in enumerate(t):
                v[(ord(ch) + j) % self.dim] += 1.0
            out.append(v)
        return out


TEXTS = ["shampoo for dry hair", "conditioner", "red lipstick", "", "nail polish remover", "eye cream"]


# ---------------------------------------------------------------- fingerprint
def test_fingerprint_changes_with_any_input():
    base = text_fingerprint(TEXTS, "m", True)
    assert isinstance(base, str) and len(base) == 64
    assert text_fingerprint(list(TEXTS), "m", True) == base
    assert text_fingerprint(TEXTS[:-1] + ["eye creams"], "m", True) != base
    assert text_fingerprint(list(reversed(TEXTS)), "m", True) != base
    assert text_fingerprint(TEXTS, "other-model", True) != base
    assert text_fingerprint(TEXTS, "m", False) != base
    # separator matters: ["ab", "c"] vs ["a", "bc"]
    assert text_fingerprint(["ab", "c"], "m", True) != text_fingerprint(["a", "bc"], "m", True)


# ---------------------------------------------------------------- encode_texts
def test_encode_texts_shape_dtype_and_unit_norm():
    emb = encode_texts(FakeEncoder(), TEXTS, batch_size=4, normalize=True)
    assert emb.shape == (len(TEXTS), FakeEncoder.dim)
    assert emb.dtype == np.float32
    norms = np.linalg.norm(emb, axis=1)
    nonzero = [i for i, t in enumerate(TEXTS) if t]
    np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-6)


def test_encode_texts_zero_vector_is_not_nan():
    emb = encode_texts(FakeEncoder(), TEXTS, normalize=True)
    assert np.isfinite(emb).all()
    np.testing.assert_array_equal(emb[TEXTS.index("")], 0.0)


def test_encode_texts_respects_batch_size_and_is_batch_invariant():
    enc = FakeEncoder()
    a = encode_texts(enc, TEXTS, batch_size=4)
    assert enc.batch_sizes == [4, 2]
    b = encode_texts(FakeEncoder(), TEXTS, batch_size=1)
    np.testing.assert_allclose(a, b, atol=1e-7)


def test_encode_texts_without_normalize_returns_raw_values():
    emb = encode_texts(FakeEncoder(), ["aaaa"], normalize=False)
    assert emb.sum() == pytest.approx(4.0)


# ---------------------------------------------------------------- caching
def test_embed_items_cached_miss_then_hit(tmp_path):
    cache = tmp_path / "art" / "item_emb.npy"
    made = []

    def factory():
        enc = FakeEncoder()
        made.append(enc)
        return enc

    e1 = embed_items_cached(TEXTS, cache, factory, model_name="fake", batch_size=2)
    assert len(made) == 1 and cache.exists()
    meta_path = tmp_path / "art" / "item_emb.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["fingerprint"] == text_fingerprint(TEXTS, "fake", True)
    assert meta["shape"] == [len(TEXTS), FakeEncoder.dim]

    e2 = embed_items_cached(TEXTS, cache, factory, model_name="fake", batch_size=2)
    assert len(made) == 1, "cache hit must not load the encoder"
    np.testing.assert_array_equal(e1, e2)


def test_embed_items_cached_recomputes_when_texts_change(tmp_path):
    cache = tmp_path / "item_emb.npy"
    calls = []
    factory = lambda: calls.append(1) or FakeEncoder()  # noqa: E731
    embed_items_cached(TEXTS, cache, factory, model_name="fake")
    new_texts = TEXTS + ["brand new item"]
    e = embed_items_cached(new_texts, cache, factory, model_name="fake")
    assert len(calls) == 2
    assert e.shape[0] == len(new_texts)


# ---------------------------------------------------------------- neighbours
def test_cosine_top_k_order_scores_and_self_exclusion():
    emb = np.array([[1, 0], [10, 1], [0, 3], [-1, 0], [5, 5]], dtype=np.float32)  # NOT normalized
    res = cosine_top_k(emb, 0, k=3)
    assert [i for i, _ in res] == [1, 4, 2]
    assert res[0][1] == pytest.approx(10 / np.sqrt(101), rel=1e-5)
    assert res[1][1] == pytest.approx(1 / np.sqrt(2), rel=1e-5)
    assert all(i != 0 for i, _ in res)


def test_cosine_top_k_include_self_and_small_k():
    emb = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    res = cosine_top_k(emb, 0, k=1, exclude_self=False)
    assert res[0][0] == 0 and res[0][1] == pytest.approx(1.0)
    assert len(cosine_top_k(emb, 0, k=10)) == 2


# ---------------------------------------------------------------- the real thing
@pytest.mark.integration
def test_real_minilm_semantics():
    from gr2mini.m03_text_embed.encoder import load_sentence_encoder

    enc = load_sentence_encoder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    texts = ["Title: Moisturizing shampoo for dry hair | Category: Beauty > Hair Care > Shampoos",
             "Title: Hydrating conditioner for dry hair | Category: Beauty > Hair Care > Conditioners",
             "Title: Matte red lipstick | Category: Beauty > Makeup > Lips"]
    emb = encode_texts(enc, texts)
    assert emb.shape == (3, 384)
    assert emb[0] @ emb[1] > emb[0] @ emb[2]
