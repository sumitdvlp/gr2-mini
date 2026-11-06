from collections import Counter

import pytest

from gr2mini.m02_data.sequences import (
    SeqExample,
    build_sequences,
    dataset_stats,
    k_core_filter,
    make_splits,
    remap_ids,
    subsample_users,
)


# ---------------------------------------------------------------- k-core
def test_k_core_readme_example_is_iterative():
    inter = [("u1", "i1", 1), ("u1", "i2", 2), ("u2", "i1", 1), ("u2", "i2", 2), ("u3", "i2", 3), ("u3", "i3", 4)]
    out = k_core_filter(inter, k=2)
    assert out == [("u1", "i1", 1), ("u1", "i2", 2), ("u2", "i1", 1), ("u2", "i2", 2)]


def test_k_core_result_satisfies_the_core_property():
    import random

    rnd = random.Random(0)
    inter = [(f"u{rnd.randint(0, 40)}", f"i{rnd.randint(0, 30)}", t) for t in range(400)]
    for k in (2, 3, 5):
        out = k_core_filter(inter, k)
        users, items = Counter(u for u, _, _ in out), Counter(i for _, i, _ in out)
        assert all(c >= k for c in users.values())
        assert all(c >= k for c in items.values())
        # order preserved: `out` is a subsequence of `inter`
        it = iter(inter)
        assert all(any(x == y for y in it) for x in out)


def test_k_core_can_remove_everything():
    assert k_core_filter([("u1", "i1", 1), ("u2", "i2", 1)], k=2) == []


def test_k_core_k1_is_identity_copy():
    inter = [("u1", "i1", 1)]
    out = k_core_filter(inter, 1)
    assert out == inter and out is not inter


# ---------------------------------------------------------------- subsample
def _many_users(n=50):
    return [(f"u{u:02d}", f"i{(u * 7 + j) % 13}", j) for u in range(n) for j in range(3)]


def test_subsample_users_count_order_and_determinism():
    inter = _many_users()
    a = subsample_users(inter, 10, seed=1)
    b = subsample_users(inter, 10, seed=1)
    c = subsample_users(inter, 10, seed=2)
    assert a == b
    assert len({u for u, _, _ in a}) == 10
    assert {u for u, _, _ in a} != {u for u, _, _ in c}
    it = iter(inter)
    assert all(any(x == y for y in it) for x in a)  # original order kept
    # each kept user keeps ALL of their interactions
    assert all(n == 3 for n in Counter(u for u, _, _ in a).values())


def test_subsample_users_none_or_large_is_noop():
    inter = _many_users(5)
    assert subsample_users(inter, None, 0) == inter
    assert subsample_users(inter, 100, 0) == inter


# ---------------------------------------------------------------- sequences & ids
def test_build_sequences_sorts_by_time_and_is_stable_on_ties():
    inter = [("u1", "c", 30), ("u2", "x", 5), ("u1", "a", 10), ("u1", "b2", 20), ("u1", "b1", 20)]
    seqs = build_sequences(inter)
    assert seqs == {"u1": ["a", "b2", "b1", "c"], "u2": ["x"]}


def test_remap_ids_is_sorted_and_consistent():
    seqs = {"uB": ["z", "a"], "uA": ["m", "z"]}
    int_seqs, user_ids, asins = remap_ids(seqs)
    assert user_ids == ["uA", "uB"]
    assert asins == ["a", "m", "z"]
    assert int_seqs == [[1, 2], [2, 0]]


# ---------------------------------------------------------------- splits
def test_make_splits_matches_readme_diagram():
    s = make_splits([10, 11, 12, 13, 14, 15])
    assert s.retriever_train == [10, 11, 12]
    assert s.rerank_train == SeqExample([10, 11, 12], 13)
    assert s.valid == SeqExample([10, 11, 12, 13], 14)
    assert s.test == SeqExample([10, 11, 12, 13, 14], 15)


def test_make_splits_minimum_length():
    s = make_splits([1, 2, 3, 4])
    assert s.retriever_train == [1] and s.test.target == 4
    with pytest.raises(ValueError):
        make_splits([1, 2, 3])


def test_make_splits_returns_fresh_lists():
    seq = [1, 2, 3, 4, 5]
    s = make_splits(seq)
    s.retriever_train.append(99)
    s.test.history.append(99)
    assert seq == [1, 2, 3, 4, 5]
    assert s.valid.history == [1, 2, 3]
    assert 99 not in s.rerank_train.history


def test_make_splits_never_leaks_targets_into_retriever_train():
    seq = list(range(100, 112))
    s = make_splits(seq)
    for ex in (s.rerank_train, s.valid, s.test):
        assert ex.target not in s.retriever_train


# ---------------------------------------------------------------- stats
def test_dataset_stats():
    st = dataset_stats([[0, 1, 2], [1, 2, 3, 4, 0]], num_items=5)
    assert st["users"] == 2 and st["items"] == 5 and st["interactions"] == 8
    assert st["avg_seq_len"] == pytest.approx(4.0)
    assert st["density"] == pytest.approx(8 / 10)
