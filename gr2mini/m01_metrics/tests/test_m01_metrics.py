import math

import numpy as np
import pytest
import torch

from gr2mini.m01_metrics.metrics import (
    collision_rate,
    ndcg_at_k,
    rank_of,
    ranking_report,
    recall_at_k,
    uniqueness,
)

# The worked example from the README.
LISTS = [[5, 3, 9], [1, 2, 3], [7, 8, 9]]
TARGETS = [3, 4, 7]


def test_rank_of():
    assert rank_of(3, [5, 3, 9]) == 2
    assert rank_of(5, [5, 3, 9]) == 1
    assert rank_of(4, [1, 2, 3]) is None
    assert rank_of(1, []) is None


def test_recall_worked_example():
    assert recall_at_k(LISTS, TARGETS, 1) == pytest.approx(1 / 3)
    assert recall_at_k(LISTS, TARGETS, 2) == pytest.approx(2 / 3)
    assert recall_at_k(LISTS, TARGETS, 3) == pytest.approx(2 / 3)


def test_recall_k_larger_than_list_is_fine():
    assert recall_at_k(LISTS, TARGETS, 100) == pytest.approx(2 / 3)


def test_ndcg_worked_example():
    expected = (1 / math.log2(3) + 0 + 1) / 3
    assert ndcg_at_k(LISTS, TARGETS, 3) == pytest.approx(expected)
    assert ndcg_at_k(LISTS, TARGETS, 1) == pytest.approx(1 / 3)  # only u3's hit at rank 1 counts


def test_ndcg_hit_at_rank_k_counts_but_after_k_does_not():
    assert ndcg_at_k([[0, 1, 2, 3, 4]], [4], 5) == pytest.approx(1 / math.log2(6))
    assert ndcg_at_k([[0, 1, 2, 3, 4]], [4], 4) == 0.0


def test_ndcg_is_at_most_recall():
    rng = np.random.default_rng(0)
    lists = [list(rng.permutation(10)) for _ in range(50)]
    targets = [int(rng.integers(0, 10)) for _ in range(50)]
    for k in (1, 3, 5, 10):
        assert ndcg_at_k(lists, targets, k) <= recall_at_k(lists, targets, k) + 1e-12


@pytest.mark.parametrize("fn", [recall_at_k, ndcg_at_k])
def test_validation(fn):
    with pytest.raises(ValueError):
        fn([[1, 2]], [1, 2], 1)  # length mismatch
    with pytest.raises(ValueError):
        fn([], [], 1)  # empty
    with pytest.raises(ValueError):
        fn([[1]], [1], 0)  # k < 1


def test_ranking_report_keys_and_values():
    rep = ranking_report(LISTS, TARGETS, recall_ks=(1, 3), ndcg_ks=(3,))
    assert list(rep) == ["R@1", "R@3", "N@3"]
    assert rep["R@1"] == pytest.approx(1 / 3)
    assert rep["N@3"] == pytest.approx(ndcg_at_k(LISTS, TARGETS, 3))


def test_ranking_report_default_keys_match_paper_tables():
    rep = ranking_report([list(range(10))], [0])
    assert list(rep) == ["R@1", "R@5", "R@9", "N@5", "N@10"]
    assert all(v == pytest.approx(1.0) for v in rep.values())


def test_collision_rate_paper_definition_not_distinct_ratio():
    sids = [(1, 2), (1, 2), (3, 4), (5, 6)]
    assert collision_rate(sids) == pytest.approx(0.5)  # NOT 0.25
    assert uniqueness(sids) == pytest.approx(0.5)


def test_collision_rate_three_way_collision():
    sids = [(0, 0), (0, 0), (0, 0), (1, 0)]
    assert collision_rate(sids) == pytest.approx(0.75)


def test_collision_rate_all_unique_and_all_same():
    assert collision_rate([(i, 0) for i in range(10)]) == 0.0
    assert uniqueness([(i, 0) for i in range(10)]) == 1.0
    assert collision_rate([(7, 7, 7)] * 5) == 1.0


def test_collision_rate_accepts_tensors_lists_and_arrays():
    t = torch.tensor([[1, 2], [1, 2], [3, 4]])
    assert collision_rate(t) == pytest.approx(2 / 3)
    assert collision_rate(t.numpy()) == pytest.approx(2 / 3)
    assert collision_rate([[1, 2], [1, 2], [3, 4]]) == pytest.approx(2 / 3)


def test_collision_rate_prefixes_are_not_collisions():
    # (1,2,3) and (1,2,4) share a prefix but are different SIDs
    assert collision_rate([(1, 2, 3), (1, 2, 4)]) == 0.0


def test_collision_rate_empty_raises():
    with pytest.raises(ValueError):
        collision_rate([])
