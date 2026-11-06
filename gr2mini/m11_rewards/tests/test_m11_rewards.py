import json

import pytest

from gr2mini.m11_rewards.rewards import RewardInfo, compute_reward, rank_reward

CANDS = list(range(100, 110))  # 10 candidates


def ex(target_rank):
    return {"user": 0, "history": [1, 2], "candidates": CANDS, "target": CANDS[target_rank - 1], "target_rank": target_rank}


def answer(order_1based):
    return json.dumps({"explanation": "because", "recommendations": [str(k) for k in order_1based]})


def test_rank_reward():
    assert rank_reward(4, 1, 10) == pytest.approx(0.3)
    assert rank_reward(4, 6, 10) == pytest.approx(-0.2)
    assert rank_reward(1, 1, 10) == 0.0


# The README table, row by row.
def test_promoted_gets_format_bonus():
    r = compute_reward(answer([4, 1, 2, 3, 5, 6, 7, 8, 9, 10]), ex(4), alpha=0.1)
    assert r == RewardInfo(total=pytest.approx(0.4), rank=pytest.approx(0.3), fmt=1.0, parsed=True, new_rank=1, identity=False)


def test_copying_the_input_earns_nothing_when_conditional():
    r = compute_reward(answer(range(1, 11)), ex(4), alpha=0.1)
    assert r.total == 0.0 and r.identity is True and r.parsed is True and r.new_rank == 4


def test_copying_the_input_is_a_free_lunch_when_unconditional():
    r = compute_reward(answer(range(1, 11)), ex(4), alpha=0.1, conditional=False)
    assert r.total == pytest.approx(0.1)


def test_demoted():
    r = compute_reward(answer([1, 2, 3, 5, 6, 4, 7, 8, 9, 10]), ex(4), alpha=0.1)
    assert r.new_rank == 6 and r.total == pytest.approx(-0.2)
    assert compute_reward(answer([1, 2, 3, 5, 6, 4, 7, 8, 9, 10]), ex(4), 0.1, conditional=False).total == pytest.approx(-0.1)


def test_keeping_a_correct_first_place_earns_the_bonus():
    r = compute_reward(answer(range(1, 11)), ex(1), alpha=0.1)
    assert r.total == pytest.approx(0.1) and r.rank == 0.0


def test_true_first_place_always_gets_the_bonus_even_if_demoted():
    # Eq. 15 literally: the condition is (R_rank > 0 OR r^D == 1)
    r = compute_reward(answer([2, 1, 3, 4, 5, 6, 7, 8, 9, 10]), ex(1), alpha=0.1)
    assert r.rank == pytest.approx(-0.1) and r.total == pytest.approx(0.0)


def test_unparseable_falls_back_to_pre_rank():
    r = compute_reward("I don't know.", ex(4), alpha=0.1)
    assert r == RewardInfo(total=0.0, rank=0.0, fmt=0.0, parsed=False, new_rank=4, identity=False)
    assert compute_reward("I don't know.", ex(1), alpha=0.1).total == 0.0, "no bonus without a parse"


def test_partial_ranking_is_completed_by_the_parser():
    r = compute_reward(answer([7]), ex(7), alpha=0.5)  # only the top choice given
    assert r.new_rank == 1 and r.total == pytest.approx(0.6 + 0.5)
