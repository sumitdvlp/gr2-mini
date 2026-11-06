import json

import pytest

from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m09_prompts.prompts import item_line
from gr2mini.m10_parsing.parsing import apply_ranking, parse_rerank_output
from gr2mini.m11_rewards.rewards import compute_reward
from gr2mini.m15_eval_demo.report import bootstrap_ci, markdown_table, render_case


def test_markdown_table():
    rows = {"Pre-rank": {"R@1": 0.28921, "N@5": 0.5}, "SFT": {"R@1": 0.2}}
    t = markdown_table(rows, ["R@1", "N@5"], digits=3)
    assert t.split("\n") == [
        "| Method | R@1 | N@5 |",
        "|---|---|---|",
        "| Pre-rank | 0.289 | 0.500 |",
        "| SFT | 0.200 | — |",
    ]


def test_bootstrap_ci():
    hits = [1.0] * 30 + [0.0] * 70
    mean, lo, hi = bootstrap_ci(hits, n_boot=500, seed=1)
    assert mean == pytest.approx(0.3) and lo < 0.3 < hi
    assert 0.18 < lo < 0.27 and 0.33 < hi < 0.42
    assert bootstrap_ci(hits, n_boot=500, seed=1) == (mean, lo, hi)
    assert bootstrap_ci([1.0, 1.0], n_boot=50) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([])


ITEMS = [{"item_id": i, "title": f"Product {i} title words here", "categories": ["Beauty", "Hair", f"C{i}"]} for i in range(30)]
INDEX = SIDIndex([(i, 0, 0, 1) for i in range(30)], codebook_size=32)
EX = {"user": 7, "history": [1, 2, 3], "candidates": list(range(10, 20)), "target": 11, "target_rank": 2}


def _case(text):
    parsed = parse_rerank_output(text, 10)
    reordered = apply_ranking(EX["candidates"], parsed.ranking)
    return render_case(EX, ITEMS, INDEX, reordered, parsed, compute_reward(text, EX)), parsed


def test_render_case_contents():
    text = json.dumps({"explanation": "The user bought hair care, so Product 11 fits.",
                       "recommendations": ["2", "1"]})
    out, _ = _case(text)
    for word in ("HISTORY", "PRE-RANKED", "RE-RANKED", "REASONING", "RESULT"):
        assert word in out
    for i in EX["history"] + EX["candidates"]:
        assert item_line(ITEMS[i], INDEX.sid_str(i), 8, 2) in out
    assert out.count("◀ bought next") == 2
    assert "2 → 1" in out and "+0.20" in out
    assert "The user bought hair care" in out


def test_render_case_unparseable():
    out, _ = _case("sorry, no idea")
    assert "(unparseable output)" in out and "2 → 2" in out and "+0.00" in out
