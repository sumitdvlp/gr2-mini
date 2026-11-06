import json

import pytest

from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m09_prompts.prompts import (
    KNOWLEDGE_PRIMING,
    SYSTEM_PROMPT,
    assistant_json,
    build_rerank_messages,
    candidate_sids,
    item_line,
    truncate_words,
)

ITEMS = [{"item_id": i, "title": f"Product number {i} with a long descriptive title",
          "categories": ["Beauty", "Hair Care", "Styling", f"Cat{i}"]} for i in range(20)]
ITEMS[5]["categories"] = []
ITEMS[6]["title"] = ""
INDEX = SIDIndex([(i, i % 3, 0, 1) for i in range(20)], codebook_size=32)
CFG = {"max_history": 3, "max_title_words": 4, "max_category_levels": 2, "knowledge_priming": False}
EX = {"user": 1, "history": [0, 1, 2, 3, 4], "candidates": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
      "target": 13, "target_rank": 4}


def test_truncate_words():
    assert truncate_words("a  b   c d", 2) == "a b"
    assert truncate_words("one", 5) == "one"


def test_item_line():
    assert item_line(ITEMS[2], INDEX.sid_str(2), 4, 2) == f"{INDEX.sid_str(2)} | Product number 2 with | Styling > Cat2"
    assert item_line(ITEMS[5], "S", 2, 2) == "S | Product number"
    assert item_line(ITEMS[6], "S", 4, 1) == "S | Unknown product | Cat6"


def test_policy_prompt_layout():
    msgs = build_rerank_messages(EX, ITEMS, INDEX, "policy", CFG)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    user = msgs[1]["content"]
    assert user.startswith("Purchase history (oldest first):")
    # only the last 3 history items, numbered and in order
    assert INDEX.sid_str(0) not in user and INDEX.sid_str(1) not in user
    for k, item in enumerate([2, 3, 4], start=1):
        assert f"{k}. {item_line(ITEMS[item], INDEX.sid_str(item), 4, 2)}" in user
    assert "Candidates:" in user
    pos = [user.index(f"Candidate {k}: {item_line(ITEMS[c], INDEX.sid_str(c), 4, 2)}")
           for k, c in enumerate(EX["candidates"], start=1)]
    assert pos == sorted(pos), "candidates must appear in the given (pre-rank) order"


def test_no_target_leakage_in_policy_and_rejection():
    other = {**EX, "target": 17, "target_rank": 8}
    for variant in ("policy", "rejection"):
        a = build_rerank_messages(EX, ITEMS, INDEX, variant, CFG)
        b = build_rerank_messages(other, ITEMS, INDEX, variant, CFG)
        assert a == b, f"{variant} prompt depends on the target!"
        assert "target" not in a[1]["content"].lower()
    assert build_rerank_messages(EX, ITEMS, INDEX, "policy", CFG) == build_rerank_messages(EX, ITEMS, INDEX, "rejection", CFG)


def test_targeted_prompt_contains_hint():
    t = build_rerank_messages(EX, ITEMS, INDEX, "targeted", CFG)[1]["content"]
    p = build_rerank_messages(EX, ITEMS, INDEX, "policy", CFG)[1]["content"]
    assert t.startswith(p) and "Candidate 4 next" in t


def test_knowledge_priming_flag_and_invalid_variant():
    kp = build_rerank_messages(EX, ITEMS, INDEX, "policy", {**CFG, "knowledge_priming": True})
    assert kp[0]["content"] == SYSTEM_PROMPT + KNOWLEDGE_PRIMING
    with pytest.raises(ValueError):
        build_rerank_messages(EX, ITEMS, INDEX, "cheating", CFG)


def test_assistant_json_roundtrip():
    s = assistant_json("Crème is great, \"quoted\".", [3, 1, 2])
    assert json.loads(s) == {"explanation": "Crème is great, \"quoted\".", "recommendations": ["3", "1", "2"]}
    assert "Crème" in s and list(json.loads(s)) == ["explanation", "recommendations"]


def test_candidate_sids():
    s = candidate_sids(EX, INDEX)
    assert s == {INDEX.sid(i) for i in EX["history"] + EX["candidates"]}
