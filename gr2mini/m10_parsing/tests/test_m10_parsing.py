import pytest

from gr2mini.m10_parsing.parsing import (
    ParseResult,
    apply_ranking,
    json_candidates,
    parse_rerank_output,
    strip_think,
    validate_ranking,
)

N = 5


# ======================================================================= helpers
def test_strip_think():
    assert strip_think("<think>hmm {x}</think>\n answer") == "answer"
    assert strip_think("a<think>1</think>b<think>\n2\n</think>c") == "abc"
    assert strip_think("answer <think> never closed {\"recommendations\": [1]}") == "answer"
    assert strip_think("plain") == "plain"


def test_json_candidates():
    assert json_candidates('a {"x": "}"} b {"y": {"z": 1}}') == ['{"x": "}"}', '{"y": {"z": 1}}']
    assert json_candidates('{"s": "say \\"{hi}\\" ok"}') == ['{"s": "say \\"{hi}\\" ok"}']
    assert json_candidates('{"open": [1, 2') == []
    assert json_candidates("no braces") == []
    assert json_candidates('}{"a": 1}') == ['{"a": 1}']


@pytest.mark.parametrize("values, expected", [
    (["3", 1, "Candidate 2"], [2, 0, 1, 3, 4]),
    ([5, 4, 3, 2, 1], [4, 3, 2, 1, 0]),
    ([2, 2, 9, 0, -1, "x", None, 1.5, True], [1, 0, 2, 3, 4]),  # dupes, out of range, junk, bool
    ([], None),
    ([0, 6, "none"], None),
])
def test_validate_ranking(values, expected):
    assert validate_ranking(values, N) == expected


# ======================================================================= Stage I: JSON
def test_clean_json():
    r = parse_rerank_output('{"explanation": "because", "recommendations": ["2", "1", "3", "5", "4"]}', N)
    assert r == ParseResult("because", [1, 0, 2, 4, 3], "json")


def test_markdown_fence_and_surrounding_text():
    text = 'Sure! Here is my answer:\n```json\n{"explanation": "x", "recommendations": [3, 1]}\n```\nHope it helps'
    r = parse_rerank_output(text, N)
    assert r.ranking == [2, 0, 1, 3, 4] and r.stage == "json"


def test_alternative_keys():
    r = parse_rerank_output('{"reasoning": "r", "ranking": ["4"]}', N)
    assert r.reasoning == "r" and r.ranking == [3, 0, 1, 2, 4]


def test_last_json_wins():
    text = 'Draft: {"recommendations": [1, 2]} ... final: {"explanation": "final", "recommendations": [5]}'
    assert parse_rerank_output(text, N).ranking == [4, 0, 1, 2, 3]


def test_skips_later_json_without_ranking():
    text = '{"explanation": "e", "recommendations": [2]} {"note": "thanks"}'
    assert parse_rerank_output(text, N).ranking == [1, 0, 2, 3, 4]


def test_braces_inside_explanation():
    text = '{"explanation": "the user likes {shampoo} and } brackets", "recommendations": ["1", "2"]}'
    r = parse_rerank_output(text, N)
    assert r.stage == "json" and r.reasoning.startswith("the user likes {shampoo}")


def test_think_block_is_ignored():
    text = '<think>maybe {"recommendations": [5]}</think>{"explanation": "e", "recommendations": [2]}'
    assert parse_rerank_output(text, N).ranking == [1, 0, 2, 3, 4]


def test_empty_recommendations_falls_through():
    r = parse_rerank_output('{"explanation": "e", "recommendations": []}', N)
    assert r.ranking is None and r.stage == "none"


def test_invalid_json_values_fall_through():
    r = parse_rerank_output('{"explanation": "e", "recommendations": ["0", "9"]}', N)
    assert r.ranking is None


# ======================================================================= Stage II: regex
def test_truncated_json_uses_regex():
    text = '{"explanation": "short reason", "recommendations": ["2", "4", "1"'
    r = parse_rerank_output(text, N)
    assert r.stage == "regex" and r.ranking == [1, 3, 0, 2, 4] and r.reasoning == "short reason"


def test_plain_text_ranking():
    r = parse_rerank_output("I think the user wants hair care.\nFinal ranking: 3, 1, 2\n", N)
    assert r.stage == "regex" and r.ranking == [2, 0, 1, 3, 4] and r.reasoning is None


def test_candidate_words_in_ranking():
    r = parse_rerank_output("Ranking: Candidate 4 > Candidate 2 > Candidate 5", N)
    assert r.ranking == [3, 1, 4, 0, 2]


def test_regex_uses_last_occurrence_and_stops_at_newline():
    text = "ranking: 1, 2\nthinking more...\nRecommendations: [5, 4]\n(and 3 is also nice)"
    assert parse_rerank_output(text, N).ranking == [4, 3, 0, 1, 2]


# ======================================================================= failures
@pytest.mark.parametrize("text", ["", "I cannot help with that.", "{broken", "ranking: none of them", "<think>only thinking"])
def test_unparseable(text):
    assert parse_rerank_output(text, N) == ParseResult(None, None, "none")


# ======================================================================= apply_ranking
def test_apply_ranking():
    assert apply_ranking([10, 11, 12], [2, 0, 1]) == [12, 10, 11]
    c = [10, 11, 12]
    out = apply_ranking(c, None)
    assert out == c and out is not c
