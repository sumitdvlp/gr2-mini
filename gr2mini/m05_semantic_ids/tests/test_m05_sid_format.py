import torch

from gr2mini.m05_semantic_ids.sid_format import (
    SID_BEGIN,
    SID_END,
    all_sid_tokens,
    code_token,
    find_all_sids,
    parse_sid,
    sid_to_str,
)

S = "<|sid_begin|><s_a_12><s_b_3><s_c_250><s_d_7><|sid_end|>"


def test_code_token():
    assert code_token(0, 12) == "<s_a_12>"
    assert code_token(3, 7) == "<s_d_7>"
    assert code_token(1, 0) == "<s_b_0>"


def test_sid_to_str_accepts_tuple_list_tensor():
    assert sid_to_str((12, 3, 250, 7)) == S
    assert sid_to_str([12, 3, 250, 7]) == S
    assert sid_to_str(torch.tensor([12, 3, 250, 7])) == S
    assert sid_to_str((5,)) == "<|sid_begin|><s_a_5><|sid_end|>"


def test_roundtrip():
    for sid in [(0, 0, 0, 0), (255, 1, 99, 3), (7, 7)]:
        assert parse_sid(sid_to_str(sid)) == sid


def test_parse_sid_inside_text_and_none():
    assert parse_sid(f"I recommend {S}, it fits.") == (12, 3, 250, 7)
    assert parse_sid("no sid here") is None
    assert parse_sid("") is None


def test_parse_skips_malformed_blocks():
    bad_order = "<|sid_begin|><s_a_1><s_c_5><|sid_end|>"  # skips level b
    bad_start = "<|sid_begin|><s_b_1><s_c_5><|sid_end|>"  # doesn't start at a
    unclosed = "<|sid_begin|><s_a_1><s_b_2>"
    junk_inside = "<|sid_begin|><s_a_1> <s_b_2><|sid_end|>"  # space inside
    for bad in (bad_order, bad_start, unclosed, junk_inside):
        assert parse_sid(bad) is None, bad
    assert parse_sid(bad_order + " then " + S) == (12, 3, 250, 7)


def test_num_levels_filter():
    two = sid_to_str((1, 2))
    assert parse_sid(two + S, num_levels=4) == (12, 3, 250, 7)
    assert parse_sid(two, num_levels=4) is None
    assert parse_sid(two) == (1, 2)


def test_find_all_sids_order():
    text = f"{sid_to_str((1, 2, 3, 4))} and {S} and <|sid_begin|><s_a_9><|sid_end|>"
    assert find_all_sids(text) == [(1, 2, 3, 4), (12, 3, 250, 7), (9,)]
    assert find_all_sids(text, num_levels=4) == [(1, 2, 3, 4), (12, 3, 250, 7)]


def test_all_sid_tokens():
    toks = all_sid_tokens(num_levels=3, codebook_size=4)
    assert len(toks) == 2 + 3 * 4
    assert toks[:2] == [SID_BEGIN, SID_END]
    assert toks[2:6] == ["<s_a_0>", "<s_a_1>", "<s_a_2>", "<s_a_3>"]
    assert toks[-1] == "<s_c_3>"
    assert len(set(toks)) == len(toks)
