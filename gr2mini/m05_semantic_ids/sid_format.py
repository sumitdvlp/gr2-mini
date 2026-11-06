"""SID <-> token-string conversion.  (README §5; paper Appendix B)

    sid (12, 3, 250, 7)  <->  "<|sid_begin|><s_a_12><s_b_3><s_c_250><s_d_7><|sid_end|>"
"""
from __future__ import annotations

from typing import Sequence

SID_BEGIN = "<|sid_begin|>"
SID_END = "<|sid_end|>"
LEVEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def code_token(level: int, code: int) -> str:
    """code_token(0, 12) == "<s_a_12>";  code_token(3, 7) == "<s_d_7>". No zero padding."""
    raise NotImplementedError("TODO(m05): implement code_token")


def sid_to_str(sid: Sequence[int]) -> str:
    """SID_BEGIN + code tokens for levels 0..L-1 + SID_END. Accepts tuples, lists, or a 1-D tensor."""
    raise NotImplementedError("TODO(m05): implement sid_to_str")


def find_all_sids(text: str, num_levels: int | None = None) -> list[tuple[int, ...]]:
    """All WELL-FORMED SIDs in `text`, in order of appearance.

    A well-formed SID is SID_BEGIN, then one or more code tokens whose letters are exactly
    a, b, c, ... in order starting from "a", then SID_END, with nothing in between.
    If num_levels is given, only SIDs with exactly that many levels count.
    Malformed blocks are skipped (not an error).

    Hint: re.finditer over r"<\\|sid_begin\\|>((?:<s_[a-z]_\\d+>)+)<\\|sid_end\\|>", then check the letters.
    """
    raise NotImplementedError("TODO(m05): implement find_all_sids")


def parse_sid(text: str, num_levels: int | None = None) -> tuple[int, ...] | None:
    """The first well-formed SID in `text` (see find_all_sids), or None."""
    raise NotImplementedError("TODO(m05): implement parse_sid")


def all_sid_tokens(num_levels: int, codebook_size: int) -> list[str]:
    """Every token the LLM needs: [SID_BEGIN, SID_END] followed by code_token(l, c) for
    l in range(num_levels), c in range(codebook_size) (level-major order). 2 + L*C strings."""
    raise NotImplementedError("TODO(m05): implement all_sid_tokens")
