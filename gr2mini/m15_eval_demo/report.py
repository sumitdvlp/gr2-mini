"""Results tables, bootstrap intervals and the case-study view.  (README)"""
from __future__ import annotations

import random  # noqa: F401
from typing import Sequence

from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m09_prompts.prompts import item_line  # noqa: F401
from gr2mini.m10_parsing.parsing import ParseResult
from gr2mini.m11_rewards.rewards import RewardInfo


def markdown_table(rows: dict[str, dict[str, float]], columns: Sequence[str], digits: int = 4) -> str:
    """A GitHub markdown table: header "| Method | col1 | col2 |", a separator "|---|---|---|", then one
    line per row (in dict order): "| name | v1 | v2 |" with values formatted f"{v:.{digits}f}".
    A missing value is rendered as "—". Lines joined with "\\n", no trailing newline."""
    raise NotImplementedError("TODO(m15): implement markdown_table")


def bootstrap_ci(
    hits: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, low, high): the mean of `hits` and the alpha/2, 1-alpha/2 quantiles of the means of
    n_boot resamples (with replacement, random.Random(seed)). Quantile = sorted_means[int(q * (n_boot - 1))].
    ValueError on empty input."""
    raise NotImplementedError("TODO(m15): implement bootstrap_ci")


def render_case(
    example: dict,
    items: Sequence[dict],
    index: SIDIndex,
    reordered: Sequence[int],
    parsed: ParseResult,
    reward: RewardInfo,
    max_title_words: int = 8,
) -> str:
    """The Appendix-F style view in the README. Requirements (tested):
        * contains "HISTORY", "PRE-RANKED", "RE-RANKED", "REASONING", "RESULT"
        * every history and candidate item appears (via item_line with max_category_levels=2)
        * the target is marked with "◀ bought next" in both lists (exactly 2 markers)
        * the RESULT line contains f"{example['target_rank']} → {reward.new_rank}" and the total
          reward formatted with a sign, f"{reward.total:+.2f}"
        * REASONING shows parsed.reasoning, or "(unparseable output)" if parsed.ranking is None
    Exact layout is up to you (side by side or one list after the other).
    """
    raise NotImplementedError("TODO(m15): implement render_case")
