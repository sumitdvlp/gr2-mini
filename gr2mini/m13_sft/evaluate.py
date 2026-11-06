"""Re-ranking evaluation shared by m13, m14 and m15.  (README "Evaluation")"""
from __future__ import annotations

from typing import Callable, Sequence

from gr2mini.m01_metrics.metrics import ranking_report  # noqa: F401
from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m09_prompts.prompts import build_rerank_messages  # noqa: F401
from gr2mini.m10_parsing.parsing import apply_ranking, parse_rerank_output  # noqa: F401
from gr2mini.m11_rewards.rewards import compute_reward  # noqa: F401

GenerateFn = Callable[[list[list[dict]]], list[str]]


def pre_rank_report(examples: Sequence[dict]) -> dict[str, float]:
    """ranking_report of the candidate lists as retrieved (the "Pre-rank" row)."""
    raise NotImplementedError("TODO(m13): implement pre_rank_report")


def evaluate_reranker(
    examples: Sequence[dict],
    items: Sequence[dict],
    index: SIDIndex,
    prompt_cfg: dict,
    generate_fn: GenerateFn,
    alpha: float = 0.1,
    return_outputs: bool = False,
) -> dict:
    """Generate (ONE generate_fn call with all policy prompts), parse, score.

    Returns ranking_report(...) of the re-ordered lists, plus
        "parse_rate":    fraction of outputs with a parsed ranking
        "identity_rate": fraction of outputs that parse to exactly the input order
        "mean_reward":   mean compute_reward(text, ex, alpha, conditional=True).total
        "n":             number of examples
    If return_outputs: also "outputs": the list of generated texts.
    """
    raise NotImplementedError("TODO(m13): implement evaluate_reranker")
