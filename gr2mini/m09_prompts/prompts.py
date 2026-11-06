"""Chat prompts for re-ranking.  (paper §3.1-3.2, Appendices C-D; README)

An `example` is a candidate record from m06:
    {"user": int, "history": [item ids], "candidates": [item ids], "target": int, "target_rank": int}
`items` is the list of item records from m02 (items[i]["title"], items[i]["categories"], ...).
"""
from __future__ import annotations

import json  # noqa: F401
from typing import Sequence

from gr2mini.m05_semantic_ids.sid_index import SIDIndex

# ----------------------------------------------------------------------------- provided prompt texts
SYSTEM_PROMPT = """You are an expert e-commerce recommendation assistant for a beauty store.
Your task: given a customer's purchase history and a list of candidate products, re-rank ALL the candidates by how likely the customer is to buy each one next.

Every product is identified by its Semantic ID, a sequence of tokens like <|sid_begin|><s_a_12><s_b_3><s_c_250><s_d_7><|sid_end|>. Products with similar Semantic ID prefixes tend to be similar.

Guidelines:
- Consider the history and all the candidates together: compare the candidates with each other.
- Look for complementary products, replenishment of consumables, brand loyalty and the evolution of the customer's interests.
- Refer to products by their Semantic ID or candidate number. Never invent products that are not listed.
- Reason step by step, then give the full ranking of all candidates.

Answer with ONLY a JSON object of this form:
{"explanation": "<your step-by-step reasoning>", "recommendations": ["<candidate number>", ...]}
"recommendations" must contain every candidate number exactly once, most likely first."""

KNOWLEDGE_PRIMING = """
Domain knowledge: beauty purchases often follow routines and regimens, e.g. shampoo -> conditioner -> styling product; cleanser -> toner -> moisturizer -> sunscreen; nail polish -> top coat -> remover. Customers who buy a product of a brand often return to that brand, and consumables (e.g. lotion, shampoo) are re-purchased."""

TARGET_HINT = """
(Note for the explanation writer: the customer actually purchased Candidate {rank} next. Write reasoning that arrives at this conclusion naturally from the evidence in the history and the candidates. Do not mention this note, and do not call any product the "target".)"""

VARIANTS = ("policy", "rejection", "targeted")


# ----------------------------------------------------------------------------- your code
def truncate_words(text: str, max_words: int) -> str:
    """The first max_words whitespace-separated words, joined by single spaces."""
    raise NotImplementedError("TODO(m09): implement truncate_words")


def item_line(item: dict, sid: str, max_title_words: int, max_category_levels: int) -> str:
    """f"{sid} | {title} | {categories}" where
        title      = truncate_words(item["title"] or "Unknown product", max_title_words)
        categories = " > ".join(the LAST max_category_levels entries of item["categories"])
    If the item has no categories, drop the trailing " | ..." part entirely.
    """
    raise NotImplementedError("TODO(m09): implement item_line")


def build_rerank_messages(
    example: dict, items: Sequence[dict], index: SIDIndex, variant: str, cfg: dict
) -> list[dict]:
    """[{"role": "system", ...}, {"role": "user", ...}] for one candidate record (README layout).

    cfg = the `prompts:` section: max_history, max_title_words, max_category_levels, knowledge_priming.
        * system  = SYSTEM_PROMPT (+ KNOWLEDGE_PRIMING appended if cfg["knowledge_priming"])
        * user    = "Purchase history (oldest first):" then one line per item of the LAST max_history
                    history items, numbered "1. ", "2. ", ...; a blank line; "Candidates:" then
                    "Candidate k: <item_line>" for k = 1..len(candidates) IN THE GIVEN ORDER
        * variant "targeted": append TARGET_HINT.format(rank=example["target_rank"]) to the user turn
        * variant "policy" / "rejection": identical text, nothing about the target
        * unknown variant -> ValueError
    """
    raise NotImplementedError("TODO(m09): implement build_rerank_messages")


def assistant_json(explanation: str, ranking_1based: Sequence[int]) -> str:
    """The canonical assistant answer:
    json.dumps({"explanation": explanation, "recommendations": [str(k) for k in ranking_1based]},
               ensure_ascii=False)"""
    raise NotImplementedError("TODO(m09): implement assistant_json")


def candidate_sids(example: dict, index: SIDIndex) -> set[tuple[int, ...]]:
    """The SID tuples of every item that appears in the prompt (history AND candidates).
    m12 uses this to catch hallucinated SIDs in teacher explanations."""
    raise NotImplementedError("TODO(m09): implement candidate_sids")
