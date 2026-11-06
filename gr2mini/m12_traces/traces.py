"""Targeted (Eq. 11) and rejection (Eq. 12) sampling of reasoning traces.  (README)"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from gr2mini.m05_semantic_ids.sid_format import find_all_sids  # noqa: F401
from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m09_prompts.prompts import assistant_json, build_rerank_messages, candidate_sids  # noqa: F401
from gr2mini.m10_parsing.parsing import ParseResult, parse_rerank_output  # noqa: F401
from gr2mini.m12_traces.teacher import TeacherClient


def check_trace(text: str, example: dict, index: SIDIndex) -> tuple[bool, str, ParseResult]:
    """(accepted, reason, parsed). Checks, in this order, with these reasons:

        no ranking parsed                                   -> (False, "unparsed", parsed)
        parsed.ranking[0] != example["target_rank"] - 1     -> (False, "wrong_top1", parsed)
        reasoning missing or blank                          -> (False, "no_reasoning", parsed)
        any SID in the reasoning (find_all_sids) that is
        not in candidate_sids(example, index)               -> (False, "hallucinated_sid", parsed)
        otherwise                                           -> (True, "ok", parsed)
    """
    raise NotImplementedError("TODO(m12): implement check_trace")


def sample_trace(
    client: TeacherClient,
    example: dict,
    items: Sequence[dict],
    index: SIDIndex,
    mode: str,
    prompt_cfg: dict,
    trace_cfg: dict,
) -> dict:
    """One record (README "What gets saved") for one example.

    mode "targeted":  the "targeted" prompt, ONE call.
    mode "rejection": the "rejection" prompt, up to trace_cfg["max_attempts"] calls; stop at the first
                      accepted trace.
    Each call: client.chat(messages, n=1, temperature=trace_cfg["temperature"],
                           max_tokens=trace_cfg["max_tokens"])[0], then check_trace.
    Other modes -> ValueError.

    Returns {"user", "mode", "accepted", "attempts", "reason" (of the last check), "raw" (last text),
             "messages": policy prompt + [{"role": "assistant",
                         "content": assistant_json(reasoning, [r + 1 for r in ranking])}] if accepted else None}
    """
    raise NotImplementedError("TODO(m12): implement sample_trace")


def generate_traces(
    client: TeacherClient,
    examples: Sequence[dict],
    items: Sequence[dict],
    index: SIDIndex,
    mode: str,
    out_path: str | Path,
    prompt_cfg: dict,
    trace_cfg: dict,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Run sample_trace over `examples`, appending one JSON line per example to out_path. Resumable.

    * If out_path exists, read it; skip examples whose user already has a record with this mode.
    * Append each new record immediately (open in "a" mode, write, flush). Create parent dirs.
    * Returns stats over the records produced IN THIS CALL:
        {"tried": n, "accepted": k, "acceptance_rate": k / n (0.0 if n == 0),
         "calls": total attempts, "skipped": #examples skipped because already done}
    """
    raise NotImplementedError("TODO(m12): implement generate_traces")
