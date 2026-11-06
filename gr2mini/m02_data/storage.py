"""Saving / loading the processed dataset.

Layout of a processed directory:
    items.jsonl      one item record per line (see item_text.build_item_record), ordered by item_id
    sequences.json   {"user_ids": [...], "sequences": [[item_id, ...], ...]}
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def save_jsonl(records: Iterable[dict], path: str | Path) -> None:
    """Write one JSON object per line (UTF-8, ensure_ascii=False). Create parent dirs."""
    raise NotImplementedError("TODO(m02): implement save_jsonl")


def load_jsonl(path: str | Path) -> list[dict]:
    """Inverse of save_jsonl (skip blank lines)."""
    raise NotImplementedError("TODO(m02): implement load_jsonl")


def save_processed(out_dir: str | Path, items: list[dict], user_ids: list[str], sequences: list[list[int]]) -> None:
    """Write items.jsonl and sequences.json into `out_dir` (created if needed).

    Raise ValueError if items are not ordered by item_id 0..N-1, or if
    len(user_ids) != len(sequences).
    """
    raise NotImplementedError("TODO(m02): implement save_processed")


def load_processed(out_dir: str | Path) -> tuple[list[dict], list[str], list[list[int]]]:
    """Return (items, user_ids, sequences) as saved by save_processed."""
    raise NotImplementedError("TODO(m02): implement load_processed")
