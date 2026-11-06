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
    import json

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Inverse of save_jsonl (skip blank lines)."""
    import json

    in_path = Path(path)
    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                records.append(record)
    return records


def save_processed(out_dir: str | Path, items: list[dict], user_ids: list[str], sequences: list[list[int]]) -> None:
    """Write items.jsonl and sequences.json into `out_dir` (created if needed).

    Raise ValueError if items are not ordered by item_id 0..N-1, or if
    len(user_ids) != len(sequences).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Validate items are ordered by item_id 0..N-1
    for expected_id, item in enumerate(items):
        if item.get("item_id") != expected_id:
            raise ValueError(f"Item at index {expected_id} has item_id {item.get('item_id')}, expected {expected_id}")

    # Validate user_ids and sequences lengths match
    if len(user_ids) != len(sequences):
        raise ValueError(f"Length of user_ids ({len(user_ids)}) does not match length of sequences ({len(sequences)})")

    # Save items.jsonl
    save_jsonl(items, out_path / "items.jsonl")

    # Save sequences.json
    import json
    sequences_data = {"user_ids": user_ids, "sequences": sequences}
    with open(out_path / "sequences.json", "w", encoding="utf-8") as f:
        json.dump(sequences_data, f, ensure_ascii=False, indent=2)


def load_processed(out_dir: str | Path) -> tuple[list[dict], list[str], list[list[int]]]:
    """Return (items, user_ids, sequences) as saved by save_processed."""
    out_path = Path(out_dir)
    items = load_jsonl(out_path / "items.jsonl")
    import json
    with open(out_path / "sequences.json", "r", encoding="utf-8") as f:
        sequences_data = json.load(f)
    user_ids = sequences_data.get("user_ids", [])
    sequences = sequences_data.get("sequences", [])
    return items, user_ids, sequences
