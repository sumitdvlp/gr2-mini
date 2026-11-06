"""Bidirectional item_id <-> SID lookup, saved as artifacts/sids.json.  (README §6)"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from gr2mini.m05_semantic_ids.sid_format import find_all_sids, sid_to_str


class SIDIndex:
    def __init__(self, sids: Sequence[Sequence[int]] | torch.Tensor, codebook_size: int):
        """Store sids (row i = SID of item i) as tuples of python ints.

        Validate, raising ValueError if: there are no rows, rows have different lengths, any code is
        outside [0, codebook_size), or two items share a SID.

        Attributes the tests rely on: self.num_items, self.num_levels, self.codebook_size
        Build whatever lookup tables make sid() / item() / items_with_prefix() fast.
        """
        if isinstance(sids, torch.Tensor):
            rows = [tuple(int(c) for c in row) for row in sids.tolist()]
        else:
            rows = [tuple(int(c) for c in row) for row in sids]
        if not rows:
            raise ValueError("SIDIndex needs at least one SID")
        num_levels = len(rows[0])
        if num_levels == 0:
            raise ValueError("SIDs must have at least one level")

        sid_to_item: dict[tuple[int, ...], int] = {}
        for i, row in enumerate(rows):
            if len(row) != num_levels:
                raise ValueError(f"item {i} has {len(row)} levels, expected {num_levels}")
            if any(c < 0 or c >= codebook_size for c in row):
                raise ValueError(f"item {i} has a code outside [0, {codebook_size}): {row}")
            if row in sid_to_item:
                raise ValueError(f"items {sid_to_item[row]} and {i} share SID {row}")
            sid_to_item[row] = i

        self.codebook_size = int(codebook_size)
        self.num_levels = num_levels
        self.num_items = len(rows)
        self._sids: list[tuple[int, ...]] = rows
        self._sid_to_item = sid_to_item
        # Every prefix (empty .. full SID) -> sorted item ids; ascending i keeps each list sorted.
        prefix_map: dict[tuple[int, ...], list[int]] = {}
        for i, row in enumerate(rows):
            for k in range(num_levels + 1):
                prefix_map.setdefault(row[:k], []).append(i)
        self._prefix_to_items = prefix_map

    def sid(self, item_id: int) -> tuple[int, ...]:
        return self._sids[item_id]

    def item(self, sid: Sequence[int]) -> int | None:
        """Item with exactly this SID, or None. Accepts tuple/list."""
        return self._sid_to_item.get(tuple(int(c) for c in sid))

    def sid_str(self, item_id: int) -> str:
        """sid_to_str(self.sid(item_id))."""
        return sid_to_str(self.sid(item_id))

    def item_from_text(self, text: str) -> int | None:
        """Item of the first well-formed, KNOWN SID in `text` (parse with find_all_sids using
        num_levels=self.num_levels; skip SIDs that map to no item). None if there is none."""
        for s in find_all_sids(text, num_levels=self.num_levels):
            item = self._sid_to_item.get(s)
            if item is not None:
                return item
        return None

    def items_with_prefix(self, prefix: Sequence[int]) -> list[int]:
        """Sorted item ids whose SID starts with `prefix`. An empty prefix matches every item."""
        return list(self._prefix_to_items.get(tuple(int(c) for c in prefix), []))

    def save(self, path: str | Path) -> None:
        """JSON: {"num_levels": L, "codebook_size": C, "sids": [[...], ...]}. Create parent dirs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "sids": [list(s) for s in self._sids],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SIDIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls(data["sids"], codebook_size=data["codebook_size"])
        if idx.num_levels != data["num_levels"]:
            raise ValueError(f"num_levels mismatch in {path}: file says {data['num_levels']}, SIDs have {idx.num_levels}")
        return idx
