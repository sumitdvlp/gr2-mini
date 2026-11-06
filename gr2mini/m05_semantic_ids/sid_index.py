"""Bidirectional item_id <-> SID lookup, saved as artifacts/sids.json.  (README §6)"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


class SIDIndex:
    def __init__(self, sids: Sequence[Sequence[int]] | torch.Tensor, codebook_size: int):
        """Store sids (row i = SID of item i) as tuples of python ints.

        Validate, raising ValueError if: there are no rows, rows have different lengths, any code is
        outside [0, codebook_size), or two items share a SID.

        Attributes the tests rely on: self.num_items, self.num_levels, self.codebook_size
        Build whatever lookup tables make sid() / item() / items_with_prefix() fast.
        """
        raise NotImplementedError("TODO(m05): implement SIDIndex.__init__")

    def sid(self, item_id: int) -> tuple[int, ...]:
        raise NotImplementedError("TODO(m05): implement SIDIndex.sid")

    def item(self, sid: Sequence[int]) -> int | None:
        """Item with exactly this SID, or None. Accepts tuple/list."""
        raise NotImplementedError("TODO(m05): implement SIDIndex.item")

    def sid_str(self, item_id: int) -> str:
        """sid_to_str(self.sid(item_id))."""
        raise NotImplementedError("TODO(m05): implement SIDIndex.sid_str")

    def item_from_text(self, text: str) -> int | None:
        """Item of the first well-formed, KNOWN SID in `text` (parse with find_all_sids using
        num_levels=self.num_levels; skip SIDs that map to no item). None if there is none."""
        raise NotImplementedError("TODO(m05): implement SIDIndex.item_from_text")

    def items_with_prefix(self, prefix: Sequence[int]) -> list[int]:
        """Sorted item ids whose SID starts with `prefix`. An empty prefix matches every item."""
        raise NotImplementedError("TODO(m05): implement SIDIndex.items_with_prefix")

    def save(self, path: str | Path) -> None:
        """JSON: {"num_levels": L, "codebook_size": C, "sids": [[...], ...]}. Create parent dirs."""
        raise NotImplementedError("TODO(m05): implement SIDIndex.save")

    @classmethod
    def load(cls, path: str | Path) -> "SIDIndex":
        raise NotImplementedError("TODO(m05): implement SIDIndex.load")
