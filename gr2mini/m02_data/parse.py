"""Parsing the raw Amazon files.  (see README, "The raw data")"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


def iter_gzip_lines(path: str | Path) -> Iterator[str]:
    """Yield the non-empty lines of a gzip-compressed UTF-8 text file, without trailing newlines.

    Hint: gzip.open(path, "rt", encoding="utf-8").
    """
    raise NotImplementedError("TODO(m02): implement iter_gzip_lines")


def read_reviews(path: str | Path) -> list[tuple[str, str, int]]:
    """Read the reviews file into a list of (reviewerID, asin, unixReviewTime) in FILE ORDER.

    Each line is a JSON object. unixReviewTime must be returned as an int.
    """
    raise NotImplementedError("TODO(m02): implement read_reviews")


def read_meta(path: str | Path, keep: set[str] | None = None) -> dict[str, dict]:
    """Read the metadata file into {asin: parsed_dict}.

    * Each line is a PYTHON dict literal -> ast.literal_eval (never eval!).
    * If `keep` is given, only return entries whose asin is in `keep`.
    * The parsed dict is returned as-is (all its keys); cleaning happens in item_text.py.
    """
    raise NotImplementedError("TODO(m02): implement read_meta")
