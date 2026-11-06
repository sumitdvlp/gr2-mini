"""Parsing the raw Amazon files.  (see README, "The raw data")"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


def iter_gzip_lines(path: str | Path) -> Iterator[str]:
    """Yield the non-empty lines of a gzip-compressed UTF-8 text file, without trailing newlines.

    Hint: gzip.open(path, "rt", encoding="utf-8").
    """
    unzip_path = Path(path)
    import gzip
    with gzip.open(unzip_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line
                


def read_reviews(path: str | Path) -> list[tuple[str, str, int]]:
    """Read the reviews file into a list of (reviewerID, asin, unixReviewTime) in FILE ORDER.

    Each line is a JSON object. unixReviewTime must be returned as an int.
    """
    import json
    import gzip
    
    reviews_path = Path(path)
    reviews = []
    with gzip.open(reviews_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    review = json.loads(line)
                    reviewer_id = review.get("reviewerID")
                    asin = review.get("asin")
                    unix_review_time = review.get("unixReviewTime")
                    if reviewer_id is not None and asin is not None and unix_review_time is not None:
                        reviews.append((reviewer_id, asin, int(unix_review_time)))
                except (ValueError, json.JSONDecodeError):
                    # Skip lines that cannot be parsed
                    continue
    return reviews


def read_meta(path: str | Path, keep: set[str] | None = None) -> dict[str, dict]:
    """Read the metadata file into {asin: parsed_dict}.

    * Each line is a PYTHON dict literal -> ast.literal_eval (never eval!).
    * If `keep` is given, only return entries whose asin is in `keep`.
    * The parsed dict is returned as-is (all its keys); cleaning happens in item_text.py.
    """
    import ast
    import gzip
    
    
    meta_path = Path(path)
    meta = {}
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    parsed_dict = ast.literal_eval(line)
                    asin = parsed_dict.get("asin")
                    if asin is not None and (keep is None or asin in keep):
                        meta[asin] = parsed_dict
                except (ValueError, SyntaxError):
                    # Skip lines that cannot be parsed
                    continue
    return meta
