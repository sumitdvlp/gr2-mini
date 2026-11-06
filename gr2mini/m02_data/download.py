"""Robust file download.  (see README, implementation order step 1)"""
from __future__ import annotations

from pathlib import Path


def download_file(url: str, dest: str | Path, overwrite: bool = False, chunk_size: int = 1 << 20) -> Path:
    """Download `url` to `dest` and return `dest` as a Path.

    Contract:
        * If `dest` already exists and not `overwrite`: do nothing, return dest.
        * Create parent directories as needed.
        * Stream into a temporary file `dest` + ".part" (e.g. "x.json.gz.part"), in chunks of
          `chunk_size` bytes, then rename it to `dest` only when the download completed.
        * If anything fails, delete the .part file and re-raise. `dest` must NOT exist afterwards
          (unless it existed before).
        * Use `urllib.request.urlopen` (standard library). It supports http(s):// AND file:// URLs,
          which is how the tests exercise this function without internet.

    Hint: `Path.replace` renames atomically and overwrites if needed.
    Bonus (not tested): show progress with tqdm using the Content-Length header.
    """
    raise NotImplementedError("TODO(m02): implement download_file")
