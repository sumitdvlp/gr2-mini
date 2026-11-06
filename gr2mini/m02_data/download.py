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
    import ssl
    import urllib.request

    import certifi

    # Verify HTTPS against certifi's CA bundle instead of the Windows cert store, which may be
    # missing the root CA (Windows fetches roots lazily; OpenSSL can't trigger that).
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    dest_path = Path(dest)
    if dest_path.exists() and not overwrite:
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, context=ssl_context) as response, open(part_path, "wb") as out_file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        part_path.replace(dest_path)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise
    return dest_path
