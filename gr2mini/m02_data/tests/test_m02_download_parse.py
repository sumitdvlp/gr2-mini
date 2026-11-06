import gzip
import json

import pytest

from gr2mini.m02_data.download import download_file
from gr2mini.m02_data.parse import iter_gzip_lines, read_meta, read_reviews


# ---------------------------------------------------------------- download_file
def test_download_file_copies_bytes_and_creates_dirs(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello amazon" * 1000)
    dest = tmp_path / "a" / "b" / "out.bin"
    out = download_file(src.as_uri(), dest, chunk_size=100)
    assert out == dest
    assert dest.read_bytes() == src.read_bytes()
    assert not (tmp_path / "a" / "b" / "out.bin.part").exists()


def test_download_file_skips_existing_unless_overwrite(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"NEW")
    dest = tmp_path / "out.bin"
    dest.write_bytes(b"OLD")
    download_file(src.as_uri(), dest)
    assert dest.read_bytes() == b"OLD"
    download_file(src.as_uri(), dest, overwrite=True)
    assert dest.read_bytes() == b"NEW"


def test_download_file_failure_leaves_no_files(tmp_path):
    missing = (tmp_path / "does_not_exist.bin").as_uri()
    dest = tmp_path / "out.bin"
    with pytest.raises(OSError):  # urllib's URLError is a subclass of OSError
        download_file(missing, dest)
    assert not dest.exists()
    assert not (tmp_path / "out.bin.part").exists()


# ---------------------------------------------------------------- parsing
def _write_gz(path, lines):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for line in lines:
            line = line.rstrip("\n")
            line = line.strip()
            if line:
                f.write(line + "\n")


def test_iter_gzip_lines_strips_newlines_and_skips_blank(tmp_path):
    p = tmp_path / "x.gz"
    _write_gz(p, ["a", "", "b c", "   "])
    assert list(iter_gzip_lines(p)) == ["a", "b c"]


def test_read_reviews(tmp_path):
    p = tmp_path / "reviews.json.gz"
    rows = [
        {"reviewerID": "U1", "asin": "B2", "overall": 5.0, "unixReviewTime": 1391040000, "reviewText": "nice"},
        {"reviewerID": "U2", "asin": "B1", "overall": 1.0, "unixReviewTime": "1300000000", "summary": "bad"},
    ]
    _write_gz(p, [json.dumps(r) for r in rows])
    out = read_reviews(p)
    assert out == [("U1", "B2", 1391040000), ("U2", "B1", 1300000000)]
    assert all(isinstance(t, int) for _, _, t in out)


META_LINES = [
    "{'asin': 'B001', 'title': 'L&#39;Oreal Shampoo', 'categories': [['Beauty', 'Hair Care', 'Shampoos']], 'brand': 'LOreal'}",
    # Python-literal quirks: a double-quoted string containing a single quote, nested dicts, no title
    "{'asin': 'B002', 'description': \"Bee's wax lotion\", 'salesRank': {'Beauty': 540255}, 'categories': [['Beauty']]}",
    "{'asin': 'B003', 'title': 'Lipstick', 'categories': [['Beauty', 'Makeup', 'Lips'], ['Beauty', 'Gift Sets']]}",
]


def test_read_meta_parses_python_literals(tmp_path):
    p = tmp_path / "meta.json.gz"
    _write_gz(p, META_LINES)
    meta = read_meta(p)
    assert set(meta) == {"B001", "B002", "B003"}
    assert meta["B001"]["title"] == "L&#39;Oreal Shampoo"  # raw: cleaning happens later
    assert meta["B002"]["description"] == "Bee's wax lotion"
    assert meta["B002"]["salesRank"] == {"Beauty": 540255}
    assert meta["B003"]["categories"][1] == ["Beauty", "Gift Sets"]


def test_read_meta_keep_filter(tmp_path):
    p = tmp_path / "meta.json.gz"
    _write_gz(p, META_LINES)
    meta = read_meta(p, keep={"B003", "B999"})
    assert set(meta) == {"B003"}


def test_read_meta_does_not_eval_code(tmp_path):
    p = tmp_path / "meta.json.gz"
    marker = tmp_path / "pwned.txt"
    evil = f"__import__('pathlib').Path(r'{marker}').write_text('x')"
    _write_gz(p, [META_LINES[0], evil])
    try:
        read_meta(p)  # raising on the bad line OR skipping it are both acceptable
    except (ValueError, SyntaxError):
        pass
    assert not marker.exists(), "read_meta executed code from the file! Use ast.literal_eval."
