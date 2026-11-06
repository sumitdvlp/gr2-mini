import json

import pytest

from gr2mini.m02_data.item_text import build_item_record, clean_text, item_text
from gr2mini.m02_data.storage import load_jsonl, load_processed, save_jsonl, save_processed


# ---------------------------------------------------------------- clean_text / records
def test_clean_text():
    assert clean_text("  L&#39;Oreal\n  Paris\t ") == "L'Oreal Paris"
    assert clean_text("Tom &amp; Jerry") == "Tom & Jerry"
    assert clean_text(None) == ""
    assert clean_text(12) == "12"


def test_build_item_record_full():
    meta = {"asin": "B1", "title": "L&#39;Oreal  Shampoo", "brand": " LOreal ",
            "categories": [["Beauty", "Hair Care", "Shampoos"], ["Beauty", "Gifts"]], "price": 3.5}
    rec = build_item_record(7, "B1", meta)
    assert rec == {
        "item_id": 7,
        "asin": "B1",
        "title": "L'Oreal Shampoo",
        "brand": "LOreal",
        "categories": ["Beauty", "Hair Care", "Shampoos"],
        "text": "Title: L'Oreal Shampoo | Brand: LOreal | Category: Beauty > Hair Care > Shampoos",
    }


def test_build_item_record_missing_meta_and_fields():
    rec = build_item_record(0, "B0", None)
    assert rec["title"] == "" and rec["brand"] == "" and rec["categories"] == []
    assert rec["text"] == "Title: Unknown product"

    rec2 = build_item_record(1, "B1", {"title": "Gel", "categories": [["Beauty", " ", "Hair"]]})
    assert rec2["categories"] == ["Beauty", "Hair"]
    assert rec2["text"] == "Title: Gel | Category: Beauty > Hair"


def test_item_text_brand_only():
    assert item_text({"title": "", "brand": "Acme", "categories": []}) == "Title: Unknown product | Brand: Acme"


# ---------------------------------------------------------------- storage
def test_jsonl_roundtrip_with_unicode(tmp_path):
    recs = [{"a": 1, "t": "Crème brûlée ✓"}, {"a": 2, "t": ""}]
    p = tmp_path / "sub" / "x.jsonl"
    save_jsonl(recs, p)
    assert load_jsonl(p) == recs
    assert "Crème" in p.read_text(encoding="utf-8")  # ensure_ascii=False


def _items(n):
    return [build_item_record(i, f"B{i}", {"title": f"item {i}"}) for i in range(n)]


def test_processed_roundtrip(tmp_path):
    items, users, seqs = _items(3), ["uA", "uB"], [[0, 1], [2, 1, 0]]
    save_processed(tmp_path / "proc", items, users, seqs)
    assert (tmp_path / "proc" / "items.jsonl").exists()
    assert json.loads((tmp_path / "proc" / "sequences.json").read_text(encoding="utf-8")) == {
        "user_ids": users, "sequences": seqs}
    assert load_processed(tmp_path / "proc") == (items, users, seqs)


def test_save_processed_validates(tmp_path):
    items = _items(3)
    with pytest.raises(ValueError):
        save_processed(tmp_path, [items[1], items[0], items[2]], ["u"], [[0]])
    with pytest.raises(ValueError):
        save_processed(tmp_path, items, ["u1", "u2"], [[0]])
