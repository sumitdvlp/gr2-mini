"""Turning raw metadata into clean item records and one-line item texts.  (README §4)"""
from __future__ import annotations

from typing import Any


def clean_text(s: Any) -> str:
    """None -> "". Otherwise str(s), HTML-unescaped (html.unescape), with every run of whitespace
    (spaces, tabs, newlines) collapsed to one space, and stripped.

    clean_text("  L&#39;Oreal\\n  Paris  ") == "L'Oreal Paris"
    """
    import html
    import re

    if s is None:
        return ""
    s = str(s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def build_item_record(item_id: int, asin: str, meta: dict | None) -> dict:
    """Build the record stored in items.jsonl.

    Returns {"item_id", "asin", "title", "brand", "categories", "text"} where
        * title, brand: clean_text of meta's "title" / "brand" ("" if missing; meta may be None)
        * categories:   the FIRST category path of meta["categories"] (a list of lists) with each
                        entry cleaned and empty entries dropped; [] if missing
        * text:         item_text(record)
    """
    if meta is None:
        title = ""
        brand = ""
        categories = []
    else:
        title = clean_text(meta.get("title"))
        brand = clean_text(meta.get("brand"))
        raw_categories = meta.get("categories", [])
        if isinstance(raw_categories, list) and raw_categories:
            first_path = raw_categories[0]
            if isinstance(first_path, list):
                categories = [clean_text(cat) for cat in first_path if clean_text(cat)]
            else:
                categories = []
        else:
            categories = []

    record = {
        "item_id": item_id,
        "asin": asin,
        "title": title,
        "brand": brand,
        "categories": categories,
    }
    record["text"] = item_text(record)
    return record


def item_text(record: dict) -> str:
    """One-line text for an item record, parts joined by " | ":

        "Title: <title>"                  always; "Unknown product" if title is empty
        "Brand: <brand>"                  only if brand is non-empty
        "Category: <c1> > <c2> > ..."     only if categories is non-empty

    Example: "Title: Hair Gel | Brand: Acme | Category: Beauty > Hair Care"
    """
    parts = []
    title = record.get("title", "")
    if title:
        parts.append(f"Title: {title}")
    else:
        parts.append("Title: Unknown product")

    brand = record.get("brand", "")
    if brand:
        parts.append(f"Brand: {brand}")

    categories = record.get("categories", [])
    if categories:
        category_path = " > ".join(categories)
        parts.append(f"Category: {category_path}")

    return " | ".join(parts)
