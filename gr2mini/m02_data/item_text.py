"""Turning raw metadata into clean item records and one-line item texts.  (README §4)"""
from __future__ import annotations

from typing import Any


def clean_text(s: Any) -> str:
    """None -> "". Otherwise str(s), HTML-unescaped (html.unescape), with every run of whitespace
    (spaces, tabs, newlines) collapsed to one space, and stripped.

    clean_text("  L&#39;Oreal\\n  Paris  ") == "L'Oreal Paris"
    """
    raise NotImplementedError("TODO(m02): implement clean_text")


def build_item_record(item_id: int, asin: str, meta: dict | None) -> dict:
    """Build the record stored in items.jsonl.

    Returns {"item_id", "asin", "title", "brand", "categories", "text"} where
        * title, brand: clean_text of meta's "title" / "brand" ("" if missing; meta may be None)
        * categories:   the FIRST category path of meta["categories"] (a list of lists) with each
                        entry cleaned and empty entries dropped; [] if missing
        * text:         item_text(record)
    """
    raise NotImplementedError("TODO(m02): implement build_item_record")


def item_text(record: dict) -> str:
    """One-line text for an item record, parts joined by " | ":

        "Title: <title>"                  always; "Unknown product" if title is empty
        "Brand: <brand>"                  only if brand is non-empty
        "Category: <c1> > <c2> > ..."     only if categories is non-empty

    Example: "Title: Hair Gel | Brand: Acme | Category: Beauty > Hair Care"
    """
    raise NotImplementedError("TODO(m02): implement item_text")
