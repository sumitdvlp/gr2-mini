# m02 — Data: Amazon Beauty → chronological user sequences

**Paper:** §5.1.1 and Table 1 (Beauty: 22,363 users, 12,101 items, avg. sequence length 8.87).
**Time:** ~2–3 hours. **Files you implement:** `download.py`, `parse.py`, `sequences.py`,
`item_text.py`, `storage.py`, `run.py`.

## The raw data

The Amazon Review dataset (McAuley et al., 2015), *Beauty* category, 5-core version:

**`reviews_Beauty_5.json.gz`**: one JSON object per line, 198,502 lines:
```json
{"reviewerID": "A1YJEY40YUW4SE", "asin": "7806397051", "overall": 1.0,
 "unixReviewTime": 1391040000, "reviewText": "...", "summary": "..."}
```
We only need `(reviewerID, asin, unixReviewTime)`. A review counts as an *interaction*.

**`meta_Beauty.json.gz`**: one product per line, 259k lines. ⚠️ These lines are **Python dict literals**,
not JSON (single quotes!):
```python
{'asin': '7806397051', 'title': 'Xtreme Brite Brightening Gel 1oz.',
 'categories': [['Beauty', 'Hair Care', 'Styling Products']], 'brand': 'Xtreme Brite'}
```
`json.loads` fails on these. Use `ast.literal_eval`, which parses Python *literals* safely.
**Never** `eval()` a line downloaded from the internet: that would execute arbitrary code.

Many items lack `title` or `brand`, and titles contain HTML entities (`L&#39;Oreal`, `&amp;`).
Use `html.unescape`.

---

## 1. k-core filtering

*k-core* = keep only users AND items with ≥ k interactions. It has to be **iterative**: dropping
a rare item can push a user below k, which can push another item below k, and so on.

```
k = 2         i1  i2  i3              round 1: i3 has 1 interaction → drop it
     u1        x   x                   → u3 now has 1 interaction  → drop u3
     u2        x   x                  round 2: i2 has 2 (u1, u2) ✓  → stable
     u3            x   x              result: {u1, u2} × {i1, i2}
```

The file is *already* 5-core, but implement the filter anyway. It's a classic, and
`tiny.yaml` relies on the same code path.

**Order in `run.py`:** read → `k_core_filter` → `subsample_users` (tiny only) → `build_sequences`.
Subsampling *after* the k-core keeps every sampled user at ≥ 5 interactions.

## 2. Sequences and IDs

* `build_sequences`: group by user, sort by timestamp. Many reviews share a day (the timestamps are
  day-granular!), so the sort must be **stable**: ties keep file order. Python's `sorted` is stable.
* `remap_ids`: map strings to dense ints `0..N-1`. Sort the ASINs (and the user ids) so the mapping is
  **deterministic**. Your SIDs, checkpoints and caches all depend on item ids, and they must not change
  between runs.

## 3. The 4-way split (and why it isn't the usual 3-way)

The paper uses leave-one-out: last item = test, second-to-last = validation. We need **one more**
held-out position, because two models are trained in sequence:

```
seq = [ i1  i2  ...  i(n-3) | i(n-2) | i(n-1) | i(n) ]
      └─ retriever_train ──┘    │        │       │
      └─ history ──────────┘ rerank_train│       │      ← re-ranker trains on these
      └─ history ───────────────────┘  valid     │
      └─ history ────────────────────────────┘ test
```

**Why?** The retriever (SASRec, m06) produces the 10 candidates the re-ranker re-orders. If the retriever
had trained on the target it's asked about, its candidate lists would be unrealistically good (target
near the top, every time). The re-ranker would then learn on lists that look nothing like test lists.
Holding out `i(n-2)` from the retriever gives the re-ranker training lists from the *same distribution*
as the test lists.

Users need at least 4 interactions for this split (the 5-core guarantees 5).

## 4. Item text

Each item becomes one line of text, used by the text encoder (m03) and later shown to the LLM:

```
Title: Xtreme Brite Brightening Gel 1oz. | Brand: Xtreme Brite | Category: Beauty > Hair Care > Styling Products
```

Rules (tested): use the **first** category path. Omit the Brand / Category parts when they're empty.
A missing title becomes `Unknown product`. Unescape HTML and collapse whitespace.

---

## Implementation order

1. `download.py`: `download_file` (download to a `.part` file, then rename. Why? If the download dies
   halfway, you don't get a corrupt file that a later run "skips" because it exists).
2. `parse.py`: `iter_gzip_lines`, `read_reviews`, `read_meta`.
3. `sequences.py`: `k_core_filter`, `subsample_users`, `build_sequences`, `remap_ids`, `make_splits`, `dataset_stats`.
4. `item_text.py`: `clean_text`, `build_item_record`, `item_text`.
5. `storage.py`: `save_jsonl`, `load_jsonl`, `save_processed`, `load_processed`.
6. `run.py`: the pipeline.

Performance hint for `read_meta`: `ast.literal_eval` on 259k lines takes about a minute. If `keep` is
given, pull the asin out with a quick regex first (`'asin': '(\w+)'`) and skip lines you don't need.

## Test & try

```powershell
python -m pytest gr2mini/m02_data -v
python -m gr2mini.m02_data.run --config configs/tiny.yaml
python -m gr2mini.m02_data.run --config configs/beauty.yaml
```

Expected for `beauty.yaml` (compare with the paper's Table 1):

```
users 22363 | items 12101 | interactions 198502 | avg_seq_len 8.876 | density 0.00073
items without a title: 7
example user ...:
  1. Title: ... | Brand: ... | Category: Beauty > Hair Care > ...
  ...
```
(198502 / 22363 = 8.876; the paper truncates it to 8.87.) For `tiny.yaml`, expect 2000 users,
~7,400 items and ~17,500 interactions. The whole run takes ~10–20 s.
