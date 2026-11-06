# m05 — Semantic IDs: codes → unique SIDs → LLM tokens

**Paper:** §2.1 (Eq. 1), §2.3 "random last levels" (Eq. 10), Appendix A (SID ⇄ item mapping),
Appendix B (token format), Appendix G (Eqs. 33–35).
**Time:** ~half a day. **Files you implement:** `sid_format.py`, `assign.py`, `sid_index.py`, `run.py`.

## Where we are

m04 gave us a trained RQ-VAE. Running every item through it deterministically gives a code tuple per
item. This module turns those codes into the **final, unique Semantic IDs** and the token strings the
LLM will see.

```
item 1234 ─► RQ-VAE.get_codes ─► (12, 3, 250, 91) ─► random last level ─► (12, 3, 250, 7)
                                                                              │
                         "<|sid_begin|><s_a_12><s_b_3><s_c_250><s_d_7><|sid_end|>"
```

---

## 1. Why collisions happen

Items with nearly identical text ("Shampoo 8 oz" vs "Shampoo 16 oz") get nearly identical latents,
and therefore identical codes at *every* level. The RQ-VAE has no reason to separate them. With the
co-engagement loss it's even worse, because frequently co-bought items are pulled together on purpose.

## 2. Random last levels (Eq. 10)

```
c_l = argmin_j ‖r^(l) − e_j^(l)‖²    for l ≤ L − M     (semantic levels, as before)
c_l ~ Uniform{0, …, C−1}             for l > L − M     (random levels)
```

With L = 4 and M = 1: three semantic levels, plus one random "disambiguation" level.

**Why it works (birthday math).** Take a group of `g` items that share the same first L−M codes. With
a random last code among C = 256, the expected number of colliding pairs in that group is
`g(g−1)/2 · 1/C`. For g = 5 that's 10/256 ≈ 0.04. Collisions become rare, though not zero, which is
why the paper reports 99.98 % uniqueness rather than 100 %.

**What we give up:** the last level carries no meaning any more. It's a pure ID suffix, much like
TIGER's "append a collision counter" trick. The semantic part is the prefix, and that's what the LLM
generalizes over.

## 3. Collision resolution (our addition)

An LLM re-ranker needs a strict one-to-one map, SID ⇄ item: when the model writes a SID, we must know
which item it means. So after the random draw, `resolve_collisions` repeats until every SID is unique:

```
while there are collision groups:
    for each group (item ids sorted): keep the first item, re-draw the last M levels of the others
    (give up with RuntimeError after max_rounds)
```

Items outside collision groups must keep their codes unchanged. The tests check this.

## 4. Reading the numbers (Table 2)

`run.py` prints three uniqueness values, each worth understanding:

| Number | Meaning | Paper analogue |
|---|---|---|
| `uniqueness_semantic` | deterministic codes at all L levels | Table 2 rows without "random last level" |
| `uniqueness_random_last` | after Eq. 10, before resolution | Table 2 rows with it (best: 99.98 %) |
| `uniqueness_final` | after resolution | always 1.0 by construction |

Also look at **`prefix_stats`**: how many distinct level-1 prefixes there are, and how many items share
the biggest one. If one level-1 code holds 40 % of all items, level 1 isn't splitting the catalogue
well. That's a sign of partial collapse, even if final uniqueness looks fine.

## 5. Token format

Following the paper's Appendix B (and OneRec-Think):

* `<|sid_begin|>` and `<|sid_end|>` delimit a SID.
* Level `l` code `c` is the token `<s_{letter}_{c}>`, with letters a, b, c, d… for levels 0, 1, 2, 3…
* `all_sid_tokens(L, C)` lists all `2 + L·C` new tokens. m07 adds them to the Qwen tokenizer as
  **single tokens**, so `<s_a_12>` is ONE token (one embedding row), not `<`, `s`, `_`, ...

**Parsing is the inverse** and has to be *robust*: LLM outputs contain partial SIDs, wrong-level tokens
(`<s_a_1><s_c_5>`), or SIDs glued to text. `parse_sid` returns the first *well-formed* SID: levels
appear in order a, b, c, ... starting from a. Malformed blocks are skipped.

## 6. `SIDIndex`: the dictionary every later module uses

```python
index = SIDIndex.load("artifacts/sids.json")
index.sid(1234)                 # (12, 3, 250, 7)
index.sid_str(1234)             # "<|sid_begin|><s_a_12>...<|sid_end|>"
index.item((12, 3, 250, 7))     # 1234
index.item_from_text("I recommend <|sid_begin|><s_a_12><s_b_3><s_c_250><s_d_7><|sid_end|>!")  # 1234
index.items_with_prefix((12, 3))  # [items sharing the first two codes]: semantic siblings
```

---

## Implementation order

1. `sid_format.py`: `code_token`, `sid_to_str`, `parse_sid`, `find_all_sids`, `all_sid_tokens`.
2. `assign.py`: `compute_semantic_codes`, `randomize_last_levels`, `find_collisions`,
   `resolve_collisions`, `codebook_utilization`, `prefix_stats`, `assign_semantic_ids`.
3. `sid_index.py`: `SIDIndex`.
4. `run.py`: the live example.

## Test & try

```powershell
python -m pytest gr2mini/m05_semantic_ids -v
python -m gr2mini.m05_semantic_ids.run --config configs/tiny.yaml
```

Live example output on full Beauty, from my reference run (your exact numbers will differ a bit):

```
uniqueness_semantic     0.968    (deterministic 4-level codes)
uniqueness_random_last  1.000    (Eq. 10, M=1)   ← tiny: ~0.994, then ~20 items get re-drawn
uniqueness_final        1.000
utilization per level   [1.00, 0.84, 0.75, 0.67]
prefix level 1: 256 distinct, largest group 146, mean 47.3 items
prefix level 2: 7453 distinct, largest group 24,  mean 1.6 items

item 4223  <|sid_begin|><s_a_61><s_b_186><s_c_162><s_d_246><|sid_end|>
  Title: Physicians Formula Powder Palette Color Corrective Powders ... | Category: Beauty > Makeup > Face
  siblings sharing (61, 186):
    Title: Physicians Formula Organic Wear 100% Natural Loose Powder, Translucent Light ...
    Title: Physicians Formula Mineral Wear Talc-free Mineral Face Powder, Beige ...
  siblings sharing (61,):  (46 more)
    ...
```

Notice the prefix statistics. Level 1 splits 12k items into groups of ~47, and level 2 already
splits them into groups of ~1.6. So many items have **no** level-2 sibling, and level 1 is where
most of the shared "meaning" lives.

**Sanity check that the SIDs are semantic:** siblings should look like the query item (same product
type, often the same brand). If they look random, revisit m04: try more epochs, and check that k-means
init ran and that level-1 usage is high.

## End of Phase B 🎉

You now have the paper's complete item tokenizer. Commit, and tell me when you're ready for Phase C
(the SASRec retriever).
