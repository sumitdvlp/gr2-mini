# m10 — Robust output parsing

**Paper:** Appendix E ("Robust Re-ranking Output Parsing").
**Time:** ~3–4 hours. **File you implement:** `parsing.py` (pure Python).

## Why it matters more than it looks

Every metric and every RL reward goes through this function. Small models produce truncated JSON,
markdown fences, mixed types, duplicate numbers and stray `<think>` blocks. Too strict, and a good
model looks bad. Too lenient, and garbage earns reward in RL. The tests encode ~20 real failure modes.

## The algorithm (Appendix E)

```
parse_rerank_output(text, num_candidates) -> ParseResult(reasoning, ranking, stage)

0. text = strip_think(text)                       remove <think>...</think> (and an unclosed <think>... tail)
Stage I   JSON   for each substring in reversed(json_candidates(text)):       (the LAST JSON is the final answer)
                    try json.loads; must be a dict with a non-empty list under "recommendations" or "ranking"
                    reasoning = "explanation" or "reasoning" (if a string)
                    → validate_ranking(list); success → stage "json"
Stage II  regex  if Stage I found nothing: the LAST occurrence of "recommendations" / "ranking"
                 (case-insensitive) → the text after it up to the first "]" or newline → all integers
                    → validate_ranking; success → stage "regex"
                    (reasoning: the value of a "explanation": "..." if one is visible, else None)
Stage III validate_ranking(values, n):
                 values → ints (int as is; str → its first integer; anything else skipped)
                 keep 1..n → drop duplicates (keep first) → empty? None
                 → 0-based → append the missing candidates in ascending order → a permutation of 0..n-1
nothing worked → ParseResult(None, None, "none")
```

`json_candidates` is the fun part. Find every **balanced** `{...}` at the top level by scanning
characters and counting braces, and **ignore braces inside JSON strings**. That means tracking whether
you're inside `"..."` and handling `\"` escapes. A regex can't do this, because braces nest.

`apply_ranking(candidates, ranking)` reorders the item ids. `None` → the original (pre-rank) order,
which is the fallback used by both evaluation and reward (top-level README §8.9).

## Test

```powershell
python -m pytest gr2mini/m10_parsing -v
```
