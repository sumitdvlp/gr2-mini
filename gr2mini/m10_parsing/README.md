# m10 — Robust output parsing  🟡 preview

**Paper:** Appendix E ("Robust Re-ranking Output Parsing").
**Status:** lesson preview. Stubs and tests come when you get here. Pure Python, a good break from tensors.

## What you'll build

`parse_rerank_output(text, num_candidates) -> (reasoning | None, ranking | None)`, the three-stage
algorithm from Appendix E:

```
Stage I   JSON:   extract all JSON-like substrings, try them in REVERSE order (the last one is usually
                  the final answer); read "explanation"/"reasoning" and "recommendations"/"ranking"
Stage II  regex:  if no JSON worked, search ranking-like patterns ("ranking: 3, 1, 7 ...") and pull
                  out the numbers
Stage III validate: keep ints in [1, |D|] → drop duplicates (keep first) → none left? (None, None)
                    → convert to 0-based → append the missing candidates in pre-rank order
                    → a full permutation of length |D|
```

## Why it matters more than it looks

Every metric and every RL reward goes through this function. Small models produce truncated JSON,
markdown code fences, `"recommendations": [3, "1", "7"]` with mixed types, or the list twice. A parser
that's too strict reads as "the model is bad", and one that's too lenient lets garbage earn reward. The
tests will contain ~25 real-world failure cases.
