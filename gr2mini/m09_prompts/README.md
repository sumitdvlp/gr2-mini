# m09 — Re-ranking prompts  🟡 preview

**Paper:** §3.1 (system / user / assistant roles), §3.2 (prompt principles), Appendices C–D (full prompts).
**Status:** lesson preview. Stubs and tests come when you get here.

## What you'll build

Pure-Python builders for the chat messages used by every later stage:

```
system:    expert e-commerce recommender persona + the re-ranking task + output format
           [+ domain-knowledge priming ("KP"), e.g. shampoo → conditioner → styling]
user:      Purchase history (≤ 10 most recent):  <SID> | title | category path
           Candidates (pre-ranked by the retriever):
             Candidate 1: <SID> | title | category path
             ...
             Candidate 10: ...
assistant: {"explanation": "step-by-step reasoning citing SIDs ...",
            "recommendations": ["3", "1", "7", ...]}
```

Three variants:

| Variant | Used by | Contains the answer? |
|---|---|---|
| `targeted` | teacher, Eq. 11 | **yes:** "the user actually bought Candidate k; explain why" (without saying "target") |
| `rejection` | teacher, Eq. 12 | no |
| `policy` | student SFT / RL / eval | no |

## Key ideas you'll learn

* **Prompt principles from §3.2,** each made concrete and testable: role, collaborative context,
  priming, critical guidelines, structured output.
* **Leakage tests:** the `policy` and `rejection` prompts must never contain the target's identity. The
  tests check every rendering.
* **Token budgets:** history and candidates with titles can exceed 1,500 tokens. We'll truncate titles
  and category paths and measure prompt lengths with the real tokenizer.
