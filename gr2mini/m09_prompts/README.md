# m09 — Re-ranking prompts

**Paper:** §3.1 (system / user / assistant roles), §3.2 (prompt principles), Appendices C–D.
**Time:** ~2–3 hours. **File you implement:** `prompts.py` (pure Python). The prompt *texts*
(`SYSTEM_PROMPT`, `KNOWLEDGE_PRIMING`, `TARGET_HINT`) are provided. Read them: they're the paper's
five principles made concrete.

## The conversation

```
system     SYSTEM_PROMPT  [+ KNOWLEDGE_PRIMING if prompts.knowledge_priming]      ← "KP" in Tables 5-6
user       Purchase history (oldest first, at most max_history):
             1. <SID> | title (≤ max_title_words words) | last max_category_levels category levels
             ...
           Candidates:
             Candidate 1: <SID> | title | categories
             ...
             Candidate 10: ...
           [TARGET_HINT with the true candidate number, ONLY in the "targeted" variant]
assistant  {"explanation": "...", "recommendations": ["3", "1", "7", ...]}     ← assistant_json
```

| Variant | Used by | Contains the answer? |
|---|---|---|
| `"targeted"` | teacher, Eq. 11 | **yes**: `TARGET_HINT` names the true candidate |
| `"rejection"` | teacher, Eq. 12 | no |
| `"policy"` | the student: SFT, RL, evaluation | no (exactly the same text as `"rejection"`) |

## The five principles of §3.2, and where each lives

1. **Concrete role and task:** the first paragraph of `SYSTEM_PROMPT`.
2. **Collaborative context:** history and *all* candidates in one prompt, so the model compares
   them instead of scoring each in isolation.
3. **Domain-knowledge priming:** `KNOWLEDGE_PRIMING` (shampoo → conditioner → styling...).
4. **Critical guidelines:** cite items by SID, don't talk about a "target", and rank all 10.
5. **Structured output:** the JSON format, last in the system prompt.

## Leakage is the bug to fear

If the policy prompt leaks the answer in *any* way, everything downstream is meaningless and the
metrics look fantastic. Leaks to watch for: the target's position, a "target" marker, or candidates
re-sorted by something that correlates with the answer. The tests build prompts for the same lists
with *different* targets and require the policy text to be byte-identical.

## Category paths

Beauty paths are long and start with the useless root:
`Beauty > Hair Care > Styling Products > Hair Extensions & Wigs > Wigs`. We keep the **last**
`max_category_levels` levels (the specific ones): `Styling Products > Hair Extensions & Wigs > Wigs`.

## Implementation order & test

`truncate_words` → `item_line` → `build_rerank_messages` → `assistant_json` → `candidate_sids`

```powershell
python -m pytest gr2mini/m09_prompts -v
python -m gr2mini.m09_prompts.run --config configs/tiny.yaml   # prints a real prompt + its token count
```

Expect ~800–1,200 tokens per prompt with the Qwen tokenizer. If you're far above that, check the truncation.
