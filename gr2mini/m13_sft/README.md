# m13 — Stage 2: supervised fine-tuning on reasoning traces

**Paper:** §4.2, Eq. 13, Table 5 "SFT-*" rows.
**Time:** ~half a day of coding + ~30 min of training. **Files you implement:** `features.py`, `evaluate.py`, `run.py`.

## The loss

```
L_SFT = − λ_r · Σ_i log P(r_i | prompt, r_<i)          reasoning tokens (the "explanation" value)
        − λ_o · Σ_j log P(o_j | prompt, τ, o_<j)       everything else in the answer: JSON syntax,
        with λ_r < λ_o                                   "recommendations", <|im_end|>          (13)
```

**Why two weights?** The ranking is what gets evaluated. The reasoning is a means to an end, and it
comes from a teacher that may be imperfect. Down-weighting it (λ_r = 0.5, λ_o = 1.0) keeps the student
from over-fitting the teacher's prose. Normalization is the same as in m08: Σ w·nll / #(tokens with w > 0).
You reuse m08's `train_lm` unchanged. Only the weights differ.

## Character spans → token weights (`features.py`)

```
full = chat_prompt(messages, add_generation_prompt=False)
       = prompt  +  '{"explanation": "The user bought ... ", "recommendations": ["3", ...]}'  +  '<|im_end|>\n'
                                     └──── explanation span ────┘
tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)  →  (start, end) chars per token

weight(token) = 0      if end <= len(prompt)                        (prompt)
                λ_r    if the token overlaps the explanation span    (start < span_end and end > span_start)
                λ_o    otherwise
```

`explanation_char_span(content)` finds the explanation **value** inside the assistant JSON: the
characters of the JSON string literal *including its quotes*. The content is produced by
`assistant_json` (json.dumps, `ensure_ascii=False`), so `json.dumps(explanation, ensure_ascii=False)`
appears verbatim after `"explanation": `. Tokens can straddle a boundary (e.g. `."` or `",`). They overlap
the span, so they get λ_r. Off-by-one bugs here are classic, and the tests decode the λ_r tokens to check.

## Evaluation (`evaluate.py`), reused by m14 and m15

```
evaluate_reranker(examples, items, index, prompt_cfg, generate_fn)
    prompts = policy messages for every example   →   generate_fn(prompts) → texts (one call)
    for each: parse (m10) → reordered = apply_ranking(...)  (pre-rank fallback)
              reward (m11, conditional, α = 0.1) → identity flag
    → ranking_report(reordered lists, targets) + {"parse_rate", "identity_rate", "mean_reward", "n"}
```

Taking a `generate_fn` instead of a model keeps this testable with a fake, and lets m14 reuse it with
any decoding settings. `pre_rank_report(examples)` gives the baseline row (lists as retrieved).

## What to expect: the paper's surprise

The paper's Table 5: **SFT-targeted-KP R@1 = 0.2006 vs pre-rank 0.2892.** SFT alone can make
the ranking *worse*. Imitating explanations doesn't teach the model to rank, and a 0.6B student
trained on a few hundred traces may also simply copy the input order (check `identity_rate`!).
Measure it yourself. Stage 3 exists for this reason.

## Budget (measured on your GPU, fp32)

Trace conversations are ~1,000 tokens (prompt ~860 + answer). Training: ~2–3 s per sequence, so
300 traces × 3 epochs ≈ 30–45 min. **Evaluation is the slow part:** greedy decoding of ~300–500 answer
tokens at ~10 steps/s (batch 4) ≈ 10–15 s per example. That's why `eval.max_examples` is 300 for Beauty
(~1 h per checkpoint) and 40 for `tiny`.

## Implementation order & test

`explanation_char_span` → `sft_feature` → `pre_rank_report` → `evaluate_reranker` → `run.py`

```powershell
python -m pytest gr2mini/m13_sft -v
python -m gr2mini.m13_sft.run --config configs/tiny.yaml --traces rejection   # → <artifacts>/sft_rejection_lora/
```
