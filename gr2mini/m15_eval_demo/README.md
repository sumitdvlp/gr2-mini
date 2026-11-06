# m15 — Evaluation tables + live re-ranking demo

**Paper:** §5.2–5.4 (Tables 5–6), Appendix F (case study), Appendix G (metrics).
**Time:** ~half a day. **Files you implement:** `report.py`, `run.py` (the table), `demo.py` (the live demo).

## The results table (`run.py`)

For every checkpoint that exists in `<artifacts_dir>`, evaluate on the kept **test** examples with
greedy decoding (m13 `evaluate_reranker`), plus the pre-rank row:

| Method (adapter dir) | paper analogue |
|---|---|
| — (candidate lists as retrieved) | Pre-rank |
| `midtrain_lora` | (no paper row: the untouched Stage-1 model) |
| `sft_targeted_lora`, `sft_rejection_lora` | SFT-targeted / SFT-rejection |
| `rl_from_midtrain_lora` | RL-zeroshot |
| `rl_from_sft_targeted_lora`, `rl_from_sft_rejection_lora` | RL-targeted / RL-rejection |

`markdown_table` renders rows like the paper's, plus `parse` and `identity` columns that the paper
doesn't report but you need to interpret the numbers. Results are written to `<artifacts_dir>/results.md`.

**Questions to answer with *your* numbers:** Does SFT hurt R@1 vs pre-rank? Does RL recover it? Do
rejection traces beat targeted ones? Is any gain bigger than the noise? `bootstrap_ci` gives a 95 %
bootstrap interval for R@1 (resample examples with replacement). The paper reports no variance. With
~1.5k test examples, ±2 points of R@1 is within noise.

## The live demo (`demo.py`)

```powershell
python -m gr2mini.m15_eval_demo.demo --config configs/beauty.yaml --adapter rl_from_sft_rejection_lora --random 3
python -m gr2mini.m15_eval_demo.demo --config configs/beauty.yaml --adapter rl_from_sft_rejection_lora --user 1234
```

`render_case` produces a case-study view like Appendix F:

```
USER 1234
HISTORY (oldest first)
   1. <SID> | Shampoo ... | Hair Care > Shampoos
   ...
PRE-RANKED                                 RE-RANKED
   1. Hair dryer ...                           1. Leave-in cream ...  ◀ bought next
   2. Leave-in cream ...  ◀ bought next        2. Hair dryer ...
   ...
REASONING
   The user bought a shampoo and a conditioner, then ...
RESULT  target moved 2 → 1   reward +0.20 (rank +0.10, format +0.10)
```

## Implementation order & test

`report.py`: `markdown_table`, `bootstrap_ci`, `render_case` → `run.py` → `demo.py`

```powershell
python -m pytest gr2mini/m15_eval_demo -v
python -m gr2mini.m15_eval_demo.run --config configs/tiny.yaml
```
