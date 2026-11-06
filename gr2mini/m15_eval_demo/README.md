# m15 — Evaluation tables + live re-ranking demo  🟡 preview

**Paper:** §5.2–5.4 (Tables 5–6), Appendix F (case study), Appendix G (metrics).
**Status:** lesson preview. Stubs and tests come when you get here.

## Evaluation

For every model checkpoint, on the kept test examples: greedy-decode → parse (m10, falling back to the
pre-rank order on failure) → `ranking_report` (m01). This gives a table shaped like the paper's Table 5:

| Method | R@1 | R@5 | R@9 | N@5 | N@10 | % parsed |
|---|---|---|---|---|---|---|
| Pre-rank (SASRec) | | | | | | — |
| SFT-targeted | | | | | | |
| SFT-rejection | | | | | | |
| RL-zeroshot (RL from the mid-trained model, no SFT) | | | | | | |
| RL-targeted | | | | | | |
| RL-rejection | | | | | | |

Questions we'll answer with *your* numbers: does SFT hurt R@1? Does RL recover it? Do rejection traces
beat targeted traces? How large are the gains relative to run-to-run noise? (The paper reports no seeds
or variance. We'll run at least 2 seeds for the best row.)

## The live demo

```powershell
python -m gr2mini.m15_eval_demo.demo --config configs/beauty.yaml --user 1234
```

```
HISTORY      1. Shampoo …  2. Conditioner …  3. Heat-protect spray …
PRE-RANKED   1. Hair dryer  2. Leave-in cream  3. ...   (true next item at position 2)
REASONING    "The user bought a shampoo + conditioner set, then a heat protectant, …"
RE-RANKED    1. Leave-in cream ✅  2. Hair dryer  …        (true item moved 2 → 1, reward +0.10)
```

Interactive mode: pick a random test user, or type your own history of SIDs/titles and watch the model
reason.
