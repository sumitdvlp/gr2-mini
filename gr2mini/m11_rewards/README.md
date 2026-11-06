# m11 — Rewards: rank improvement + conditional format reward

**Paper:** §4.3, Eqs. 14–15.
**Time:** ~1–2 hours. **File you implement:** `rewards.py` (pure Python, uses m10).

## The reward

```
R_rank = ( r^D − r^o ) / |D|                                   (14)
         r^D = 1-based rank of the true item in the PRE-ranked list   (example["target_rank"])
         r^o = its rank after the model re-ranked
         > 0 if the model moved it up, < 0 if it moved it down

R = R_rank + α · R_fmt     if R_rank > 0  or  r^D == 1        (15)
R = R_rank                 otherwise
         R_fmt = 1 if the output parsed (m10 returned a ranking), else 0
```

Worked examples (|D| = 10, α = 0.1):

| pre-rank r^D | new rank r^o | parsed | R_rank | conditional R | unconditional R |
|---|---|---|---|---|---|
| 4 | 1 | yes | 0.3 | **0.4** | 0.4 |
| 4 | 4 | yes (copied input) | 0.0 | **0.0** | 0.1 ← free lunch |
| 4 | 6 | yes | −0.2 | −0.2 | −0.1 |
| 1 | 1 | yes | 0.0 | **0.1** (kept a correct #1) | 0.1 |
| 1 | 2 | yes | −0.1 | **0.0** (Eq. 15 literally: r^D = 1 earns the bonus regardless) | 0.0 |
| 4 | — | **no** → pre-rank order | 0.0 | 0.0 | 0.0 |

## Why "conditional"? Reward hacking.

With an *unconditional* format bonus, copying the input order earns +α every time, risk-free, while a
real attempt to re-rank risks a negative R_rank. The policy learns to output the original order with a
plausible explanation. It looks great on "% valid format" and changes nothing. Making the bonus
conditional on actual improvement (or on keeping a true #1 at #1) removes the free lunch.

`RewardInfo.identity` flags outputs that parse to exactly the input order. m14 logs the fraction as a
**reward-hacking alarm**. Run the ablation (`--set rl.conditional_format=false`) and watch it climb.

## Unparseable outputs (our choice; the paper doesn't say)

Fall back to the pre-rank order: r^o = r^D, so R_rank = 0, with R_fmt = 0. Evaluation uses the same
fallback, so reward and metric agree.

## Test

```powershell
python -m pytest gr2mini/m11_rewards -v
```
