# m11 — Rewards: rank improvement + conditional format reward  🟡 preview

**Paper:** §4.3, Eqs. 14–15.
**Status:** lesson preview. Stubs and tests come when you get here. Pure Python.

## The reward

```
R_rank = ( r^D − r^o ) / |D|                                   (14)
         r^D = 1-based rank of the true item in the PRE-ranked list
         r^o = its rank after the model re-ranked
         > 0 if the model moved it up, < 0 if it moved it down

R = R_rank + α · R_fmt     if R_rank > 0  or  r^D == 1        (15)
R = R_rank                 otherwise
         R_fmt = 1 if the output parsed (m10), else 0
```

## Why "conditional"? Reward hacking.

Suppose the format bonus were *unconditional*. Copying the input order gives R_rank = 0 and +α for
valid format, every time, risk-free. Any real attempt to re-rank risks a negative R_rank. So the policy
learns to output the original order with a plausible explanation. **That's reward hacking.** It looks
fine on the format metric and changes nothing. Making the bonus conditional on actual improvement (or on
correctly keeping a true #1 at #1) removes the free lunch.

You'll implement both versions and run a short RL experiment in m14 to *see* the hacking happen:
watch the fraction of identity rankings go up.

## Our choice for unparseable outputs

The paper doesn't say. We fall back to the pre-rank order (r^o = r^D, so R_rank = 0) with R_fmt = 0.
This is the same fallback the evaluation uses, so reward and metric agree.
