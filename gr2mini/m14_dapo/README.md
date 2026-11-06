# m14 — Stage 3: reinforcement learning with DAPO  🟡 preview

**Paper:** §4.3, Eqs. 16–17. Background: GRPO (DeepSeekMath, 2024), DAPO (Yu et al., 2025).
**Status:** lesson preview. Stubs and tests come when you get here. We write it **from scratch**
(no TRL), because this is the most educational part of the paper.

## One RL step

```
for a batch of prompts q (history + 10 candidates):
    sample G answers o_1..o_G ~ π_old(·|q)                    (HF generate, temperature 1.0)
    R_i = reward(o_i)                                          (m10 parse → m11 reward)
    A_i = (R_i − mean(R)) / std(R)       group-relative advantage, same for every token of o_i  (17)
    dynamic sampling: drop groups with std(R) = 0 (all equal → zero advantage → no signal)
    ratio_{i,t} = π_θ(o_{i,t}|q, o_{i,<t}) / π_old(o_{i,t}|q, o_{i,<t})
    J = 1/Σ_i|o_i| · Σ_i Σ_t min( ratio·A, clip(ratio, 1−ε_low, 1+ε_high)·A )              (16)
    maximize J  (no KL penalty, no value network)
```

## The DAPO tricks vs. plain GRPO and why each exists

| Trick | Problem it solves |
|---|---|
| **Clip-higher** (ε_high = 0.28 > ε_low = 0.2) | symmetric clipping stops low-probability tokens from growing → entropy collapse, no exploration |
| **Token-level normalization** (divide by total tokens, not per sequence) | per-sequence means make long answers' tokens count less |
| **Dynamic sampling** | groups where all G answers get the same reward contribute nothing but cost compute |
| **No KL term** | frees memory (no reference model) and lets the policy move further |

## Key ideas you'll learn

* Why the advantage is **group-relative** (no critic network to train or fit in VRAM).
* **Gradient accumulation with token-level normalization:** you must know the total token count of the
  whole batch *before* the first micro-batch backward pass.
* Logging what matters: mean reward, % valid format, **% identity rankings** (the reward-hacking alarm
  from m11), entropy, and R@1 on valid every N steps.

## Budget on 6 GB

G = 4 samples × ~350 new tokens per prompt, LoRA, fp16: ~15–25 s per prompt. Expect ~200–400 prompts in
2–3 h.
