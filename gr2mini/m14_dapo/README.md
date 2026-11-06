# m14 — Stage 3: reinforcement learning with DAPO

**Paper:** §4.3, Eqs. 16–17. Background: GRPO (DeepSeekMath, 2024), DAPO (Yu et al., 2025).
**Time:** ~1–2 days of coding + 2–3 h of training. **Files you implement:** `algo.py`, `rollout.py`,
`trainer.py`, `run.py`. Written **from scratch** (no TRL), because this is the most educational part of the paper.

## One RL step

```
repeat until prompts_per_step INFORMATIVE groups (or the resample budget runs out):      collect_groups
    q = policy prompt of a random rerank_train example
    o_1..o_G ~ π_old(·|q)                     sample G answers (temperature 1.0)              rollout.py
    R_i = compute_reward(o_i)                 m10 parse → m11 reward (conditional format)
    keep the group only if not all R_i are equal                       dynamic sampling      algo.py
A_i = (R_i − mean(R)) / (std(R) + ε)          group-relative advantage, same for every token of o_i  (17)
old_logp = log π_old(o_{i,t}) for every completion token   (no grad, before any update)
for ppo_epochs:
    ratio_{i,t} = exp(logp − old_logp)
    J = 1/Σ_i|o_i| · Σ_i Σ_t min( ratio·A_i, clip(ratio, 1−ε_low, 1+ε_high)·A_i )         (16)
    minimize −J   (no KL penalty, no value network, no reference model)
```

## The DAPO tricks vs. plain GRPO

| Trick | Problem it solves |
|---|---|
| **Clip-higher** (ε_high = 0.28 > ε_low = 0.2) | symmetric clipping stops low-probability tokens from growing → entropy collapse, no exploration |
| **Token-level normalization** (divide by the total number of tokens in the batch) | per-sequence means make the tokens of long answers count less |
| **Dynamic sampling** | groups where all G answers get the same reward have zero advantage: no signal, wasted compute |
| **No KL term** | no reference model in memory, and the policy may move further from the start |

## Details that bite

* **Dropout must be off.** With dropout, π_old and π_θ are different random functions even before any
  update, so the "ratio" is noise. `train_dapo` sets `p = 0` on every `nn.Dropout` (LoRA dropout included).
* **Completion masks.** `generate` pads finished sequences. A completion's tokens are those up to *and
  including* the first stop token (`<|im_end|>` or the pad/eos token). Learning to stop matters too.
  Truncated answers (no stop token within `max_new_tokens`) keep all their tokens. They're usually
  unparseable, so they get R = 0.
* **Token normalization under gradient accumulation.** One prompt's G sequences of ~1,400 tokens don't
  fit in one backward pass on 6 GB, so we go `micro_batch_size` sequences at a time. Each micro-batch's
  loss must be divided by the **total** completion-token count of the whole step, not its own. So the
  total is computed *before* the first backward. `dapo_loss(..., num_tokens=total)` makes this explicit.
  The tests check that two half-batches give the same gradient as one full batch.
* **Sign:** we *maximize* J, so the loss is −J.

## What to log (and what it means)

| metric | healthy | alarm |
|---|---|---|
| `reward_mean` | slowly rising | flat at 0: no informative groups |
| `parse_rate` | → ~1.0 | falling: the policy is drifting off format |
| **`identity_rate`** | low | rising toward 1: **reward hacking** (copying the input), see m11 |
| `kept_groups / drawn` | 0.3–0.8 | ≈ 0: all-equal rewards, so raise temperature or G |
| `clip_frac` | small (< 0.2) | large: lr too high |
| valid `R@1` every `eval_every` | ≥ pre-rank | below the SFT start |

**The reward-hacking ablation:** `--set rl.conditional_format=false` and compare `identity_rate`.

## Budget on 6 GB

Each prompt: G = 4 samples × up to 384 new tokens at ~8–10 decoding steps/s ≈ 40 s, plus ~10 s of
training. A step with 2 kept prompts (+ rejected draws) takes ~1.5–2 min, so 100 steps ≈ 3 h. `tiny`: 6 steps.

## Implementation order & test

`algo.py` (`group_advantages`, `informative_groups`, `completion_mask`, `dapo_loss`) → `rollout.py`
(`rollout_group`, `sequence_batch`) → `trainer.py` (`disable_dropout`, `collect_groups`, `dapo_update`,
`train_dapo`) → `run.py`

```powershell
python -m pytest gr2mini/m14_dapo -v
python -m gr2mini.m14_dapo.run --config configs/tiny.yaml --init sft_rejection_lora   # RL-rejection
python -m gr2mini.m14_dapo.run --config configs/tiny.yaml --init midtrain_lora        # RL-zeroshot
```
