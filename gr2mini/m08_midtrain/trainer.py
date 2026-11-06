"""A small, reusable LM fine-tuning loop (m08 mid-training, m13 SFT).  (README "The training loop")"""
from __future__ import annotations

from typing import Callable, Sequence

import torch

from gr2mini.m07_llm.logprobs import token_logprobs  # noqa: F401
from gr2mini.m08_midtrain.features import collate  # noqa: F401


def weighted_lm_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """loss = -sum(loss_weights * logp) / number of positions with loss_weights > 0

    logp = token_logprobs(model, input_ids, attention_mask, target_mask=loss_weights > 0).
    (Position 0 never has a positive weight in practice: it's always the prompt.)
    """
    raise NotImplementedError("TODO(m08): implement weighted_lm_loss")


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Multiplier for LambdaLR: linear warmup 0 -> 1 over warmup_steps (step+1)/warmup_steps,
    then linear decay to 0 at total_steps. Never negative. warmup_steps == 0 -> no warmup."""
    raise NotImplementedError("TODO(m08): implement lr_lambda")


def train_lm(
    model,
    features: Sequence[dict],
    cfg: dict,
    pad_id: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    log: Callable[[str], None] | None = print,
) -> list[dict]:
    """Fine-tune `model` on tokenized features. Returns the history: one entry per optimizer step,
    {"step": 1-based int, "loss": mean micro-batch loss of that step, "lr": current lr}.

    cfg keys: epochs, batch_size, grad_accum, lr, warmup_steps, log_every.
        * model.train(); optimizer = AdamW over parameters with requires_grad only
        * total optimizer steps = epochs * ceil(ceil(N / batch_size) / grad_accum)
          (a trailing partial accumulation still makes a step)
        * each epoch: order = torch.randperm(N, generator=generator); micro-batches of batch_size
        * CUDA: torch.autocast("cuda", dtype=torch.float16) + torch.amp.GradScaler("cuda").
          CPU: plain fp32, no scaler.
        * backward(loss / grad_accum); at each step: (unscale), clip_grad_norm_(trainable, 1.0),
          optimizer.step(), scheduler.step() (LambdaLR with lr_lambda), zero_grad
        * log every `log_every` steps if log is not None
    """
    raise NotImplementedError("TODO(m08): implement train_lm")
