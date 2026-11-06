"""The DAPO training loop.  (README "One RL step")"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
from torch import nn  # noqa: F401

from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m07_llm.logprobs import token_logprobs  # noqa: F401
from gr2mini.m09_prompts.prompts import build_rerank_messages  # noqa: F401
from gr2mini.m11_rewards.rewards import RewardInfo, compute_reward  # noqa: F401
from gr2mini.m14_dapo.algo import completion_mask, dapo_loss, group_advantages, informative_groups  # noqa: F401
from gr2mini.m14_dapo.rollout import rollout_group, sequence_batch  # noqa: F401


@dataclass
class Group:
    example: dict
    prompt_ids: torch.Tensor  # [Tp]
    completion_ids: torch.Tensor  # [G, Tc]
    mask: torch.Tensor  # [G, Tc] bool (completion_mask)
    texts: list[str]
    rewards: torch.Tensor  # [G] float32
    infos: list[RewardInfo] = field(default_factory=list)


def disable_dropout(model: nn.Module) -> int:
    """Set p = 0.0 on every nn.Dropout inside the model (LoRA dropout included). Returns how many."""
    raise NotImplementedError("TODO(m14): implement disable_dropout")


def collect_groups(
    model,
    tokenizer,
    examples: Sequence[dict],
    items: Sequence[dict],
    index: SIDIndex,
    rl_cfg: dict,
    prompt_cfg: dict,
    rng: random.Random,
    stop_ids: Sequence[int],
    rollout_fn: Callable = rollout_group,
) -> tuple[list[Group], dict]:
    """Draw examples with rng.choice(examples) until rl_cfg["prompts_per_step"] informative groups are kept,
    or prompts_per_step + rl_cfg["max_resample"] draws have been made.

    For each draw: out = rollout_fn(model, tokenizer, policy_messages, rl_cfg["group_size"],
                                    rl_cfg["max_new_tokens"], rl_cfg["temperature"])
                   infos = [compute_reward(t, ex, rl_cfg["format_alpha"], rl_cfg["conditional_format"])]
                   keep if informative_groups(rewards[None])[0]
    Returns (kept groups, stats) with stats over ALL drawn samples:
        {"drawn": #draws, "kept": #kept groups, "reward_mean", "parse_rate", "identity_rate"}
    """
    raise NotImplementedError("TODO(m14): implement collect_groups")


def dapo_update(model, optimizer, groups: Sequence[Group], rl_cfg: dict, pad_id: int, scaler=None) -> dict:
    """One DAPO update on the kept groups (README). Returns {"loss", "clip_frac", "ratio_mean"} (means).

    1. advantages: group_advantages(stack of group rewards [P, G])
    2. old_logp per group with torch.no_grad() (token_logprobs on sequence_batch tensors, moved to the
       model's device)
    3. num_tokens = total number of True mask entries over all groups
    4. for each of rl_cfg["ppo_epochs"]: zero_grad; for each group, in chunks of
       rl_cfg.get("micro_batch_size", 1) sequences: logp = token_logprobs(...);
       loss, st = dapo_loss(logp, old, adv_chunk, target_mask_chunk, eps_low, eps_high, num_tokens);
       backward (through scaler.scale if a scaler is given); then clip grads at 1.0 and step
       (scaler.step / scaler.update if given). Use fp16 autocast only if a scaler is given.
    The model must be in train mode during the updates.
    """
    raise NotImplementedError("TODO(m14): implement dapo_update")


def train_dapo(
    model,
    tokenizer,
    examples: Sequence[dict],
    items: Sequence[dict],
    index: SIDIndex,
    rl_cfg: dict,
    prompt_cfg: dict,
    seed: int = 0,
    eval_fn: Callable | None = None,
    log: Callable[[str], None] | None = print,
) -> list[dict]:
    """The full loop. Returns one history entry per step.

    * disable_dropout(model); model.train(); AdamW(trainable params, lr=rl_cfg["lr"]);
      GradScaler("cuda") only if the model is on CUDA AND holds fp16 weights, else None (fp32: the
      default on your GPU); rng = random.Random(seed)
    * stop_ids = {id of "<|im_end|>", tokenizer.eos_token_id, tokenizer.pad_token_id} (ignore None)
    * for step in 1..rl_cfg["steps"]:
          groups, st = collect_groups(...)
          upd = dapo_update(...) if groups else {"loss": 0.0, "clip_frac": 0.0, "ratio_mean": 1.0}
          entry = {"step": step, **st, **upd}
          if eval_fn and step % rl_cfg["eval_every"] == 0: entry.update(eval_fn(model))
          append; log a one-liner
    """
    raise NotImplementedError("TODO(m14): implement train_dapo")
