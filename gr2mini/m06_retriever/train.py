"""SASRec training with full-softmax next-item prediction and early stopping.  (README §2)"""
from __future__ import annotations

from typing import Callable, Sequence

import torch

from gr2mini.m02_data.sequences import SeqExample
from gr2mini.m06_retriever.model import SASRec, encode_histories  # noqa: F401


def make_training_batch(seqs: Sequence[Sequence[int]], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and targets for next-item prediction at every position.

    For each sequence s (len >= 2): keep s[-(max_len + 1):], inputs = s[:-1], targets = s[1:].
        inputs  -> encode_histories (shift +1, left-pad with 0)      LongTensor [B, max_len]
        targets -> plain item ids, LEFT-padded with -100             LongTensor [B, max_len]
    So targets[b, t] is the item that follows the input at inputs[b, t].
    Sequences with fewer than 2 items: ValueError.
    """
    raise NotImplementedError("TODO(m06): implement make_training_batch")


def sasrec_loss(model: SASRec, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over all items, at every position whose target != -100."""
    raise NotImplementedError("TODO(m06): implement sasrec_loss")


@torch.no_grad()
def evaluate_full_ranking(
    model: SASRec,
    examples: Sequence[SeqExample],
    ks: Sequence[int] = (10,),
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> dict[str, float]:
    """Full-ranking metrics over ALL items (history items excluded), e.g. {"R@10": .., "N@10": ..}.

    Use candidates.topk_candidates with k = max(ks), then m01 recall_at_k / ndcg_at_k for every k.
    Keys: "R@k" for every k, then "N@k" for every k. Restores the model's train/eval mode afterwards.
    """
    raise NotImplementedError("TODO(m06): implement evaluate_full_ranking")


def train_sasrec(
    model: SASRec,
    train_seqs: Sequence[Sequence[int]],
    valid_examples: Sequence[SeqExample],
    cfg: dict,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    log: Callable[[str], None] | None = print,
) -> list[dict]:
    """Train with Adam(lr), early stopping on valid N@10. Returns the history.

    cfg = the `retriever:` section (epochs, batch_size, lr, eval_every, patience).
        * Drop train sequences with < 2 items.
        * Each epoch: shuffle with torch.randperm(n, generator=generator), batches of batch_size,
          make_training_batch, sasrec_loss, step. model.train() during training.
        * Every eval_every epochs (and the last epoch): evaluate_full_ranking(model, valid, ks=(10,))
          and append {"epoch", "loss" (mean over the epoch), "R@10", "N@10"} to the history.
        * Track the best N@10 and a deep copy of its state_dict. Stop early after `patience`
          evaluations without improvement.
        * At the end: load the best state_dict and put the model in eval mode.
    """
    raise NotImplementedError("TODO(m06): implement train_sasrec")
