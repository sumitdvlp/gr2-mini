"""Memory-efficient token log-probabilities.  (README §3)  Used by SFT (m08/m13) and RL (m14)."""
from __future__ import annotations

import torch


def get_backbone_and_head(model):
    """(backbone, head) for a HF causal LM or a PEFT-wrapped one.

    base = model.get_base_model() if the model has that method, else model
    return base.model, base.lm_head
    """
    raise NotImplementedError("TODO(m07): implement get_backbone_and_head")


def token_logprobs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    """log p(input_ids[b, t] | input_ids[b, :t]) at every position where target_mask[b, t] is True.

    Returns a float32 tensor [B, T]: the log-prob at target positions, 0.0 everywhere else.
    target_mask[:, 0] is ignored (nothing predicts the first token).

    Run the backbone once, then apply the head ONLY to the hidden states at positions t-1 of the
    targets: never materialize [B, T, vocab] logits. Compute log_softmax in float32.
    Must be differentiable (it's the loss for SFT and RL).
    """
    raise NotImplementedError("TODO(m07): implement token_logprobs")
