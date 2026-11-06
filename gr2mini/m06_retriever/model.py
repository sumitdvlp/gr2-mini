"""SASRec: a causal transformer over item-id sequences.  (README §1)

Id convention INSIDE this module: model token = item id + 1, and 0 is padding.
Everything that leaves the module (scores, candidates) uses plain item ids 0..N-1 again.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def encode_histories(histories: Sequence[Sequence[int]], max_len: int) -> torch.Tensor:
    """Item-id histories -> LongTensor [B, max_len] of model tokens.

    * keep only the LAST max_len items of each history
    * shift ids by +1 (0 is reserved for padding)
    * LEFT-pad with 0, so the most recent item is always at position max_len - 1

    encode_histories([[3, 4, 5]], 5) -> [[0, 0, 4, 5, 6]];  an empty history -> all zeros.
    """
    raise NotImplementedError("TODO(m06): implement encode_histories")


def attention_mask(tokens: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Bool mask [B * num_heads, T, T] for nn.TransformerEncoderLayer, True = NOT allowed.

    masked[b, i, j] = (j > i)                          causal
                      or (tokens[b, j] == 0 and j != i)   pad keys (a pad query may still see itself)
    Repeated for each head, batch-major (rows b*num_heads ... b*num_heads + num_heads - 1 are sequence b).
    Same device as `tokens`.
    """
    raise NotImplementedError("TODO(m06): implement attention_mask")


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        max_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ):
        """Build:
            self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
            self.pos_emb  = nn.Embedding(max_len, hidden_dim)
            self.dropout  = nn.Dropout(dropout)
            self.blocks   = nn.ModuleList of num_blocks nn.TransformerEncoderLayer(
                                hidden_dim, num_heads, dim_feedforward=4 * hidden_dim, dropout=dropout,
                                batch_first=True, norm_first=True)
            self.final_norm = nn.LayerNorm(hidden_dim)
        INIT (this matters a lot, see README §1): re-initialize item_emb and pos_emb with
        nn.init.normal_(std=0.02) and zero the padding row item_emb.weight[0] again afterwards.
        Also store num_items, hidden_dim, max_len, num_heads, and self.init_kwargs (all constructor args)
        so a checkpoint can be rebuilt.
        """
        super().__init__()
        raise NotImplementedError("TODO(m06): implement SASRec.__init__")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, T] (T <= max_len, from encode_histories) -> hidden states [B, T, H] (README §1).

        x = item_emb(tokens) * sqrt(H) + pos_emb(arange(T)); dropout; every block with
        src_mask=attention_mask(tokens, num_heads); final_norm.
        """
        raise NotImplementedError("TODO(m06): implement SASRec.forward")

    def score_all(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [..., H] -> scores [..., num_items] against item_emb.weight[1:] (item id order)."""
        raise NotImplementedError("TODO(m06): implement SASRec.score_all")

    def user_scores(self, tokens: torch.Tensor) -> torch.Tensor:
        """Scores [B, num_items] for the next item, from the LAST position's hidden state."""
        raise NotImplementedError("TODO(m06): implement SASRec.user_scores")
