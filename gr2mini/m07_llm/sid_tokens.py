"""Adding the Semantic-ID tokens to the tokenizer and the embedding table.  (README §1)"""
from __future__ import annotations

import torch  # noqa: F401

from gr2mini.m05_semantic_ids.sid_format import all_sid_tokens  # noqa: F401


def add_sid_tokens(tokenizer, model, num_levels: int, codebook_size: int) -> list[int]:
    """Add all 2 + L*C SID tokens; make sure each has its own, sensibly initialized embedding row.

    1. If any SID token is already in the vocabulary -> ValueError (don't add twice).
    2. n_before = len(tokenizer); tokenizer.add_tokens(all_sid_tokens(num_levels, codebook_size))
    3. If len(tokenizer) > number of embedding rows:
           model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
       (otherwise leave the size alone: the new ids fit into existing padding rows)
    4. Under torch.no_grad(): set the input-embedding row of EVERY new id to the mean of rows
       0 .. n_before-1 (compute the mean in float32, cast back to the embedding dtype).
    Returns the new token ids, in all_sid_tokens order.
    """
    raise NotImplementedError("TODO(m07): implement add_sid_tokens")


def check_sid_tokenization(tokenizer, num_levels: int, codebook_size: int) -> bool:
    """True iff every SID token encodes to exactly ONE token id
    (tokenizer.encode(tok, add_special_tokens=False))."""
    raise NotImplementedError("TODO(m07): implement check_sid_tokenization")
