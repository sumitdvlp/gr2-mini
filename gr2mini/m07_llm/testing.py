"""Test helpers (provided, not an exercise): a tiny random Qwen3 + the real Qwen3 tokenizer.

The tiny model has the REAL vocabulary size (151,936 rows) so it reproduces the padding-row gotcha,
but only 2 layers of width 32, so tests run in seconds on the CPU.
"""
from __future__ import annotations

from functools import lru_cache

import torch

QWEN = "Qwen/Qwen3-0.6B"
REAL_VOCAB_ROWS = 151936
TINY_LORA_CFG = {
    "lora_r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gradient_checkpointing": False,
}


@lru_cache(maxsize=1)
def _tokenizer_files():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN)


def qwen_tokenizer():
    """A FRESH copy of the real Qwen3 tokenizer (each test may add tokens to its own copy)."""
    import copy

    return copy.deepcopy(_tokenizer_files())


def tiny_qwen(vocab_rows: int = REAL_VOCAB_ROWS, seed: int = 0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab_rows, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=4096,
        tie_word_embeddings=True, attn_implementation="sdpa",
    )
    return Qwen3ForCausalLM(cfg).float()
