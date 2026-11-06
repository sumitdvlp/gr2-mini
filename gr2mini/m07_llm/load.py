"""Loading the student, end to end.  (integration-tested on your GPU)"""
from __future__ import annotations

from pathlib import Path

import torch  # noqa: F401


def load_student(
    llm_cfg: dict,
    num_levels: int,
    codebook_size: int,
    device: str | torch.device = "cuda",
    adapter_dir: str | Path | None = None,
    trainable: bool = True,
):
    """Return (model, tokenizer, sid_token_ids).

    1. tokenizer = AutoTokenizer.from_pretrained(llm_cfg["model_name"])
    2. model = AutoModelForCausalLM.from_pretrained(name, dtype=pick_dtype(device, llm_cfg["dtype"]),
                                                    attn_implementation="sdpa")
    3. ids = add_sid_tokens(tokenizer, model, num_levels, codebook_size)
       (deterministic: the same ids every time, so saved adapters stay compatible)
    4. adapter_dir is None -> model = apply_lora(model, llm_cfg, ids)
       else -> model = peft.PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
               (+ the same gradient checkpointing setup as apply_lora if trainable)
    5. model.to(device); return model, tokenizer, ids

    Saving is just model.save_pretrained(dir): PEFT writes the LoRA weights AND the SID-row deltas.
    """
    raise NotImplementedError("TODO(m07): implement load_student")
