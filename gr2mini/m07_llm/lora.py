"""LoRA + trainable SID-token rows.  (README §2)"""
from __future__ import annotations

from typing import Sequence


def apply_lora(model, llm_cfg: dict, trainable_token_ids: Sequence[int]):
    """Wrap `model` with PEFT and return the PeftModel.

    peft.LoraConfig(
        r=llm_cfg["lora_r"], lora_alpha=llm_cfg["lora_alpha"], lora_dropout=llm_cfg["lora_dropout"],
        target_modules=llm_cfg["lora_targets"], task_type="CAUSAL_LM",
        trainable_token_indices={"embed_tokens": list(trainable_token_ids)})
    peft.get_peft_model(model, config)

    If llm_cfg["gradient_checkpointing"]: call
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    on the PeftModel, and set model.config.use_cache = False (the KV cache is useless when training).
    """
    raise NotImplementedError("TODO(m07): implement apply_lora")


def trainable_summary(model) -> dict:
    """{"trainable": #params with requires_grad, "total": #all params, "fraction": trainable / total}"""
    raise NotImplementedError("TODO(m07): implement trainable_summary")
