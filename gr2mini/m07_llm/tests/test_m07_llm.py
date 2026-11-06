import pytest
import torch

from gr2mini.m05_semantic_ids.sid_format import all_sid_tokens, sid_to_str
from gr2mini.m07_llm.generate import chat_prompt, generate_chat
from gr2mini.m07_llm.logprobs import get_backbone_and_head, token_logprobs
from gr2mini.m07_llm.lora import apply_lora, trainable_summary
from gr2mini.m07_llm.sid_tokens import add_sid_tokens, check_sid_tokenization
from gr2mini.m07_llm.testing import REAL_VOCAB_ROWS, TINY_LORA_CFG, qwen_tokenizer, tiny_qwen

L, C = 4, 8  # 34 new tokens: few enough to all land in Qwen's 267 padding rows
L_BIG, C_BIG = 4, 256  # 1,026 new tokens: the real setting, needs a resize


# ======================================================================= add_sid_tokens
def test_add_sid_tokens_real_size_resizes_and_mean_inits():
    tok, model = qwen_tokenizer(), tiny_qwen()
    n_before = len(tok)
    old = model.get_input_embeddings().weight[:n_before].detach().clone()
    ids = add_sid_tokens(tok, model, L_BIG, C_BIG)
    assert len(ids) == 2 + L_BIG * C_BIG
    assert ids == list(range(n_before, n_before + len(ids)))
    assert ids == tok.convert_tokens_to_ids(all_sid_tokens(L_BIG, C_BIG))
    W = model.get_input_embeddings().weight
    assert W.shape[0] >= len(tok) > REAL_VOCAB_ROWS
    mean = old.float().mean(0)
    torch.testing.assert_close(W[ids].detach().float(), mean.expand(len(ids), -1))
    torch.testing.assert_close(W[:n_before].detach(), old, msg="original rows must not change")
    # tied: the output head sees the same rows
    assert model.get_output_embeddings().weight.data_ptr() == W.data_ptr()


def test_add_sid_tokens_reuses_padding_rows_without_resizing():
    tok, model = qwen_tokenizer(), tiny_qwen()
    junk = model.get_input_embeddings().weight[len(tok):len(tok) + 34].detach().clone()
    ids = add_sid_tokens(tok, model, L, C)
    W = model.get_input_embeddings().weight
    assert W.shape[0] == REAL_VOCAB_ROWS, "34 tokens fit into the 267 padding rows: no resize"
    assert max(ids) < REAL_VOCAB_ROWS
    assert not torch.allclose(W[ids].detach(), junk), "padding rows hold junk: they must be re-initialized"
    torch.testing.assert_close(W[ids[0]].detach(), W[ids[-1]].detach())


def test_add_sid_tokens_twice_raises():
    tok, model = qwen_tokenizer(), tiny_qwen()
    add_sid_tokens(tok, model, L, C)
    with pytest.raises(ValueError):
        add_sid_tokens(tok, model, L, C)


def test_check_sid_tokenization():
    tok = qwen_tokenizer()
    assert check_sid_tokenization(tok, L, C) is False, "before adding, '<s_a_0>' splits into many pieces"
    add_sid_tokens(tok, tiny_qwen(), L, C)
    assert check_sid_tokenization(tok, L, C) is True
    ids = tok.encode(sid_to_str((1, 2, 3, 4)), add_special_tokens=False)
    assert len(ids) == L + 2


# ======================================================================= logprobs
def _batch(tok):
    texts = ["Hello there, how are you?", "Short."]
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    target = enc.attention_mask.bool().clone()
    target[:, :2] = False
    return enc.input_ids, enc.attention_mask, target


def _naive_logprobs(model, ids, mask, target):
    logits = model(input_ids=ids, attention_mask=mask).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    out = torch.zeros(ids.shape, dtype=torch.float32)
    out[:, 1:] = lp
    return out * target


def test_token_logprobs_matches_naive_full_logits():
    tok, model = qwen_tokenizer(), tiny_qwen()
    ids, mask, target = _batch(tok)
    lp = token_logprobs(model, ids, mask, target)
    assert lp.shape == ids.shape and lp.dtype == torch.float32
    torch.testing.assert_close(lp, _naive_logprobs(model, ids, mask, target), atol=1e-4, rtol=1e-4)
    assert (lp[~target] == 0).all() and (lp[target] < 0).all()


def test_token_logprobs_ignores_position_zero_and_is_differentiable():
    tok, model = qwen_tokenizer(), tiny_qwen()
    ids, mask, _ = _batch(tok)
    target = mask.bool()  # includes position 0
    lp = token_logprobs(model, ids, mask, target)
    assert (lp[:, 0] == 0).all()
    lp.sum().backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_backbone_and_head_unwrap_peft():
    tok, model = qwen_tokenizer(), tiny_qwen()
    ids = add_sid_tokens(tok, model, L, C)
    backbone, head = get_backbone_and_head(model)
    assert backbone is model.model and head is model.lm_head
    pm = apply_lora(model, TINY_LORA_CFG, ids)
    b2, _ = get_backbone_and_head(pm)
    assert b2 is pm.get_base_model().model


# ======================================================================= LoRA + trainable rows
def test_apply_lora_trains_only_adapters_and_new_rows():
    tok, model = qwen_tokenizer(), tiny_qwen()
    ids = add_sid_tokens(tok, model, L, C)
    pm = apply_lora(model, TINY_LORA_CFG, ids)
    summ = trainable_summary(pm)
    assert set(summ) == {"trainable", "total", "fraction"}
    assert 0 < summ["fraction"] < 0.05, "the full embedding matrix must NOT be trainable"
    names = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert any("lora_A" in n for n in names) and any("trainable_tokens" in n for n in names)

    # one optimizer step on a loss over SID tokens
    seq = torch.tensor([[100, 200] + ids[:6]])
    target = torch.zeros_like(seq, dtype=torch.bool)
    target[:, 2:] = True
    opt = torch.optim.SGD([p for p in pm.parameters() if p.requires_grad], lr=1.0)
    emb = pm.get_input_embeddings()
    probe = torch.tensor([ids[:3] + [100, 200]])
    before_in = emb(probe).detach().clone()
    before_out = pm(input_ids=probe).logits.detach().clone()
    pm.train()
    loss = -token_logprobs(pm, seq, torch.ones_like(seq), target).sum()
    loss.backward()
    opt.step()
    pm.eval()
    after_in = emb(probe).detach()
    assert (after_in[0, :3] - before_in[0, :3]).abs().sum() > 0, "SID rows must train"
    torch.testing.assert_close(after_in[0, 3:], before_in[0, 3:], msg="ordinary token rows must stay frozen")
    after_out = pm(input_ids=probe).logits.detach()
    assert not torch.allclose(after_out, before_out)


def test_apply_lora_gradient_checkpointing_flag():
    tok, model = qwen_tokenizer(), tiny_qwen()
    ids = add_sid_tokens(tok, model, L, C)
    pm = apply_lora(model, {**TINY_LORA_CFG, "gradient_checkpointing": True}, ids)
    assert pm.get_base_model().is_gradient_checkpointing
    assert pm.config.use_cache is False


# ======================================================================= generation
def test_chat_prompt_disables_thinking():
    tok = qwen_tokenizer()
    p = chat_prompt(tok, [{"role": "user", "content": "hi"}])
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    full = chat_prompt(tok, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
                       add_generation_prompt=False)
    assert full.startswith(p) and "ok<|im_end|>" in full


def test_generate_chat_structure_determinism_and_mode_restore():
    tok, model = qwen_tokenizer(), tiny_qwen()
    add_sid_tokens(tok, model, L, C)
    model.train()
    model.config.use_cache = False
    tok.padding_side = "right"
    convs =[[{"role": "user", "content": "Say something."}], [{"role": "user", "content": "A much longer question, please answer."}]]
    out1 = generate_chat(model, tok, convs, max_new_tokens=5, batch_size=2)
    out2 = generate_chat(model, tok, convs, max_new_tokens=5, batch_size=1)
    assert len(out1) == 2 and all(len(o) == 1 and isinstance(o[0], str) for o in out1)
    assert out1 == out2, "greedy decoding must not depend on batching (left padding!)"
    assert model.training and model.config.use_cache is False, "restore mode and use_cache"
    assert tok.padding_side == "right", "restore the tokenizer's padding side"
    samp = generate_chat(model, tok, convs[:1], max_new_tokens=5, do_sample=True, num_return_sequences=3)
    assert len(samp) == 1 and len(samp[0]) == 3


# ======================================================================= the real model
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_load_student_on_gpu():
    from gr2mini.m07_llm.load import load_student

    cfg = {"model_name": "Qwen/Qwen3-0.6B", "dtype": "fp16", **TINY_LORA_CFG, "lora_r": 16, "lora_alpha": 32,
           "gradient_checkpointing": True}
    model, tok, ids = load_student(cfg, 4, 256, device="cuda")
    assert ids[0] == 151669 and ids[-1] == 152694
    assert check_sid_tokenization(tok, 4, 256)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    out = generate_chat(model, tok, [[{"role": "user", "content": "Name one shampoo brand."}]], max_new_tokens=20)
    assert len(out[0][0]) > 0
