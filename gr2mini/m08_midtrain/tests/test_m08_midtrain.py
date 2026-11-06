import math

import pytest
import torch

from gr2mini.m02_data.sequences import SeqExample
from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m07_llm.generate import chat_prompt
from gr2mini.m07_llm.lora import apply_lora
from gr2mini.m07_llm.sid_tokens import add_sid_tokens
from gr2mini.m07_llm.testing import TINY_LORA_CFG, qwen_tokenizer, tiny_qwen
from gr2mini.m08_midtrain.features import collate, tokenize_chat
from gr2mini.m08_midtrain.probe import probe_caption, probe_next_item, word_f1
from gr2mini.m08_midtrain.tasks import (
    build_midtrain_examples,
    caption_example,
    reverse_example,
    sequential_example,
    sequential_prompt,
)
from gr2mini.m08_midtrain.trainer import lr_lambda, train_lm, weighted_lm_loss

ITEMS = [{"item_id": i, "title": f"Product {i}", "text": f"Title: Product {i} | Category: Beauty > Hair"} for i in range(8)]
INDEX = SIDIndex([(i % 4, i // 4, 0, i) for i in range(8)], codebook_size=8)
SEQS = [[0, 1, 2, 3, 4, 5], [6, 7, 0, 1, 2], [3, 4, 5, 6, 7, 0, 1]]
CFG = {"caption": True, "reverse": True, "sequential_per_user": 2, "max_history": 3, "max_examples": None}


# ======================================================================= tasks
def test_single_examples():
    sid = INDEX.sid_str(3)
    cap = caption_example(ITEMS[3], sid)
    assert [m["role"] for m in cap] == ["user", "assistant"]
    assert sid in cap[0]["content"] and cap[1]["content"] == ITEMS[3]["text"]
    rev = reverse_example(ITEMS[3], sid)
    assert ITEMS[3]["text"] in rev[0]["content"] and rev[1]["content"] == sid
    hist = [INDEX.sid_str(i) for i in (1, 2)]
    seq = sequential_example(hist, INDEX.sid_str(5))
    assert seq[0]["content"] == sequential_prompt(hist) and seq[1]["content"] == INDEX.sid_str(5)
    p = sequential_prompt(hist)
    assert p.index(hist[0]) < p.index(hist[1]) and "1." in p and "2." in p


def test_build_midtrain_examples_counts_and_no_leakage():
    exs = build_midtrain_examples(ITEMS, INDEX, SEQS, CFG, seed=0)
    assert len(exs) == 8 + 8 + 2 * 3
    seq_only = {**CFG, "caption": False, "reverse": False, "sequential_per_user": 10}
    for s in SEQS:  # (in these toy sequences, the valid/test items never occur earlier in seq[:-2])
        targets = {e[1]["content"] for e in build_midtrain_examples(ITEMS, INDEX, [s], seq_only)}
        assert targets == {INDEX.sid_str(x) for x in s[1:-2]}
        assert INDEX.sid_str(s[-1]) not in targets and INDEX.sid_str(s[-2]) not in targets, "leakage!"


def test_build_midtrain_examples_uses_latest_positions_and_max_history():
    cfg = {**CFG, "caption": False, "reverse": False, "sequential_per_user": 1}
    exs = build_midtrain_examples(ITEMS, INDEX, [[0, 1, 2, 3, 4, 5]], cfg)
    # s = seq[:-2] = [0,1,2,3]; last position t=3 -> history [0,1,2], target 3
    assert exs == [sequential_example([INDEX.sid_str(i) for i in (0, 1, 2)], INDEX.sid_str(3))]
    cfg2 = {**cfg, "max_history": 2}
    assert build_midtrain_examples(ITEMS, INDEX, [[0, 1, 2, 3, 4, 5]], cfg2)[0][0]["content"] == sequential_prompt(
        [INDEX.sid_str(1), INDEX.sid_str(2)])


def test_build_midtrain_examples_shuffle_cap_and_flags():
    a = build_midtrain_examples(ITEMS, INDEX, SEQS, CFG, seed=1)
    b = build_midtrain_examples(ITEMS, INDEX, SEQS, CFG, seed=1)
    c = build_midtrain_examples(ITEMS, INDEX, SEQS, CFG, seed=2)
    assert a == b and a != c
    assert len(build_midtrain_examples(ITEMS, INDEX, SEQS, {**CFG, "max_examples": 5})) == 5
    only_cap = build_midtrain_examples(ITEMS, INDEX, SEQS, {**CFG, "reverse": False, "sequential_per_user": 0})
    assert len(only_cap) == 8 and all(e[1]["content"].startswith("Title:") for e in only_cap)


# ======================================================================= features
def _tok_with_sids():
    tok, model = qwen_tokenizer(), tiny_qwen()
    ids = add_sid_tokens(tok, model, 4, 8)
    return tok, model, ids


def test_tokenize_chat_masks_prompt_and_learns_answer_and_stop():
    tok, _, _ = _tok_with_sids()
    msgs = sequential_example([INDEX.sid_str(1)], INDEX.sid_str(2))
    f = tokenize_chat(tok, msgs, max_len=512)
    ids, w = f["input_ids"], f["loss_weights"]
    assert len(ids) == len(w)
    n_prompt = len(tok.encode(chat_prompt(tok, msgs[:-1]), add_special_tokens=False))
    assert all(x == 0.0 for x in w[:n_prompt]) and all(x == 1.0 for x in w[n_prompt:])
    answer = tok.decode([t for t, x in zip(ids, w) if x > 0])
    assert answer.startswith(INDEX.sid_str(2)) and "<|im_end|>" in answer
    assert "<think>" not in answer, "the empty think block belongs to the prompt"


def test_tokenize_chat_drops_too_long():
    tok, _, _ = _tok_with_sids()
    assert tokenize_chat(tok, caption_example(ITEMS[0], INDEX.sid_str(0)), max_len=10) is None


def test_collate_right_pads():
    feats = [{"input_ids": [5, 6, 7], "loss_weights": [0.0, 1.0, 1.0]}, {"input_ids": [8], "loss_weights": [0.5]}]
    b = collate(feats, pad_id=0)
    assert b["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["loss_weights"].tolist() == [[0.0, 1.0, 1.0], [0.5, 0.0, 0.0]]
    assert b["input_ids"].dtype == torch.long and b["loss_weights"].dtype == torch.float32


# ======================================================================= trainer
def test_weighted_lm_loss_is_weighted_mean_nll():
    tok, model, _ = _tok_with_sids()
    feats = [{"input_ids": [10, 11, 12, 13], "loss_weights": [0.0, 0.0, 1.0, 0.5]}]
    batch = collate(feats, tok.pad_token_id)
    logits = model(input_ids=batch["input_ids"]).logits.float()
    lp = torch.log_softmax(logits, -1)
    expected = -(1.0 * lp[0, 1, 12] + 0.5 * lp[0, 2, 13]) / 2
    torch.testing.assert_close(weighted_lm_loss(model, batch), expected, atol=1e-5, rtol=1e-5)


def test_lr_lambda():
    assert lr_lambda(0, 10, 100) == pytest.approx(0.1)
    assert lr_lambda(9, 10, 100) == pytest.approx(1.0)
    assert lr_lambda(10, 10, 100) == pytest.approx(1.0)
    assert lr_lambda(55, 10, 100) == pytest.approx(0.5)
    assert lr_lambda(100, 10, 100) == 0.0 and lr_lambda(150, 10, 100) == 0.0
    assert lr_lambda(0, 0, 10) == pytest.approx(1.0)


def test_train_lm_learns_and_counts_steps():
    tok, model, ids = _tok_with_sids()
    pm = apply_lora(model, TINY_LORA_CFG, ids)
    msgs = caption_example(ITEMS[0], INDEX.sid_str(0))
    feats = [tokenize_chat(tok, msgs, 512)] * 10
    frozen_before = pm.get_base_model().model.layers[0].mlp.up_proj.base_layer.weight.detach().clone()
    cfg = {"epochs": 3, "batch_size": 4, "grad_accum": 2, "lr": 0.01, "warmup_steps": 0, "log_every": 100}
    hist = train_lm(pm, feats, cfg, tok.pad_token_id, generator=torch.Generator().manual_seed(0), log=None)
    assert len(hist) == 3 * math.ceil(math.ceil(10 / 4) / 2) == 6
    assert [h["step"] for h in hist] == list(range(1, 7)) and {"step", "loss", "lr"} <= set(hist[0])
    assert hist[-1]["loss"] < hist[0]["loss"]
    after = pm.get_base_model().model.layers[0].mlp.up_proj.base_layer.weight
    assert torch.equal(after, frozen_before), "frozen base weights must not move"
    assert pm.training


# ======================================================================= probes
def test_word_f1():
    assert word_f1("Argan Oil Shampoo", "argan oil shampoo 8oz") == pytest.approx(2 * 0.75 / 1.75)
    assert word_f1("", "x") == 0.0 and word_f1("a b", "c") == 0.0
    assert word_f1("the the cat", "the cat cat") == pytest.approx(2 / 3)


def test_probe_next_item_with_fake_generator():
    calls = []

    def fake(convs):
        calls.append(convs)
        return [INDEX.sid_str(3), "no idea", f"maybe {INDEX.sid_str(7)}"]

    exs = [SeqExample([0, 1, 2], 3), SeqExample([4], 5), SeqExample([5, 6], 6)]
    rep = probe_next_item(fake, INDEX, exs, max_history=2)
    assert len(calls) == 1 and len(calls[0]) == 3
    assert calls[0][0][-1]["content"] == sequential_prompt([INDEX.sid_str(1), INDEX.sid_str(2)])
    assert rep == {"hit@1": pytest.approx(1 / 3), "valid_sid_rate": pytest.approx(2 / 3)}


def test_probe_caption_with_fake_generator():
    rep = probe_caption(lambda convs: ["Product 2", "Product nine"], ITEMS, INDEX, [2, 4])
    assert rep["title_f1"] == pytest.approx((1.0 + 0.5) / 2)
