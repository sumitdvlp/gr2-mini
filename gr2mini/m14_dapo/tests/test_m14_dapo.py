import json
import random

import pytest
import torch

from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m07_llm.lora import apply_lora
from gr2mini.m07_llm.sid_tokens import add_sid_tokens
from gr2mini.m07_llm.testing import TINY_LORA_CFG, qwen_tokenizer, tiny_qwen
from gr2mini.m09_prompts.prompts import build_rerank_messages
from gr2mini.m14_dapo.algo import completion_mask, dapo_loss, group_advantages, informative_groups
from gr2mini.m14_dapo.rollout import rollout_group, sequence_batch
from gr2mini.m14_dapo.trainer import Group, collect_groups, dapo_update, disable_dropout

# ======================================================================= advantages & sampling
def test_group_advantages():
    r = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.3, 0.3, 0.3, 0.3]])
    a = group_advantages(r)
    std = torch.tensor([1.0, 0.0, 0.0, 0.0]).std(unbiased=False)
    torch.testing.assert_close(a[0], (r[0] - 0.25) / (std + 1e-6))
    torch.testing.assert_close(a[1], torch.zeros(4))
    torch.testing.assert_close(a.mean(1), torch.zeros(2), atol=1e-6, rtol=0)


def test_informative_groups():
    r = torch.tensor([[0.1, 0.1, 0.1], [0.0, 0.2, 0.0], [-0.1, -0.1, -0.1]])
    assert informative_groups(r).tolist() == [False, True, False]


def test_completion_mask():
    ids = torch.tensor([[5, 6, 99, 0, 0], [5, 99, 7, 99, 0], [1, 2, 3, 4, 5], [98, 0, 0, 0, 0]])
    m = completion_mask(ids, stop_ids={99, 98})
    assert m.tolist() == [
        [True, True, True, False, False],
        [True, True, False, False, False],
        [True, True, True, True, True],  # truncated: no stop token
        [True, False, False, False, False],
    ]


# ======================================================================= the loss
def test_dapo_loss_value_and_stats():
    logp = torch.log(torch.tensor([[0.5, 0.5], [0.2, 0.9]]))
    old = torch.log(torch.tensor([[0.25, 0.5], [0.4, 0.9]]))
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[True, True], [True, False]])
    loss, st = dapo_loss(logp, old, adv, mask, eps_low=0.2, eps_high=0.28)
    # ratios: 2.0 (A=+1 -> clipped to 1.28), 1.0, 0.5 (A=-1 -> min(-0.5, -0.8) = -0.8)
    expected = -(1.28 + 1.0 - 0.8) / 3
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    assert st["clip_frac"] == pytest.approx(2 / 3)
    assert st["ratio_mean"] == pytest.approx((2.0 + 1.0 + 0.5) / 3, rel=1e-5)


def test_dapo_loss_gradients_respect_clipping():
    old = torch.zeros(1, 4)
    # ratio: e^0.5=1.65 (A>0: clipped -> no grad), e^-0.5=0.61 (A>0: unclipped side -> grad),
    logp = torch.tensor([[0.5, -0.5, 0.5, -0.5]], requires_grad=True)
    adv = torch.tensor([1.0])
    loss, _ = dapo_loss(logp, old, adv, torch.ones(1, 4, dtype=torch.bool))
    loss.backward()
    g = logp.grad[0]
    assert g[0] == 0 and g[2] == 0, "ratio above 1+eps_high with A>0 is clipped: no gradient"
    assert g[1] < 0 and g[3] < 0, "minimizing -J pushes up the log-prob of a positive-advantage token"
    logp2 = torch.tensor([[0.5, -0.5]], requires_grad=True)
    loss2, _ = dapo_loss(logp2, torch.zeros(1, 2), torch.tensor([-1.0]), torch.ones(1, 2, dtype=torch.bool))
    loss2.backward()
    assert logp2.grad[0, 0] > 0 and logp2.grad[0, 1] == 0, "A<0: ratio below 1-eps_low is clipped"


def test_clip_higher_is_asymmetric():
    adv = torch.tensor([1.0])
    m = torch.ones(1, 1, dtype=torch.bool)
    up = dapo_loss(torch.log(torch.tensor([[1.25]])), torch.zeros(1, 1), adv, m)[0]
    assert up.item() == pytest.approx(-1.25), "1.25 < 1 + eps_high = 1.28: not clipped (plain PPO's 1.2 would clip)"


def test_token_level_normalization_across_micro_batches():
    torch.manual_seed(0)
    logp = torch.randn(4, 6, requires_grad=True)
    old = logp.detach() + 0.1 * torch.randn(4, 6)
    adv = torch.randn(4)
    mask = torch.rand(4, 6) > 0.3
    full, _ = dapo_loss(logp, old, adv, mask)
    full.backward()
    g_full = logp.grad.clone()
    logp.grad = None
    total = int(mask.sum())
    for sl in (slice(0, 1), slice(1, 4)):
        part, _ = dapo_loss(logp[sl], old[sl], adv[sl], mask[sl], num_tokens=total)
        part.backward()
    torch.testing.assert_close(logp.grad, g_full)


# ======================================================================= rollout tensors
def test_sequence_batch():
    p = torch.tensor([7, 8, 9])
    c = torch.tensor([[1, 2, 99], [3, 99, 0]])
    m = completion_mask(c, {99})
    b = sequence_batch(p, c, m, pad_id=0)
    assert b["input_ids"].tolist() == [[7, 8, 9, 1, 2, 99], [7, 8, 9, 3, 99, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]
    assert b["target_mask"].tolist() == [[False, False, False, True, True, True], [False, False, False, True, True, False]]


ITEMS = [{"item_id": i, "title": f"Product {i}", "categories": ["Beauty", "Hair"]} for i in range(30)]
INDEX = SIDIndex([(i % 8, i // 8, 0, 1) for i in range(30)], codebook_size=8)
PCFG = {"max_history": 3, "max_title_words": 4, "max_category_levels": 1, "knowledge_priming": False}
EX = {"user": 0, "history": [1, 2], "candidates": list(range(10, 20)), "target": 13, "target_rank": 4}


def _student():
    tok, model = qwen_tokenizer(), tiny_qwen()
    ids = add_sid_tokens(tok, model, 4, 8)
    return tok, apply_lora(model, {**TINY_LORA_CFG, "lora_dropout": 0.1}, ids)


def test_rollout_group_shapes_and_mode_restore():
    tok, model = _student()
    model.train()
    msgs = build_rerank_messages(EX, ITEMS, INDEX, "policy", PCFG)
    out = rollout_group(model, tok, msgs, group_size=3, max_new_tokens=6)
    assert out["prompt_ids"].dim() == 1 and out["completion_ids"].shape[0] == 3
    assert out["completion_ids"].shape[1] <= 6 and len(out["texts"]) == 3
    assert model.training


def test_disable_dropout():
    tok, model = _student()
    assert disable_dropout(model) > 0
    assert all(m.p == 0.0 for m in model.modules() if isinstance(m, torch.nn.Dropout))


# ======================================================================= collect + update
def _ans(order):
    return json.dumps({"explanation": "x", "recommendations": [str(k) for k in order]})


def test_collect_groups_dynamic_sampling_with_fake_rollouts():
    scripts = [
        [_ans(range(1, 11))] * 2,  # all identity -> rewards equal -> dropped
        [_ans([4]), "junk"],  # 0.4 vs 0.0 -> kept
        [_ans([4]), _ans([1, 2, 3, 5, 4])],  # 0.4 vs -0.1 -> kept
    ]

    def fake_rollout(model, tok, msgs, G, max_new, temp):
        texts = scripts.pop(0)
        return {"prompt_ids": torch.tensor([1, 2]), "completion_ids": torch.tensor([[5, 99], [6, 99]]), "texts": texts}

    cfg = {"prompts_per_step": 2, "max_resample": 3, "group_size": 2, "max_new_tokens": 4, "temperature": 1.0,
           "format_alpha": 0.1, "conditional_format": True}
    groups, st = collect_groups(None, None, [EX], ITEMS, INDEX, cfg, PCFG, random.Random(0), [99], rollout_fn=fake_rollout)
    assert len(groups) == 2 and st["drawn"] == 3 and st["kept"] == 2
    assert isinstance(groups[0], Group) and groups[0].mask.dtype == torch.bool
    torch.testing.assert_close(groups[0].rewards, torch.tensor([0.4, 0.0]))
    assert st["parse_rate"] == pytest.approx(5 / 6) and st["identity_rate"] == pytest.approx(2 / 6)
    assert st["reward_mean"] == pytest.approx((0 + 0 + 0.4 + 0 + 0.4 - 0.1) / 6)


def test_collect_groups_respects_resample_budget():
    def fake_rollout(*a):
        return {"prompt_ids": torch.tensor([1]), "completion_ids": torch.tensor([[2], [2]]), "texts": ["x", "y"]}

    cfg = {"prompts_per_step": 2, "max_resample": 1, "group_size": 2, "max_new_tokens": 4, "temperature": 1.0,
           "format_alpha": 0.1, "conditional_format": True}
    groups, st = collect_groups(None, None, [EX], ITEMS, INDEX, cfg, PCFG, random.Random(0), [99], rollout_fn=fake_rollout)
    assert groups == [] and st["drawn"] == 3


def test_dapo_update_changes_trainable_params_only():
    tok, model = _student()
    disable_dropout(model)
    model.train()
    p = torch.tensor([100, 200, 300])
    c = torch.tensor([[400, 500, 151645], [600, 151645, 151643]])
    m = completion_mask(c, {151645, 151643})
    g = Group(EX, p, c, m, ["a", "b"], torch.tensor([1.0, 0.0]))
    trainable = [q for q in model.parameters() if q.requires_grad]
    before = [q.detach().clone() for q in trainable]
    base_w = model.get_base_model().model.layers[0].mlp.up_proj.base_layer.weight.detach().clone()
    opt = torch.optim.AdamW(trainable, lr=0.01)
    st = dapo_update(model, opt, [g, g], {"ppo_epochs": 2, "eps_low": 0.2, "eps_high": 0.28, "micro_batch_size": 1},
                     pad_id=tok.pad_token_id)
    assert set(st) >= {"loss", "clip_frac", "ratio_mean"}
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, trainable))
    assert torch.equal(base_w, model.get_base_model().model.layers[0].mlp.up_proj.base_layer.weight)
