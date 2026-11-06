import json

import pytest

from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m07_llm.generate import chat_prompt
from gr2mini.m07_llm.sid_tokens import add_sid_tokens
from gr2mini.m07_llm.testing import qwen_tokenizer, tiny_qwen
from gr2mini.m09_prompts.prompts import assistant_json, build_rerank_messages
from gr2mini.m13_sft.evaluate import evaluate_reranker, pre_rank_report
from gr2mini.m13_sft.features import explanation_char_span, sft_feature

ITEMS = [{"item_id": i, "title": f"Product {i}", "categories": ["Beauty", "Hair"]} for i in range(30)]
INDEX = SIDIndex([(i, 0, 0, 1) for i in range(30)], codebook_size=32)
PCFG = {"max_history": 5, "max_title_words": 10, "max_category_levels": 2, "knowledge_priming": False}


def ex(target_rank=3, user=0):
    cands = list(range(10, 20))
    return {"user": user, "history": [1, 2], "candidates": cands, "target": cands[target_rank - 1], "target_rank": target_rank}


EXPL = f'The user likes "hair" products like {INDEX.sid_str(2)}, so candidate 3 fits best.'


def trace_messages():
    return build_rerank_messages(ex(), ITEMS, INDEX, "policy", PCFG) + [
        {"role": "assistant", "content": assistant_json(EXPL, [3, 1, 2, 4, 5, 6, 7, 8, 9, 10])}]


# ======================================================================= spans
def test_explanation_char_span():
    content = assistant_json("a \"quoted\" é", [2, 1])
    s, e = explanation_char_span(content)
    assert json.loads(content[s:e]) == "a \"quoted\" é"
    assert content[s - 2:s] == ": " and content[e:e + 2] == ", "
    with pytest.raises(ValueError):
        explanation_char_span("not json")


# ======================================================================= features
def _tok():
    tok = qwen_tokenizer()
    add_sid_tokens(tok, tiny_qwen(), 4, 32)
    return tok


def test_sft_feature_weights():
    tok = _tok()
    msgs = trace_messages()
    f = sft_feature(tok, msgs, lambda_r=0.5, lambda_o=1.0, max_len=4096)
    ids, w = f["input_ids"], f["loss_weights"]
    assert len(ids) == len(w)
    n_prompt = len(tok.encode(chat_prompt(tok, msgs[:-1]), add_special_tokens=False))
    assert all(x == 0.0 for x in w[:n_prompt]) and all(x > 0 for x in w[n_prompt:])
    assert set(w[n_prompt:]) == {0.5, 1.0}
    reasoning = tok.decode([t for t, x in zip(ids, w) if x == 0.5])
    output = tok.decode([t for t, x in zip(ids, w) if x == 1.0])
    assert "The user likes" in reasoning and "fits best" in reasoning and INDEX.sid_str(2) in reasoning
    assert "recommendations" in output and '"3"' in output and "<|im_end|>" in output
    assert "fits best" not in output and "recommendations" not in reasoning
    assert output.lstrip().startswith('{"explanation'), "the JSON key before the value is an OUTPUT token"


def test_sft_feature_too_long_and_lambda_values():
    tok = _tok()
    assert sft_feature(tok, trace_messages(), 0.5, 1.0, max_len=50) is None
    f = sft_feature(tok, trace_messages(), 0.2, 0.9, max_len=4096)
    assert set(f["loss_weights"]) == {0.0, 0.2, 0.9}


# ======================================================================= evaluation
def _ans(order):
    return json.dumps({"explanation": "x", "recommendations": [str(k) for k in order]})


def test_pre_rank_report():
    exs = [ex(1), ex(3)]
    rep = pre_rank_report(exs)
    assert rep["R@1"] == pytest.approx(0.5) and rep["R@5"] == pytest.approx(1.0)


def test_evaluate_reranker_with_fake_generator():
    exs = [ex(3, user=0), ex(2, user=1), ex(5, user=2), ex(1, user=3)]
    outs = [_ans([3]), "garbage", _ans(range(1, 11)), _ans([2, 1])]
    calls = []

    def fake(convs):
        calls.append(convs)
        return outs

    rep = evaluate_reranker(exs, ITEMS, INDEX, PCFG, fake, return_outputs=True)
    assert len(calls) == 1 and calls[0][0] == build_rerank_messages(exs[0], ITEMS, INDEX, "policy", PCFG)
    # new target ranks: 1, 2 (unparsed -> pre-rank), 5 (identity), 2
    assert rep["R@1"] == pytest.approx(0.25) and rep["R@5"] == pytest.approx(1.0)
    assert rep["parse_rate"] == pytest.approx(0.75) and rep["identity_rate"] == pytest.approx(0.25)
    # last one: demoted from #1 -> R_rank = -0.1, but r^D == 1 still earns the format bonus (Eq. 15)
    assert rep["mean_reward"] == pytest.approx((0.3 + 0.0 + 0.0 + (-0.1 + 0.1)) / 4)
    assert rep["n"] == 4 and rep["outputs"] == outs
    assert "outputs" not in evaluate_reranker(exs, ITEMS, INDEX, PCFG, fake)
