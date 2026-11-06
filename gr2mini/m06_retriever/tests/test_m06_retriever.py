import pytest
import torch

from gr2mini.m02_data.sequences import SeqExample
from gr2mini.m06_retriever.candidates import build_candidate_examples, topk_candidates
from gr2mini.m06_retriever.model import SASRec, attention_mask, encode_histories
from gr2mini.m06_retriever.train import evaluate_full_ranking, make_training_batch, sasrec_loss, train_sasrec


def small_model(num_items=20, dropout=0.0, max_len=8):
    return SASRec(num_items, hidden_dim=16, max_len=max_len, num_blocks=2, num_heads=2, dropout=dropout)


# ======================================================================= encoding
def test_encode_histories_shift_pad_truncate():
    assert encode_histories([[3, 4, 5]], 5).tolist() == [[0, 0, 4, 5, 6]]
    assert encode_histories([[0, 1, 2, 3, 4, 5]], 3).tolist() == [[4, 5, 6]]
    assert encode_histories([[], [7]], 2).tolist() == [[0, 0], [0, 8]]
    assert encode_histories([[1]], 2).dtype == torch.long


# ======================================================================= mask
def test_attention_mask_rules():
    tokens = torch.tensor([[0, 0, 5, 6], [1, 2, 3, 4]])
    m = attention_mask(tokens, num_heads=3)
    assert m.shape == (6, 4, 4) and m.dtype == torch.bool
    a = m[0]  # sequence 0, head 0
    assert torch.equal(m[0], m[1]) and torch.equal(m[0], m[2]), "same mask for every head of a sequence"
    # causal
    assert a[2, 3] and not a[3, 3] and not a[3, 2]
    # real queries never see pad keys
    assert a[2, 0] and a[2, 1] and a[3, 0]
    # pad queries see (at least) themselves -> no empty rows
    assert not a[0, 0] and not a[1, 1]
    assert (~m).any(-1).all(), "every row must allow at least one key"
    # sequence 1 has no padding: pure causal mask
    causal = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.equal(m[3], causal)


# ======================================================================= model
def test_forward_shapes_and_scores():
    model = small_model().eval()
    tok = encode_histories([[1, 2, 3], [4]], 8)
    h = model(tok)
    assert h.shape == (2, 8, 16)
    assert model.score_all(h[:, -1]).shape == (2, 20)
    assert model.user_scores(tok).shape == (2, 20)
    torch.testing.assert_close(model.user_scores(tok), model.score_all(h[:, -1]))
    assert model.init_kwargs == dict(num_items=20, hidden_dim=16, max_len=8, num_blocks=2, num_heads=2, dropout=0.0)


def test_small_embedding_init():
    model = SASRec(5000, hidden_dim=64)
    w = model.item_emb.weight
    assert torch.all(w[0] == 0), "padding row must be zero"
    assert 0.015 < w[1:].std().item() < 0.025, "init item_emb with std 0.02 (default N(0,1) ruins training)"
    assert 0.015 < model.pos_emb.weight.std().item() < 0.025


def test_score_all_uses_item_embeddings_without_pad_row():
    model = small_model().eval()
    h = torch.randn(3, 16)
    torch.testing.assert_close(model.score_all(h), h @ model.item_emb.weight[1:].T)


def test_heavily_padded_input_is_finite():
    model = small_model().eval()
    tok = encode_histories([[5]], 8)  # 7 pads
    assert torch.isfinite(model(tok)).all(), "NaN: check the 'pad query may see itself' rule"


def test_causality():
    model = small_model().eval()
    a = encode_histories([[1, 2, 3, 4, 5, 6, 7, 8]], 8)
    b = a.clone()
    b[0, 6:] = torch.tensor([15, 16])
    ha, hb = model(a), model(b)
    torch.testing.assert_close(ha[:, :6], hb[:, :6])
    assert not torch.allclose(ha[:, 7], hb[:, 7])


def test_pad_content_cannot_leak_into_real_positions():
    model = small_model().eval()
    tok = encode_histories([[1, 2, 3]], 8)
    before = model(tok)[:, -3:].clone()
    with torch.no_grad():
        model.item_emb.weight[0] = torch.randn(16) * 10  # corrupt the pad embedding
    torch.testing.assert_close(model(tok)[:, -3:], before)


# ======================================================================= training pieces
def test_make_training_batch_alignment():
    inp, tgt = make_training_batch([[0, 1, 2, 3], [5, 6]], max_len=4)
    assert inp.tolist() == [[0, 1, 2, 3], [0, 0, 0, 6]]
    assert tgt.tolist() == [[-100, 1, 2, 3], [-100, -100, -100, 6]]
    inp, tgt = make_training_batch([list(range(10))], max_len=3)
    assert inp.tolist() == [[7, 8, 9]] and tgt.tolist() == [[7, 8, 9]]
    with pytest.raises(ValueError):
        make_training_batch([[1]], 4)


def test_sasrec_loss_ignores_padding_and_matches_cross_entropy():
    model = small_model().eval()
    inp, tgt = make_training_batch([[0, 1, 2, 3], [5, 6]], max_len=4)
    loss = sasrec_loss(model, inp, tgt)
    scores = model.score_all(model(inp))
    expected = torch.nn.functional.cross_entropy(scores.reshape(-1, 20), tgt.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(loss, expected)


# ======================================================================= candidates
def test_topk_candidates_excludes_full_history_and_is_distinct():
    model = small_model(max_len=3)
    model.train()
    hist = [[1, 2, 3, 4, 5, 6], [0]]
    cands = topk_candidates(model, hist, k=5)
    assert model.training, "mode must be restored"
    assert [len(c) for c in cands] == [5, 5]
    assert all(isinstance(x, int) for c in cands for x in c)
    assert not set(cands[0]) & set(hist[0]), "items 1-3 fell outside max_len but must still be excluded"
    assert len(set(cands[0])) == 5 and all(0 <= x < 20 for x in cands[0])
    allc = topk_candidates(model, hist, k=19, exclude_history=False)
    assert 1 in allc[0] or 2 in allc[0]


def test_topk_order_matches_scores():
    model = small_model().eval()
    cands = topk_candidates(model, [[4, 5]], k=3, exclude_history=True)[0]
    s = model.user_scores(encode_histories([[4, 5]], 8))[0]
    s[[4, 5]] = -float("inf")
    assert cands == s.topk(3).indices.tolist()


def test_build_candidate_examples_filters_and_ranks():
    exs = [SeqExample([1, 2], 9), SeqExample([3], 4), SeqExample([5], 7)]
    cands = [[9, 8, 7], [1, 2, 3], [6, 5, 7]]
    out = build_candidate_examples([10, 11, 12], exs, cands)
    assert out == [
        {"user": 10, "history": [1, 2], "candidates": [9, 8, 7], "target": 9, "target_rank": 1},
        {"user": 12, "history": [5], "candidates": [6, 5, 7], "target": 7, "target_rank": 3},
    ]


# ======================================================================= end-to-end
def _cyclic_sequences(n_users=60, n_items=30, length=8):
    # length < n_items, so the target is never already in the history (histories are excluded!)
    return [[(u + t) % n_items for t in range(length)] for u in range(n_users)]


def test_train_sasrec_learns_a_deterministic_pattern():
    torch.manual_seed(0)
    seqs = _cyclic_sequences()
    model = SASRec(30, hidden_dim=32, max_len=8, num_blocks=1, num_heads=2, dropout=0.0)
    valid = [SeqExample(s[:-1], s[-1]) for s in seqs[:20]]
    cfg = dict(epochs=40, batch_size=32, lr=0.01, eval_every=10, patience=10)
    hist = train_sasrec(model, seqs, valid, cfg, generator=torch.Generator().manual_seed(0), log=None)
    assert [h["epoch"] for h in hist] == [10, 20, 30, 40]
    assert {"epoch", "loss", "R@10", "N@10"} <= set(hist[0])
    assert not model.training
    assert hist[-1]["loss"] < hist[0]["loss"]
    rep = evaluate_full_ranking(model, valid, ks=(1, 5))
    assert list(rep) == ["R@1", "R@5", "N@1", "N@5"]
    assert rep["R@1"] > 0.9, "item (t+1) % 30 always follows item t: SASRec should nail this"


def test_train_sasrec_early_stopping_restores_best():
    torch.manual_seed(0)
    seqs = _cyclic_sequences()
    model = SASRec(30, hidden_dim=8, max_len=8, num_blocks=1, num_heads=1, dropout=0.0)
    valid = [SeqExample(s[:-1], s[-1]) for s in seqs[:20]]
    cfg = dict(epochs=200, batch_size=32, lr=0.01, eval_every=2, patience=2)
    hist = train_sasrec(model, seqs, valid, cfg, generator=torch.Generator().manual_seed(0), log=None)
    assert hist[-1]["epoch"] < 200, "should have stopped early once N@10 plateaued at 1.0"
    best = max(h["N@10"] for h in hist)
    assert evaluate_full_ranking(model, valid, ks=(10,))["N@10"] == pytest.approx(best)
