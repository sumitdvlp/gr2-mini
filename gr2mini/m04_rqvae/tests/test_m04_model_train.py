import pytest
import torch
from torch import nn

from gr2mini.m04_rqvae.contrastive import build_coengagement_neighbors
from gr2mini.m04_rqvae.model import RQVAE, load_rqvae, mlp, save_rqvae
from gr2mini.m04_rqvae.train import rqvae_loss, train_rqvae

SMALL = dict(in_dim=16, latent_dim=8, hidden_dims=[32], num_levels=3, codebook_size=16)


def _toy_embeddings(n=256, d=16, n_clusters=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_clusters, d, generator=g)
    x = centers[torch.arange(n) % n_clusters] + 0.3 * torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


def _train_cfg(**over):
    cfg = dict(epochs=20, batch_size=64, lr=0.003, kmeans_iters=10, eval_every=5,
               commitment_weight=0.25, codebook_weight=1.0, diversity_weight=0.0,
               diversity_temperature=1.0, contrastive_weight=0.0, contrastive_temperature=0.1)
    cfg.update(over)
    return cfg


# ======================================================================= model
def test_mlp_structure():
    m = mlp([4, 8, 2])
    assert [type(layer) for layer in m] == [nn.Linear, nn.ReLU, nn.Linear]
    assert (m[0].in_features, m[0].out_features, m[2].out_features) == (4, 8, 2)
    assert [type(layer) for layer in mlp([3, 5])] == [nn.Linear]


def test_rqvae_shapes_and_parameters():
    model = RQVAE(**SMALL, use_ema=True)
    out = model(torch.randn(10, 16))
    assert out.x_hat.shape == (10, 16) and out.z.shape == (10, 8) and out.rq.codes.shape == (10, 3)
    assert model.encoder[0].in_features == 16 and model.encoder[-1].out_features == 8
    assert model.decoder[0].in_features == 8 and model.decoder[-1].out_features == 16
    n_codebook = sum(p.numel() for p in model.quantizer.parameters())
    assert n_codebook == 0, "EMA codebooks are buffers, not parameters"


def test_rqvae_init_kwargs_and_from_config():
    rq_cfg = {"latent_dim": 8, "hidden_dims": [32], "num_levels": 3, "codebook_size": 16,
              "use_ema": False, "ema_decay": 0.95, "dead_code_threshold": None, "epochs": 5}
    model = RQVAE.from_config(16, rq_cfg)
    assert model.init_kwargs == {**SMALL, "use_ema": False, "ema_decay": 0.95, "dead_code_threshold": None}
    assert isinstance(model.quantizer.codebooks, nn.Parameter)


def test_decoder_sees_the_quantized_latent():
    model = RQVAE(**SMALL)
    x = torch.randn(4, 16)
    out = model(x)
    torch.testing.assert_close(out.x_hat, model.decoder(out.rq.z_hat))


def test_get_codes_matches_forward_and_is_chunk_invariant():
    model = RQVAE(**SMALL)
    x = torch.randn(23, 16)
    assert torch.equal(model.get_codes(x, batch_size=5), model(x).rq.codes)
    model.train()
    model.get_codes(x)
    assert model.training, "get_codes must not switch the model to eval mode permanently"


def test_save_load_roundtrip(tmp_path):
    model = RQVAE(**SMALL, dead_code_threshold=2)
    x = _toy_embeddings()
    with torch.no_grad():
        model.quantizer.init_codebooks(model.encode(x))
    model.quantizer.update_codebooks(model(x).rq)  # make the EMA buffers non-trivial
    path = tmp_path / "ckpt" / "rqvae.pt"
    save_rqvae(model, path, extra={"note": "hi"})
    loaded = load_rqvae(path)
    assert not loaded.training
    assert torch.equal(loaded.get_codes(x), model.get_codes(x))
    for (k, v), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k == k2 and torch.equal(v, v2), k
    assert torch.load(path, weights_only=True)["extra"] == {"note": "hi"}


# ======================================================================= loss
def test_rqvae_loss_total_is_weighted_sum_of_terms():
    model = RQVAE(**SMALL, use_ema=False)
    x, pos = _toy_embeddings(32), _toy_embeddings(32, seed=1)
    valid = torch.ones(32, dtype=torch.bool)
    w = dict(commitment_weight=0.3, codebook_weight=0.7, diversity_weight=0.2, contrastive_weight=0.5)
    total, terms, out = rqvae_loss(model, x, pos, valid, **w)
    assert set(terms) == {"loss", "rec", "codebook", "commitment", "diversity", "contrastive"}
    assert all(isinstance(v, float) for v in terms.values())
    expected = (terms["rec"] + 0.7 * terms["codebook"] + 0.3 * terms["commitment"]
                + 0.2 * terms["diversity"] + 0.5 * terms["contrastive"])
    assert total.item() == pytest.approx(expected, rel=1e-5)
    assert terms["loss"] == pytest.approx(total.item())
    assert terms["diversity"] > 0 and terms["contrastive"] > 0 and terms["codebook"] > 0


def test_rqvae_loss_disabled_terms_are_zero_and_grads_reach_encoder_and_decoder():
    model = RQVAE(**SMALL)
    total, terms, _ = rqvae_loss(model, _toy_embeddings(16))
    assert terms["diversity"] == 0.0 and terms["contrastive"] == 0.0 and terms["codebook"] == 0.0
    total.backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.decoder[0].weight.grad.abs().sum() > 0


def test_rqvae_loss_contrastive_requires_positives():
    with pytest.raises(ValueError):
        rqvae_loss(RQVAE(**SMALL), _toy_embeddings(8), contrastive_weight=0.1)


# ======================================================================= training loop
def test_train_rqvae_history_and_learning():
    x = _toy_embeddings()
    model = RQVAE(**SMALL, dead_code_threshold=2)
    hist = train_rqvae(model, x, _train_cfg(), generator=torch.Generator().manual_seed(0), log=None)
    assert [h["epoch"] for h in hist] == [1, 5, 10, 15, 20]
    for h in hist:
        assert {"epoch", "loss", "rec", "commitment", "uniqueness", "used_codes"} <= set(h)
        assert 0.0 <= h["uniqueness"] <= 1.0 and len(h["used_codes"]) == 3
        assert all(1 <= u <= 16 for u in h["used_codes"])
        assert all(v == v for k, v in h.items() if isinstance(v, float)), "NaN in history"
    assert hist[-1]["rec"] < hist[0]["rec"], "reconstruction should improve"
    assert bool(model.quantizer.initialized)
    assert model.training


def test_train_rqvae_with_contrastive_and_diversity():
    x = _toy_embeddings()
    seqs = [list(range(i, i + 6)) for i in range(0, 250, 3)]
    nb = build_coengagement_neighbors(seqs, num_items=256, window=1)
    cfg = _train_cfg(epochs=6, eval_every=3, contrastive_weight=0.1, diversity_weight=0.1)
    hist = train_rqvae(RQVAE(**SMALL), x, cfg, neighbors=nb, generator=torch.Generator().manual_seed(0), log=None)
    assert [h["epoch"] for h in hist] == [1, 3, 6]
    assert hist[-1]["contrastive"] > 0 and hist[-1]["diversity"] > 0


def test_train_rqvae_contrastive_without_neighbors_raises():
    with pytest.raises(ValueError):
        train_rqvae(RQVAE(**SMALL), _toy_embeddings(), _train_cfg(contrastive_weight=0.1), neighbors=None, log=None)


def test_train_rqvae_is_reproducible():
    x = _toy_embeddings()
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        model = RQVAE(**SMALL, dead_code_threshold=2)
        runs.append(train_rqvae(model, x, _train_cfg(epochs=4, eval_every=2),
                                generator=torch.Generator().manual_seed(0), log=None))
    assert runs[0] == runs[1]
