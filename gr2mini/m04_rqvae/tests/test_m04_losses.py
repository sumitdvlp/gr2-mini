import math

import pytest
import torch
import torch.nn.functional as F

from gr2mini.m04_rqvae.contrastive import build_coengagement_neighbors, info_nce, sample_positives
from gr2mini.m04_rqvae.losses import diversity_loss, reconstruction_loss, soft_assignments


# ======================================================================= reconstruction
def test_reconstruction_loss_is_sum_over_dims_mean_over_batch():
    x_hat = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    x = torch.zeros(2, 2)
    assert reconstruction_loss(x_hat, x).item() == pytest.approx(2.5)  # (1+4 + 0) / 2
    assert reconstruction_loss(x_hat, x).item() == pytest.approx(2 * F.mse_loss(x_hat, x).item())


# ======================================================================= diversity
def test_soft_assignments_rows_sum_to_one_and_prefer_close_codes():
    d = torch.tensor([[0.0, 1.0, 4.0], [9.0, 0.0, 9.0]])
    p = soft_assignments(d, temperature=1.0)
    torch.testing.assert_close(p.sum(-1), torch.ones(2))
    assert p[0].argmax() == 0 and p[1].argmax() == 1
    torch.testing.assert_close(p[0], torch.softmax(-d[0], 0))
    sharp = soft_assignments(d, temperature=0.01)
    assert sharp[0, 0] > 0.999


def test_diversity_loss_uniform_usage_equals_number_of_levels():
    dists = [torch.zeros(10, 4), torch.zeros(10, 8), torch.zeros(10, 16)]  # every code equally likely
    assert diversity_loss(dists).item() == pytest.approx(3.0)


def test_diversity_loss_total_collapse_equals_sum_of_codebook_sizes():
    def collapsed(C):
        d = torch.full((10, C), 1e4)
        d[:, 0] = 0.0
        return d

    assert diversity_loss([collapsed(4), collapsed(8)]).item() == pytest.approx(12.0, rel=1e-4)


def test_diversity_loss_is_between_and_differentiable():
    d = torch.rand(32, 8, requires_grad=True)
    val = diversity_loss([d])
    assert 1.0 - 1e-6 <= val.item() <= 8.0
    val.backward()
    assert d.grad is not None and d.grad.abs().sum() > 0


# ======================================================================= InfoNCE
def _manual_info_nce(a, p, T):
    """Eq. 2 written out literally, one anchor at a time."""
    a = a / a.norm(dim=1, keepdim=True)
    p = p / p.norm(dim=1, keepdim=True)
    losses = []
    for i in range(len(a)):
        pos = math.exp(float(a[i] @ p[i]) / T)
        neg = sum(math.exp(float(a[i] @ p[n]) / T) for n in range(len(a)) if n != i)
        losses.append(-math.log(pos / (pos + neg)))
    return losses


def test_info_nce_matches_eq2(gen):
    a, p = torch.randn(6, 5, generator=gen), torch.randn(6, 5, generator=gen)
    expected = sum(_manual_info_nce(a, p, 0.5)) / 6
    assert info_nce(a, p, temperature=0.5).item() == pytest.approx(expected, rel=1e-5)


def test_info_nce_valid_mask_averages_only_valid_rows(gen):
    a, p = torch.randn(6, 5, generator=gen), torch.randn(6, 5, generator=gen)
    valid = torch.tensor([True, False, True, True, False, False])
    per_row = _manual_info_nce(a, p, 0.2)
    expected = (per_row[0] + per_row[2] + per_row[3]) / 3
    assert info_nce(a, p, 0.2, valid).item() == pytest.approx(expected, rel=1e-5)


def test_info_nce_no_valid_rows_is_zero_and_backpropagates():
    a = torch.randn(4, 3, requires_grad=True)
    loss = info_nce(a, torch.randn(4, 3), 0.1, torch.zeros(4, dtype=torch.bool))
    assert loss.item() == 0.0
    loss.backward()  # must not raise


def test_info_nce_is_scale_invariant_and_near_zero_when_aligned():
    eye = torch.eye(4)
    assert info_nce(eye, eye, temperature=0.05).item() < 1e-6
    a, p = torch.randn(5, 3), torch.randn(5, 3)
    assert info_nce(10 * a, p, 0.3).item() == pytest.approx(info_nce(a, p, 0.3).item(), rel=1e-5)


# ======================================================================= co-engagement
def test_build_coengagement_neighbors_docstring_example():
    nb = build_coengagement_neighbors([[0, 1, 2], [2, 3]], num_items=5, window=1)
    assert nb == [[1], [0, 2], [1, 3], [2], []]


def test_build_coengagement_neighbors_window_and_no_self():
    nb = build_coengagement_neighbors([[5, 5, 6, 7]], num_items=8, window=2)
    assert nb[5] == [6, 7]  # 5 is never its own neighbour even though it repeats
    assert nb[6] == [5, 7] and nb[7] == [5, 6]
    assert nb[0] == []


def test_sample_positives_draws_neighbours_and_flags_missing(gen):
    nb = [[1, 2], [0], [], [0, 1, 2]]
    ids = torch.tensor([0, 1, 2, 3, 0, 2])
    pos, valid = sample_positives(ids, nb, generator=gen)
    assert pos.dtype == torch.long and valid.dtype == torch.bool and pos.shape == ids.shape
    assert valid.tolist() == [True, True, False, True, True, False]
    for i, p, v in zip(ids.tolist(), pos.tolist(), valid.tolist()):
        assert (p in nb[i]) if v else (p == i)


def test_sample_positives_is_random_but_reproducible():
    nb = [list(range(1, 50))] + [[0]] * 49
    ids = torch.zeros(200, dtype=torch.long)
    p1, _ = sample_positives(ids, nb, generator=torch.Generator().manual_seed(0))
    p2, _ = sample_positives(ids, nb, generator=torch.Generator().manual_seed(0))
    assert torch.equal(p1, p2)
    assert p1.unique().numel() > 20, "should sample uniformly among neighbours, not always the first"
