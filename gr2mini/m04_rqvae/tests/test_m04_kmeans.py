import pytest
import torch

from gr2mini.m04_rqvae.kmeans import kmeans, kmeans_plusplus_init, pairwise_sq_dists

CENTERS = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])


def _blobs(centers, n_per, spread, gen):
    return torch.cat([c + spread * torch.randn(n_per, centers.shape[1], generator=gen) for c in centers])


def _grid_centers(n_side=4, step=50.0):
    return torch.tensor([[i * step, j * step] for i in range(n_side) for j in range(n_side)])


# ---------------------------------------------------------------- distances
def test_pairwise_sq_dists_matches_brute_force(gen):
    x, c = torch.randn(7, 5, generator=gen), torch.randn(4, 5, generator=gen)
    expected = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    torch.testing.assert_close(pairwise_sq_dists(x, c), expected, atol=1e-5, rtol=1e-5)


def test_pairwise_sq_dists_nonnegative_and_zero_diagonal(gen):
    big = torch.randn(30, 3, generator=gen) * 1000
    assert (pairwise_sq_dists(big, big) >= 0).all(), "clamp float-error negatives to 0"
    x = torch.randn(10, 3, generator=gen)
    torch.testing.assert_close(pairwise_sq_dists(x, x).diagonal(), torch.zeros(10), atol=1e-4, rtol=0)


def test_pairwise_sq_dists_is_differentiable(gen):
    x = torch.randn(4, 3, generator=gen, requires_grad=True)
    c = torch.randn(2, 3, generator=gen, requires_grad=True)
    pairwise_sq_dists(x, c).sum().backward()
    assert x.grad is not None and c.grad is not None


# ---------------------------------------------------------------- k-means++
def test_kmeanspp_returns_distinct_rows_of_x(gen):
    x = torch.randn(50, 3, generator=gen)
    c = kmeans_plusplus_init(x, 5, generator=gen)
    assert c.shape == (5, 3)
    for row in c:
        assert any(torch.equal(row, xr) for xr in x), "every centre must be a data point"
    assert len({tuple(r.tolist()) for r in c}) == 5


def test_kmeanspp_spreads_centres_over_separated_blobs(gen):
    x = _blobs(CENTERS, 25, 0.5, gen)
    c = kmeans_plusplus_init(x, 4, generator=gen)
    assert sorted(pairwise_sq_dists(c, CENTERS).argmin(1).tolist()) == [0, 1, 2, 3]


def test_kmeanspp_is_deterministic_given_generator():
    x = torch.randn(40, 2)
    a = kmeans_plusplus_init(x, 6, generator=torch.Generator().manual_seed(5))
    torch.rand(100)  # global RNG must not matter
    b = kmeans_plusplus_init(x, 6, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_kmeanspp_handles_fewer_distinct_points_than_k(gen):
    x = torch.tensor([[0.0, 0.0]] * 5 + [[1.0, 1.0]] * 5)
    c = kmeans_plusplus_init(x, 4, generator=gen)
    assert c.shape == (4, 2) and torch.isfinite(c).all()


def test_kmeanspp_too_few_rows_raises():
    with pytest.raises(ValueError):
        kmeans_plusplus_init(torch.randn(3, 2), 4)


# ---------------------------------------------------------------- Lloyd's k-means
def test_kmeans_recovers_blob_centres(gen):
    x = _blobs(CENTERS, 50, 1.0, gen)
    cents, assign = kmeans(x, 4, n_iters=20, generator=gen)
    d = pairwise_sq_dists(cents, CENTERS)
    assert sorted(d.argmin(1).tolist()) == [0, 1, 2, 3]
    assert (d.min(1).values.sqrt() < 0.5).all(), "centroids should sit at the blob means"
    for b in range(4):
        assert assign[b * 50:(b + 1) * 50].unique().numel() == 1, "each blob should be one cluster"


def test_kmeans_assignments_consistent_with_returned_centroids(gen):
    x = torch.randn(200, 4, generator=gen)
    cents, assign = kmeans(x, 8, n_iters=3, generator=gen)  # few iters: likely NOT converged
    assert cents.shape == (8, 4) and assign.shape == (200,) and assign.dtype == torch.long
    d = pairwise_sq_dists(x, cents)
    chosen = d.gather(1, assign[:, None]).squeeze(1)
    torch.testing.assert_close(chosen, d.min(1).values)


def test_kmeans_uses_all_clusters_and_leaves_input_alone(gen):
    x = _blobs(_grid_centers(), 8, 1.0, gen)  # 16 blobs
    x_before = x.clone()
    _, assign = kmeans(x, 16, n_iters=30, generator=gen)
    assert torch.equal(x, x_before)
    assert assign.unique().numel() == 16


def test_kmeans_improves_on_its_initialization(gen):
    x = torch.randn(300, 3, generator=gen)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    init = kmeans_plusplus_init(x, 10, generator=g1)
    cents, _ = kmeans(x, 10, n_iters=25, generator=g2)
    inertia = lambda c: pairwise_sq_dists(x, c).min(1).values.sum()  # noqa: E731
    assert inertia(cents) < inertia(init)
