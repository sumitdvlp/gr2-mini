import torch
from torch import nn

from gr2mini.m04_rqvae.kmeans import pairwise_sq_dists
from gr2mini.m04_rqvae.quantizer import ResidualQuantizer


def make_rq(L=3, C=8, D=4, **kw):
    return ResidualQuantizer(L, C, D, **kw)


# ======================================================================= construction
def test_construction_with_ema_uses_buffers_only():
    rq = make_rq(use_ema=True)
    assert rq.codebooks.shape == (3, 8, 4)
    assert not isinstance(rq.codebooks, nn.Parameter)
    assert len(list(rq.parameters())) == 0, "with EMA the codebooks are not trained by the optimizer"
    buffers = dict(rq.named_buffers())
    for name in ["codebooks", "ema_cluster_size", "ema_embed_sum", "unused_steps", "initialized"]:
        assert name in buffers, f"missing buffer {name}"
    assert torch.equal(rq.ema_cluster_size, torch.ones(3, 8))
    assert torch.equal(rq.ema_embed_sum, rq.codebooks)
    assert rq.ema_embed_sum.data_ptr() != rq.codebooks.data_ptr(), "ema_embed_sum must be a copy"
    assert rq.unused_steps.dtype == torch.long and int(rq.unused_steps.sum()) == 0
    assert bool(rq.initialized) is False


def test_construction_without_ema_makes_codebooks_a_parameter():
    rq = make_rq(use_ema=False)
    assert isinstance(rq.codebooks, nn.Parameter)
    assert [p.shape for p in rq.parameters()] == [torch.Size([3, 8, 4])]


# ======================================================================= forward
def test_forward_shapes_and_residual_identities():
    rq = make_rq()
    z = torch.randn(16, 4)
    out = rq(z)
    assert out.codes.shape == (16, 3) and out.codes.dtype == torch.long
    assert out.residuals.shape == (16, 3, 4) and out.quantized.shape == (16, 3, 4)
    assert len(out.dists) == 3 and out.dists[0].shape == (16, 8)
    torch.testing.assert_close(out.residuals[:, 0], z)  # r_1 = z  (Eq. 19)
    for k in range(2):  # r_{k+1} = r_k - q_k  (Eq. 22)
        torch.testing.assert_close(out.residuals[:, k + 1], out.residuals[:, k] - out.quantized[:, k])
    torch.testing.assert_close(out.z_hat, out.quantized.sum(1))  # Eq. 23
    torch.testing.assert_close(out.z_q, out.z_hat)  # straight-through: same forward value


def test_forward_picks_the_nearest_codeword_at_every_level():
    rq = make_rq()
    out = rq(torch.randn(32, 4))
    for k in range(3):
        brute = ((out.residuals[:, k, None, :] - rq.codebooks[k][None]) ** 2).sum(-1)
        chosen = brute.gather(1, out.codes[:, k:k + 1]).squeeze(1)
        torch.testing.assert_close(chosen, brute.min(1).values, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out.quantized[:, k], rq.codebooks[k][out.codes[:, k]])
        torch.testing.assert_close(out.dists[k], brute, atol=1e-5, rtol=1e-5)


def test_straight_through_gradient_is_identity():
    rq = make_rq()
    z = torch.randn(5, 4, requires_grad=True)
    rq(z).z_q.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z))


def test_commitment_loss_value_and_it_never_trains_codebooks():
    rq = make_rq(use_ema=False)
    z = torch.randn(10, 4, requires_grad=True)
    out = rq(z)
    r, q = out.residuals.detach(), out.quantized.detach()
    expected = sum(((r[:, k] - q[:, k]) ** 2).sum(-1).mean() for k in range(3))
    torch.testing.assert_close(out.commitment_loss.detach(), expected)
    out.commitment_loss.backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert rq.codebooks.grad is None or rq.codebooks.grad.abs().sum() == 0, (
        "commitment loss must not reach the codebooks: detach q_k AND the residual chain"
    )


def test_codebook_loss_zero_with_ema_positive_and_trainable_without():
    z = torch.randn(10, 4)
    assert make_rq(use_ema=True)(z).codebook_loss.item() == 0.0
    rq = make_rq(use_ema=False)
    out = rq(z)
    r, q = out.residuals.detach(), out.quantized.detach()
    expected = sum(((r[:, k] - q[:, k]) ** 2).sum(-1).mean() for k in range(3))
    torch.testing.assert_close(out.codebook_loss.detach(), expected)
    out.codebook_loss.backward()
    assert rq.codebooks.grad is not None and rq.codebooks.grad.abs().sum() > 0


def test_dists_are_differentiable_wrt_z():
    rq = make_rq()
    z = torch.randn(6, 4, requires_grad=True)
    sum(d.sum() for d in rq(z).dists).backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_forward_does_not_modify_any_state():
    rq = make_rq(use_ema=True, dead_code_threshold=1)
    rq.train()
    before = {k: v.clone() for k, v in rq.state_dict().items()}
    rq(torch.randn(20, 4))
    for k, v in rq.state_dict().items():
        assert torch.equal(v, before[k]), f"forward() changed {k}; state updates belong in update_codebooks()"


def test_encode_matches_forward():
    rq = make_rq()
    z = torch.randn(25, 4)
    assert torch.equal(rq.encode(z), rq(z).codes)


# ======================================================================= RQ-K-means init
def test_init_codebooks_level1_finds_the_clusters(gen):
    centers = torch.tensor([[5.0, 0, 0, 0], [-5.0, 0, 0, 0], [0, 5.0, 0, 0], [0, -5.0, 0, 0]])
    z = centers.repeat_interleave(30, 0) + 0.1 * torch.randn(120, 4, generator=gen)
    rq = ResidualQuantizer(2, 4, 4)
    rq.init_codebooks(z, kmeans_iters=20, generator=gen)
    assert bool(rq.initialized) is True
    d = pairwise_sq_dists(rq.codebooks[0], centers)
    assert sorted(d.argmin(1).tolist()) == [0, 1, 2, 3] and (d.min(1).values < 0.1).all()
    assert rq.codebooks[1].norm(dim=1).max() < 1.0, "level 2 must be fit on the (small) residuals"
    torch.testing.assert_close(rq.ema_embed_sum, rq.codebooks)
    assert torch.equal(rq.ema_cluster_size, torch.ones(2, 4))


def test_init_codebooks_each_level_reduces_error(gen):
    z = torch.randn(300, 4, generator=gen)
    rq = ResidualQuantizer(3, 16, 4)
    rq.init_codebooks(z, generator=gen)
    out = rq(z)
    errs = [((z - out.quantized[:, : k + 1].sum(1)) ** 2).sum(-1).mean().item() for k in range(3)]
    assert errs[0] > errs[1] > errs[2]


def test_init_codebooks_keeps_the_same_parameter_object():
    rq = ResidualQuantizer(2, 4, 3, use_ema=False)
    param = rq.codebooks
    rq.init_codebooks(torch.randn(50, 3))
    assert rq.codebooks is param and isinstance(rq.codebooks, nn.Parameter)


# ======================================================================= EMA
def test_ema_update_matches_hand_computation():
    rq = ResidualQuantizer(1, 2, 1, use_ema=True, ema_decay=0.5, eps=1e-5)
    rq.codebooks[0] = torch.tensor([[0.0], [10.0]])
    rq.ema_embed_sum[0] = torch.tensor([[0.0], [10.0]])
    rq.ema_cluster_size[0] = torch.tensor([1.0, 1.0])
    rq.ema_update(0, torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([0, 0, 0]))
    # n = [3, 0], s = [6, 0];  N = 0.5*[1,1] + 0.5*[3,0] = [2, 0.5];  M = 0.5*[0,10] + 0.5*[6,0] = [3, 5]
    torch.testing.assert_close(rq.ema_cluster_size[0], torch.tensor([2.0, 0.5]))
    torch.testing.assert_close(rq.ema_embed_sum[0], torch.tensor([[3.0], [5.0]]))
    total = 2.5
    n_smooth = (torch.tensor([2.0, 0.5]) + 1e-5) / (total + 2 * 1e-5) * total
    torch.testing.assert_close(rq.codebooks[0], torch.tensor([[3.0], [5.0]]) / n_smooth[:, None])
    # i.e. ~[[1.5], [10.0]]: the used code moved toward its points, the unused code stayed put
    assert abs(rq.codebooks[0, 1, 0].item() - 10.0) < 1e-3


def test_update_codebooks_with_decay_zero_is_a_kmeans_step():
    rq = ResidualQuantizer(1, 2, 1, use_ema=True, ema_decay=0.0)
    rq.codebooks[0] = torch.tensor([[0.0], [10.0]])
    out = rq(torch.tensor([[1.0], [3.0], [9.0], [11.0]]))
    info = rq.update_codebooks(out)
    torch.testing.assert_close(rq.codebooks[0], torch.tensor([[2.0], [10.0]]), atol=1e-3, rtol=0)
    assert info == {"dead_codes_reset": [0]}


def test_update_codebooks_updates_every_level():
    rq = ResidualQuantizer(3, 4, 2, use_ema=True, ema_decay=0.5)
    rq.init_codebooks(torch.randn(64, 2))
    before = rq.codebooks.clone()
    rq.update_codebooks(rq(torch.randn(64, 2) * 3))
    for k in range(3):
        assert not torch.equal(rq.codebooks[k], before[k]), f"level {k} was not updated"


# ======================================================================= dead codes
def test_dead_code_counter_then_reset_to_worst_residual():
    rq = ResidualQuantizer(1, 3, 2, use_ema=False, dead_code_threshold=2)
    with torch.no_grad():
        rq.codebooks[0] = torch.tensor([[0.0, 0.0], [1.0, 0.0], [100.0, 100.0]])  # code 2 is far away
    r = torch.tensor([[0.1, 0.0], [0.9, 0.0], [0.0, 3.0], [0.2, 0.0]])
    codes = torch.tensor([0, 1, 0, 0])  # [0, 3] is served worst (error 3)

    assert rq.reset_dead_codes(0, r, codes) == 0  # unused for 1 step < threshold 2
    assert rq.unused_steps[0].tolist() == [0, 0, 1]
    assert rq.codebooks[0, 2].tolist() == [100.0, 100.0]

    assert rq.reset_dead_codes(0, r, codes) == 1  # now 2 steps -> reset
    assert rq.codebooks[0, 2].tolist() == [0.0, 3.0]
    assert rq.ema_embed_sum[0, 2].tolist() == [0.0, 3.0]
    assert rq.ema_cluster_size[0, 2].item() == 1.0
    assert rq.unused_steps[0].tolist() == [0, 0, 0]


def test_multiple_dead_codes_get_distinct_worst_residuals():
    rq = ResidualQuantizer(1, 4, 1, use_ema=False, dead_code_threshold=1)
    with torch.no_grad():
        rq.codebooks[0] = torch.tensor([[0.0], [50.0], [60.0], [70.0]])
    r = torch.tensor([[1.0], [-3.0], [2.0], [-7.0]])  # all served by code 0; errors 1, 3, 2, 7
    assert rq.reset_dead_codes(0, r, torch.zeros(4, dtype=torch.long)) == 3
    assert sorted(rq.codebooks[0, 1:, 0].tolist()) == [-7.0, -3.0, 2.0]
    assert rq.codebooks[0, 0, 0].item() == 0.0


def test_update_codebooks_reports_resets_per_level():
    rq = ResidualQuantizer(2, 8, 2, use_ema=True, dead_code_threshold=1)
    rq.init_codebooks(torch.randn(100, 2))
    with torch.no_grad():
        rq.codebooks[:, 7] = 1e4  # code 7 can never be chosen at either level
    info = rq.update_codebooks(rq(torch.randn(100, 2)))
    assert len(info["dead_codes_reset"]) == 2
    assert all(n >= 1 for n in info["dead_codes_reset"])
    assert rq.codebooks[:, 7].abs().max() < 1e3, "the dead code should have been moved onto the data"


def test_update_codebooks_without_dead_code_reset():
    rq = ResidualQuantizer(2, 4, 2, use_ema=True, dead_code_threshold=None)
    rq.init_codebooks(torch.randn(50, 2))
    assert rq.update_codebooks(rq(torch.randn(50, 2))) == {"dead_codes_reset": [0, 0]}
