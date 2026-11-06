import json

import pytest
import torch

from gr2mini.m01_metrics.metrics import uniqueness
from gr2mini.m04_rqvae.model import RQVAE
from gr2mini.m05_semantic_ids.assign import (
    assign_semantic_ids,
    codebook_utilization,
    compute_semantic_codes,
    find_collisions,
    prefix_stats,
    randomize_last_levels,
    resolve_collisions,
)
from gr2mini.m05_semantic_ids.sid_index import SIDIndex
from gr2mini.m05_semantic_ids.sid_format import sid_to_str

CODES = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 4], [0, 0, 0], [1, 2, 3], [0, 0, 0]])


# ======================================================================= deterministic codes
def test_compute_semantic_codes_matches_model_and_is_cpu_long():
    model = RQVAE(in_dim=6, latent_dim=4, hidden_dims=[8], num_levels=3, codebook_size=5)
    emb = torch.randn(17, 6)
    model.train()
    codes = compute_semantic_codes(model, emb, batch_size=4)
    assert codes.shape == (17, 3) and codes.dtype == torch.long and codes.device.type == "cpu"
    assert torch.equal(codes, model.get_codes(emb))


# ======================================================================= Eq. 10
def test_randomize_last_levels(gen):
    codes = torch.zeros(1000, 4, dtype=torch.long)
    out = randomize_last_levels(codes, 1, codebook_size=16, generator=gen)
    assert torch.equal(codes, torch.zeros(1000, 4, dtype=torch.long)), "input must not be modified"
    assert torch.equal(out[:, :3], codes[:, :3])
    assert out[:, 3].min() >= 0 and out[:, 3].max() < 16
    assert out[:, 3].unique().numel() == 16, "should cover the whole codebook"
    out2 = randomize_last_levels(codes, 2, 16, generator=gen)
    assert torch.equal(out2[:, :2], codes[:, :2]) and out2[:, 2].unique().numel() > 1


def test_randomize_last_levels_zero_and_invalid(gen):
    out = randomize_last_levels(CODES, 0, 8, generator=gen)
    assert torch.equal(out, CODES) and out.data_ptr() != CODES.data_ptr()
    with pytest.raises(ValueError):
        randomize_last_levels(CODES, 4, 8)
    with pytest.raises(ValueError):
        randomize_last_levels(CODES, -1, 8)


def test_randomize_is_reproducible():
    a = randomize_last_levels(CODES, 1, 50, generator=torch.Generator().manual_seed(3))
    b = randomize_last_levels(CODES, 1, 50, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


# ======================================================================= collisions
def test_find_collisions():
    assert find_collisions(CODES) == {(1, 2, 3): [0, 1, 4], (0, 0, 0): [3, 5]}
    assert find_collisions(torch.tensor([[0, 1], [1, 0]])) == {}


def test_resolve_collisions_makes_all_unique_and_only_touches_colliders(gen):
    out = resolve_collisions(CODES, 1, codebook_size=50, generator=gen)
    assert uniqueness(out) == 1.0
    assert torch.equal(CODES[1], torch.tensor([1, 2, 3])), "input must not be modified"
    assert torch.equal(out[2], CODES[2]), "item 2 never collided, so it must not change"
    assert torch.equal(out[0], CODES[0]) and torch.equal(out[3], CODES[3]), "lowest id keeps its SID"
    assert torch.equal(out[:, :2], CODES[:, :2]), "only the last level may be re-drawn"


def test_resolve_collisions_needs_more_rounds_with_tiny_codebook(gen):
    codes = torch.zeros(4, 2, dtype=torch.long)  # 4 identical SIDs, only 4 possible last codes
    out = resolve_collisions(codes, 1, codebook_size=4, generator=gen, max_rounds=1000)
    assert sorted(out[:, 1].tolist()) == [0, 1, 2, 3]


def test_resolve_collisions_errors():
    with pytest.raises(ValueError):
        resolve_collisions(CODES, 0, codebook_size=8)
    impossible = torch.zeros(5, 2, dtype=torch.long)  # 5 items, but only 4 possible last codes
    with pytest.raises(RuntimeError):
        resolve_collisions(impossible, 1, codebook_size=4, max_rounds=50)


def test_resolve_collisions_no_collisions_is_copy():
    codes = torch.tensor([[0, 1], [1, 0]])
    out = resolve_collisions(codes, 1, codebook_size=4)
    assert torch.equal(out, codes) and out.data_ptr() != codes.data_ptr()


# ======================================================================= stats
def test_codebook_utilization():
    assert codebook_utilization(CODES, codebook_size=4) == [0.5, 0.5, 0.75]


def test_prefix_stats():
    st = prefix_stats(CODES)
    assert [s["level"] for s in st] == [1, 2, 3]
    assert st[0] == {"level": 1, "distinct": 2, "largest_group": 4, "mean_group": 3.0}
    assert st[2]["distinct"] == 3 and st[2]["largest_group"] == 3


def test_assign_semantic_ids_report(gen):
    torch.manual_seed(0)
    # codebook (64) > #items (40), so resolution is always possible, whatever the prefixes are
    model = RQVAE(in_dim=6, latent_dim=4, hidden_dims=[8], num_levels=3, codebook_size=64)
    emb = torch.randn(40, 6)
    emb[20:] = emb[:20]  # duplicate items -> guaranteed semantic collisions
    codes, rep = assign_semantic_ids(model, emb, num_random_levels=1, codebook_size=64, generator=gen, max_rounds=1000)
    assert set(rep) == {"uniqueness_semantic", "uniqueness_random_last", "uniqueness_final", "utilization", "num_redrawn"}
    assert rep["uniqueness_final"] == 1.0 and uniqueness(codes) == 1.0
    assert rep["uniqueness_semantic"] == uniqueness(compute_semantic_codes(model, emb))
    assert rep["uniqueness_semantic"] == 0.0  # every item has an exact duplicate
    assert rep["uniqueness_random_last"] > rep["uniqueness_semantic"]  # Eq. 10 at work
    assert len(rep["utilization"]) == 3 and rep["num_redrawn"] >= 0
    assert torch.equal(codes[:, :2], compute_semantic_codes(model, emb)[:, :2])


# ======================================================================= SIDIndex
SIDS = [(1, 2, 3), (1, 2, 4), (1, 5, 0), (0, 0, 0)]


def test_index_lookups():
    idx = SIDIndex(SIDS, codebook_size=8)
    assert (idx.num_items, idx.num_levels, idx.codebook_size) == (4, 3, 8)
    assert idx.sid(1) == (1, 2, 4) and isinstance(idx.sid(1)[0], int)
    assert idx.item((1, 5, 0)) == 2 and idx.item([1, 5, 0]) == 2
    assert idx.item((7, 7, 7)) is None
    assert idx.sid_str(0) == sid_to_str((1, 2, 3))


def test_index_from_tensor():
    idx = SIDIndex(torch.tensor(SIDS), codebook_size=8)
    assert idx.sid(3) == (0, 0, 0) and type(idx.sid(3)[0]) is int


def test_index_item_from_text():
    idx = SIDIndex(SIDS, codebook_size=8)
    unknown = sid_to_str((6, 6, 6))
    short = sid_to_str((1, 2))
    assert idx.item_from_text(f"maybe {unknown} or {short} but really {sid_to_str((1, 5, 0))}") == 2
    assert idx.item_from_text("nothing") is None


def test_index_prefix():
    idx = SIDIndex(SIDS, codebook_size=8)
    assert idx.items_with_prefix((1, 2)) == [0, 1]
    assert idx.items_with_prefix((1,)) == [0, 1, 2]
    assert idx.items_with_prefix((9,)) == []
    assert idx.items_with_prefix(()) == [0, 1, 2, 3]


@pytest.mark.parametrize("bad", [[], [(1, 2), (1, 2, 3)], [(1, 2), (1, 9)], [(1, 2), (1, 2)], [(1, -1)]])
def test_index_validation(bad):
    with pytest.raises(ValueError):
        SIDIndex(bad, codebook_size=8)


def test_index_save_load(tmp_path):
    idx = SIDIndex(SIDS, codebook_size=8)
    p = tmp_path / "a" / "sids.json"
    idx.save(p)
    assert json.loads(p.read_text(encoding="utf-8")) == {"num_levels": 3, "codebook_size": 8, "sids": [list(s) for s in SIDS]}
    idx2 = SIDIndex.load(p)
    assert [idx2.sid(i) for i in range(4)] == SIDS and idx2.codebook_size == 8
