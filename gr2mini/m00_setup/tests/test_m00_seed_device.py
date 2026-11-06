import random

import numpy as np
import pytest
import torch

from gr2mini.m00_setup.device import fp16_sanity_check, get_device, pick_dtype, vram_info
from gr2mini.m00_setup.seed import make_generator, seed_everything


# ---------------------------------------------------------------- seeding
def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_seed_everything_makes_all_three_rngs_reproducible():
    seed_everything(123)
    first = _draw()
    _draw()  # advance the state
    seed_everything(123)
    assert _draw() == first


def test_different_seeds_give_different_draws():
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    b = _draw()
    assert all(x != y for x, y in zip(a, b))


@pytest.fixture
def _restore_determinism_flags():
    old = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
           torch.are_deterministic_algorithms_enabled())
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old[0], old[1]
    torch.use_deterministic_algorithms(old[2])


def test_seed_everything_deterministic_flag(_restore_determinism_flags):
    seed_everything(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()


def test_make_generator_is_independent_of_global_state():
    g1 = make_generator(7)
    x1 = torch.rand(5, generator=g1)
    torch.manual_seed(999)  # changing global state must not matter
    torch.rand(10)
    g2 = make_generator(7)
    x2 = torch.rand(5, generator=g2)
    assert isinstance(g1, torch.Generator)
    assert torch.equal(x1, x2)


# ---------------------------------------------------------------- devices
def test_get_device_cpu_and_invalid():
    assert get_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError):
        get_device("tpu")


def test_get_device_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device("cuda").type == "cpu"
    assert get_device("auto").type == "cpu"


def test_get_device_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device("cuda").type == "cuda"


def test_pick_dtype_cpu_is_always_fp32():
    for prefer in ("fp16", "bf16", "fp32"):
        assert pick_dtype(torch.device("cpu"), prefer) == torch.float32


def test_pick_dtype_turing_has_no_native_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    cuda = torch.device("cuda")
    assert pick_dtype(cuda, "bf16") == torch.float16
    assert pick_dtype(cuda, "fp16") == torch.float16
    assert pick_dtype(cuda, "fp32") == torch.float32


def test_pick_dtype_ampere_gets_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 6))
    assert pick_dtype(torch.device("cuda"), "bf16") == torch.bfloat16


def test_pick_dtype_invalid():
    with pytest.raises(ValueError):
        pick_dtype(torch.device("cpu"), "int8")


def test_vram_info_cpu_is_none():
    assert vram_info(torch.device("cpu")) is None


def test_fp16_sanity_check_on_cpu():
    out = fp16_sanity_check(torch.device("cpu"), n=64)
    assert set(out) == {"ok", "finite", "max_rel_err"}
    assert out["ok"] is True and out["finite"] is True
    assert 0.0 <= out["max_rel_err"] < 1e-2


def test_fp16_sanity_check_is_deterministic():
    a = fp16_sanity_check(torch.device("cpu"), n=32, seed=3)
    torch.rand(100)
    b = fp16_sanity_check(torch.device("cpu"), n=32, seed=3)
    assert a == b


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_real_gpu_report():
    dev = get_device("cuda")
    info = vram_info(dev)
    assert info["total_gb"] > 0 and 0 < info["free_gb"] <= info["total_gb"]
    assert fp16_sanity_check(dev)["ok"]
