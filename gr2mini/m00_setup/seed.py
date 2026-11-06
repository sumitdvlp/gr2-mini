"""Seeding helpers.  (see README §2)"""
from __future__ import annotations

import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python `random`, NumPy, and torch (CPU and all CUDA devices).

    If `deterministic` is True, additionally:
        * torch.backends.cudnn.deterministic = True
        * torch.backends.cudnn.benchmark = False
        * torch.use_deterministic_algorithms(True, warn_only=True)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Return a `torch.Generator` on `device`, seeded with `seed`."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
