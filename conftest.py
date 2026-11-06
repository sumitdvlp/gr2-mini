"""Shared pytest fixtures for all modules.

Every test starts from the same global random state, so a failure is reproducible.
"""
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _fixed_global_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield


@pytest.fixture
def gen() -> torch.Generator:
    """A seeded CPU torch.Generator for functions that take an explicit generator."""
    return torch.Generator().manual_seed(1234)
