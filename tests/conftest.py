"""
Pytest configuration and fixtures for regime-switching tests.
"""

import numpy as np
import pytest


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return {
        "T": 20,           # 20 months
        "K": 2,            # 2 regimes
        "G": 2,            # 2 geographies
        "H": 2,            # 2 sectors
        "d_pub": 3,        # 3 public assets
        "d_priv": 2,       # 2 private assets
        "r": 2,            # 2 factors
        "seed": 42,
    }


@pytest.fixture
def full_config():
    """Full config for realistic testing."""
    return {
        "T": 120,          # 120 months
        "K": 3,            # 3 regimes
        "G": 2,            # 2 geographies
        "H": 2,            # 2 sectors
        "d_pub": 4,        # 4 public assets
        "d_priv": 3,       # 3 private assets
        "r": 2,            # 2 factors
        "seed": 42,
    }


@pytest.fixture
def random_seed():
    """Fixture to ensure reproducibility."""
    np.random.seed(42)
    return 42
