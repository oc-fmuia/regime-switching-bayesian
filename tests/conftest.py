"""Pytest configuration and shared fixtures for v0 regime-switching tests."""

import numpy as np
import pytest

from src.data_gen import generate_hmm_data


@pytest.fixture
def small_data():
    """Small synthetic dataset (T=30, d=2) for fast unit tests."""
    return generate_hmm_data(T=30, K=2, d=2, seed=42)


@pytest.fixture
def medium_data():
    """Medium synthetic dataset (T=120, d=3) for statistical tests."""
    return generate_hmm_data(T=120, K=2, d=3, seed=42)
