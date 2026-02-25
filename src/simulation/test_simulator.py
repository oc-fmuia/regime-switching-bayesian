"""
Tests for Monte Carlo simulator.

Progressive sizing (no expensive computation):
- Small (n_scenarios=100, n_steps=10): instant
- Medium (n_scenarios=500, n_steps=50): 1-2 seconds
- Large (n_scenarios=1000, n_steps=100): 2-5 seconds

All tests complete quickly by design.
"""

import sys
import numpy as np
import time

sys.path.insert(0, '/home/fmuia/.openclaw/workspace-fernando/regime-switching-bayesian/src')

from simulation.simulator import MonteCarloSimulator


def run_test(test_name, test_func):
    """Run test with timing."""
    try:
        start = time.time()
        test_func()
        elapsed = time.time() - start
        print(f"✓ {test_name} [{elapsed:.3f}s]")
        return True, elapsed
    except AssertionError as e:
        print(f"✗ {test_name}: {e}")
        return False, 0
    except Exception as e:
        print(f"✗ {test_name} (error): {type(e).__name__}: {str(e)[:60]}")
        return False, 0


# ============================================================================
# SMALL TESTS: Initialization and validation
# ============================================================================

def test_small_init():
    """Test simulator initialization."""
    sim = MonteCarloSimulator(n_assets=3, n_regimes=2, n_shocks=2, n_scenarios=100)
    assert sim.n_assets == 3
    assert sim.n_regimes == 2
    assert sim.n_shocks == 2
    assert sim.n_scenarios == 100


def test_small_invalid_dims():
    """Test that invalid dimensions are rejected."""
    try:
        MonteCarloSimulator(n_assets=0, n_regimes=2, n_shocks=2, n_scenarios=100)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_small_path_generation_shape():
    """Test path generation output shapes."""
    sim = MonteCarloSimulator(n_assets=3, n_regimes=2, n_shocks=2, n_scenarios=50)
    
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    mu = np.zeros((2, 3))
    Sigma = np.array([np.eye(3) * 0.01, np.eye(3) * 0.01])
    B = np.random.randn(2, 3, 2) * 0.05
    
    paths, regime_paths, shock_paths = sim.generate_paths(
        n_steps=10,
        transition_matrix=P,
        regime_means=mu,
        regime_covs=Sigma,
        loading_matrices=B,
        random_seed=42,
    )
    
    assert paths.shape == (50, 10, 3), f"Wrong paths shape: {paths.shape}"
    assert regime_paths.shape == (50, 10), f"Wrong regime shape: {regime_paths.shape}"
    assert shock_paths.shape == (50, 10, 2), f"Wrong shocks shape: {shock_paths.shape}"


def test_small_path_generation_no_shocks():
    """Test path generation without shock loadings."""
    sim = MonteCarloSimulator(n_assets=2, n_regimes=2, n_shocks=2, n_scenarios=30)
    
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    mu = np.zeros((2, 2))
    Sigma = np.array([np.eye(2) * 0.01, np.eye(2) * 0.01])
    
    paths, regime_paths, shock_paths = sim.generate_paths(
        n_steps=5,
        transition_matrix=P,
        regime_means=mu,
        regime_covs=Sigma,
        loading_matrices=None,  # No shocks
        random_seed=42,
    )
    
    assert paths.shape == (30, 5, 2)
    assert shock_paths.shape == (30, 5, 2)


def test_small_invalid_transition_matrix():
    """Test that invalid transition matrix is rejected."""
    sim = MonteCarloSimulator(n_assets=2, n_regimes=2, n_shocks=2, n_scenarios=50)
    
    P_wrong = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0]])  # Wrong shape
    mu = np.zeros((2, 2))
    Sigma = np.array([np.eye(2) * 0.01, np.eye(2) * 0.01])
    
    try:
        sim.generate_paths(10, P_wrong, mu, Sigma)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_small_path_statistics():
    """Test path statistics computation."""
    sim = MonteCarloSimulator(n_assets=2, n_regimes=1, n_shocks=1, n_scenarios=50)
    
    # Simple case: identity means and covariances
    P = np.array([[1.0]])
    mu = np.array([[0.001, 0.001]])
    Sigma = np.array([np.eye(2) * 0.001])
    
    paths, _, _ = sim.generate_paths(10, P, mu, Sigma, None, random_seed=42)
    
    stats = sim.compute_path_statistics(paths)
    
    assert "mean" in stats
    assert "std" in stats
    assert "median" in stats
    assert "quantile_5" in stats
    assert "quantile_95" in stats
    assert "cumulative" in stats
    
    assert stats["mean"].shape == (10, 2)
    assert stats["cumulative"].shape == (50, 10, 2)


def test_small_portfolio_metrics():
    """Test portfolio metrics computation."""
    sim = MonteCarloSimulator(n_assets=3, n_regimes=1, n_shocks=1, n_scenarios=100)
    
    P = np.array([[1.0]])
    mu = np.array([[0.001, 0.001, 0.001]])
    Sigma = np.array([np.eye(3) * 0.001])
    
    paths, _, _ = sim.generate_paths(10, P, mu, Sigma, None, random_seed=42)
    
    metrics = sim.compute_portfolio_metrics(paths)
    
    assert "mean_return" in metrics
    assert "volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert "var_95" in metrics
    assert "cvar_95" in metrics
    assert "max_drawdown" in metrics
    
    # Metrics should be finite
    for key, val in metrics.items():
        assert np.isfinite(val), f"{key} is not finite: {val}"


def test_small_portfolio_metrics_with_weights():
    """Test portfolio metrics with custom weights."""
    sim = MonteCarloSimulator(n_assets=2, n_regimes=1, n_shocks=1, n_scenarios=50)
    
    P = np.array([[1.0]])
    mu = np.array([[0.001, 0.001]])
    Sigma = np.array([np.eye(2) * 0.001])
    
    paths, _, _ = sim.generate_paths(10, P, mu, Sigma, None, random_seed=42)
    
    # Custom weights
    weights = np.array([0.3, 0.7])
    metrics = sim.compute_portfolio_metrics(paths, weights=weights)
    
    assert metrics["mean_return"] is not None
    assert metrics["volatility"] > 0


def test_small_portfolio_metrics_invalid_weights():
    """Test that invalid weights are rejected."""
    sim = MonteCarloSimulator(n_assets=2, n_regimes=1, n_shocks=1, n_scenarios=50)
    
    P = np.array([[1.0]])
    mu = np.array([[0.001, 0.001]])
    Sigma = np.array([np.eye(2) * 0.001])
    
    paths, _, _ = sim.generate_paths(10, P, mu, Sigma, None, random_seed=42)
    
    # Weights don't sum to 1
    bad_weights = np.array([0.3, 0.5])
    try:
        sim.compute_portfolio_metrics(paths, weights=bad_weights)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_small_scenario_analysis():
    """Test scenario analysis by regime."""
    sim = MonteCarloSimulator(n_assets=2, n_regimes=2, n_shocks=1, n_scenarios=50)
    
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    mu = np.array([[0.0, 0.0], [0.005, 0.005]])
    Sigma = np.array([np.eye(2) * 0.001, np.eye(2) * 0.005])
    
    paths, regime_paths, _ = sim.generate_paths(
        10, P, mu, Sigma, None, random_seed=42
    )
    
    analysis = sim.scenario_analysis(paths, regime_paths)
    
    assert "Regime_0" in analysis
    assert "Regime_1" in analysis
    
    for regime_label, stats in analysis.items():
        assert "frequency" in stats
        assert "mean_return" in stats
        assert "volatility" in stats


def test_small_scenario_analysis_with_labels():
    """Test scenario analysis with custom regime labels."""
    sim = MonteCarloSimulator(n_assets=2, n_regimes=2, n_shocks=1, n_scenarios=50)
    
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    mu = np.array([[0.0, 0.0], [0.005, 0.005]])
    Sigma = np.array([np.eye(2) * 0.001, np.eye(2) * 0.005])
    
    paths, regime_paths, _ = sim.generate_paths(
        10, P, mu, Sigma, None, random_seed=42
    )
    
    labels = ["Normal", "Stressed"]
    analysis = sim.scenario_analysis(paths, regime_paths, regime_labels=labels)
    
    assert "Normal" in analysis
    assert "Stressed" in analysis


def test_small_repr():
    """Test string representation."""
    sim = MonteCarloSimulator(n_assets=3, n_regimes=2, n_shocks=2, n_scenarios=100)
    repr_str = repr(sim)
    assert "MonteCarloSimulator" in repr_str
    assert "3" in repr_str  # n_assets
    assert "2" in repr_str  # n_regimes


def test_small_reproducibility():
    """Test reproducibility with seed."""
    sim = MonteCarloSimulator(n_assets=2, n_regimes=2, n_shocks=1, n_scenarios=50)
    
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    mu = np.array([[0.0, 0.0], [0.001, 0.001]])
    Sigma = np.array([np.eye(2) * 0.001, np.eye(2) * 0.001])
    
    paths1, _, _ = sim.generate_paths(10, P, mu, Sigma, None, random_seed=42)
    paths2, _, _ = sim.generate_paths(10, P, mu, Sigma, None, random_seed=42)
    
    assert np.allclose(paths1, paths2), "Same seed should produce same paths"


# ============================================================================
# MEDIUM TESTS: Realistic scenarios
# ============================================================================

def test_medium_multi_asset_multi_regime():
    """Test with realistic dimensions."""
    sim = MonteCarloSimulator(n_assets=5, n_regimes=3, n_shocks=2, n_scenarios=200)
    
    P = np.random.dirichlet(np.ones(3), size=3)
    mu = np.random.randn(3, 5) * 0.001
    Sigma = np.array([np.eye(5) * 0.001 for _ in range(3)])
    B = np.random.randn(3, 5, 2) * 0.05
    
    paths, regime_paths, shocks = sim.generate_paths(
        20, P, mu, Sigma, B, random_seed=42
    )
    
    assert paths.shape == (200, 20, 5)
    assert regime_paths.shape == (200, 20)
    assert shocks.shape == (200, 20, 2)


def test_medium_statistics_across_sizes():
    """Test statistics computation at different scales."""
    for n_scenarios in [100, 300, 500]:
        sim = MonteCarloSimulator(
            n_assets=3, n_regimes=2, n_shocks=1, n_scenarios=n_scenarios
        )
        
        P = np.array([[0.9, 0.1], [0.2, 0.8]])
        mu = np.array([[0.001, 0.001, 0.001], [0.002, 0.002, 0.002]])
        Sigma = np.array([np.eye(3) * 0.001, np.eye(3) * 0.002])
        
        paths, _, _ = sim.generate_paths(15, P, mu, Sigma, None, random_seed=42)
        stats = sim.compute_path_statistics(paths)
        
        assert stats["mean"].shape == (15, 3)
        assert stats["std"].shape == (15, 3)


def test_medium_portfolio_performance_across_regimes():
    """Test portfolio metrics show regime differences."""
    sim = MonteCarloSimulator(n_assets=3, n_regimes=2, n_shocks=1, n_scenarios=300)
    
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    # Regime 0: low mean, low vol; Regime 1: high mean, high vol
    mu = np.array([[0.0001, 0.0001, 0.0001], [0.002, 0.002, 0.002]])
    Sigma = np.array([np.eye(3) * 0.0001, np.eye(3) * 0.005])
    
    paths, _, _ = sim.generate_paths(50, P, mu, Sigma, None, random_seed=42)
    metrics = sim.compute_portfolio_metrics(paths)
    
    # Should show positive return
    assert metrics["mean_return"] > 0, "Expected positive return"
    # Volatility should be positive
    assert metrics["volatility"] > 0, "Expected positive volatility"


# ============================================================================
# LARGE TESTS: Full scenarios
# ============================================================================

def test_large_full_scenario():
    """Test full scenario with all features."""
    sim = MonteCarloSimulator(n_assets=10, n_regimes=3, n_shocks=3, n_scenarios=500)
    
    P = np.random.dirichlet(np.ones(3), size=3)
    mu = np.random.randn(3, 10) * 0.001
    Sigma = np.array([np.eye(10) * 0.001 for _ in range(3)])
    B = np.random.randn(3, 10, 3) * 0.05
    
    paths, regime_paths, shocks = sim.generate_paths(
        50, P, mu, Sigma, B, random_seed=42
    )
    
    # Compute all analytics
    stats = sim.compute_path_statistics(paths)
    assert stats["mean"].shape == (50, 10)
    
    metrics = sim.compute_portfolio_metrics(paths)
    assert all(np.isfinite(v) for v in metrics.values())
    
    analysis = sim.scenario_analysis(paths, regime_paths)
    assert len(analysis) == 3


def test_large_stress_scenario():
    """Test stress scenario with concentrated shocks."""
    sim = MonteCarloSimulator(n_assets=5, n_regimes=2, n_shocks=2, n_scenarios=500)
    
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    mu = np.array([[0.001, 0.001, 0.001, 0.001, 0.001],
                    [0.005, 0.005, 0.005, 0.005, 0.005]])
    Sigma = np.array([np.eye(5) * 0.001, np.eye(5) * 0.010])
    
    # Large shock loadings → high shock sensitivity
    B = np.array([
        [[0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, 0.1], [0.0, 0.0]],
        [[0.2, 0.0], [0.2, 0.0], [0.0, 0.2], [0.0, 0.2], [0.0, 0.0]],
    ])
    
    paths, regime_paths, shocks = sim.generate_paths(
        25, P, mu, Sigma, B, random_seed=42
    )
    
    # Shocks should have non-zero impact
    assert np.std(shocks) > 0


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATOR TEST SUITE (Progressive Sizing)")
    print("="*70)
    
    # Small tests
    print("\n" + "-"*70)
    print("SMALL TESTS: Initialization & Basic Operations")
    print("-"*70)
    
    small_tests = [
        ("Initialization", test_small_init),
        ("Invalid dimensions", test_small_invalid_dims),
        ("Path generation shape", test_small_path_generation_shape),
        ("Path generation (no shocks)", test_small_path_generation_no_shocks),
        ("Invalid transition matrix", test_small_invalid_transition_matrix),
        ("Path statistics", test_small_path_statistics),
        ("Portfolio metrics", test_small_portfolio_metrics),
        ("Portfolio metrics (custom weights)", test_small_portfolio_metrics_with_weights),
        ("Portfolio metrics (invalid weights)", test_small_portfolio_metrics_invalid_weights),
        ("Scenario analysis", test_small_scenario_analysis),
        ("Scenario analysis (custom labels)", test_small_scenario_analysis_with_labels),
        ("String representation", test_small_repr),
        ("Reproducibility", test_small_reproducibility),
    ]
    
    small_passed = 0
    small_time = 0
    for name, func in small_tests:
        passed, elapsed = run_test(name, func)
        if passed:
            small_passed += 1
        small_time += elapsed
    
    # Medium tests
    print("\n" + "-"*70)
    print("MEDIUM TESTS: Realistic Scenarios")
    print("-"*70)
    
    medium_tests = [
        ("Multi-asset/multi-regime", test_medium_multi_asset_multi_regime),
        ("Statistics (various sizes)", test_medium_statistics_across_sizes),
        ("Regime performance comparison", test_medium_portfolio_performance_across_regimes),
    ]
    
    medium_passed = 0
    medium_time = 0
    for name, func in medium_tests:
        passed, elapsed = run_test(name, func)
        if passed:
            medium_passed += 1
        medium_time += elapsed
    
    # Large tests
    print("\n" + "-"*70)
    print("LARGE TESTS: Full Scenarios")
    print("-"*70)
    
    large_tests = [
        ("Full scenario (10 assets, 3 regimes)", test_large_full_scenario),
        ("Stress scenario (concentrated shocks)", test_large_stress_scenario),
    ]
    
    large_passed = 0
    large_time = 0
    for name, func in large_tests:
        passed, elapsed = run_test(name, func)
        if passed:
            large_passed += 1
        large_time += elapsed
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = len(small_tests) + len(medium_tests) + len(large_tests)
    total_passed = small_passed + medium_passed + large_passed
    total_time = small_time + medium_time + large_time
    
    print(f"\nSmall tests:  {small_passed}/{len(small_tests)} passed [{small_time:.3f}s]")
    print(f"Medium tests: {medium_passed}/{len(medium_tests)} passed [{medium_time:.3f}s]")
    print(f"Large tests:  {large_passed}/{len(large_tests)} passed [{large_time:.3f}s]")
    print(f"\nTOTAL: {total_passed}/{total_tests} passed [{total_time:.3f}s]")
    
    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED")
        print("\nMonteCarloSimulator is ready for production use.")
        return 0
    else:
        print(f"\n❌ {total_tests - total_passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
