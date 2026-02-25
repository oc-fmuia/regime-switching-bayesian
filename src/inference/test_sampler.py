"""
Tests for NUTS sampler and diagnostics.

Progressive sizing:
- Small (synthetic): Diagnostic computation only (instant)
- Medium: Sampler initialization and parameters (instant)
- Large: Optional sampling on small model (commented out, can uncomment)

All fast tests complete in <1 second.
"""

import sys
import numpy as np
import time

sys.path.insert(0, '/home/fmuia/.openclaw/workspace-fernando/regime-switching-bayesian/src')

from inference.sampler import NUTSSampler, DiagnosticsComputer, PosteriorPredictiveCheck, InferenceSummary


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
# SMALL TESTS: Synthetic diagnostics (no real sampling)
# ============================================================================

def test_small_rhat_perfect_convergence():
    """Test Rhat with perfectly converged chains."""
    # All chains identical → Rhat = 1.0
    chains = np.array([
        np.ones(100),  # Chain 0: all 1.0
        np.ones(100),  # Chain 1: all 1.0
    ])
    
    rhat = DiagnosticsComputer.rhat(chains)
    assert np.isclose(rhat, 1.0, atol=0.01), f"Expected ~1.0, got {rhat}"


def test_small_rhat_poor_convergence():
    """Test Rhat with divergent chains."""
    np.random.seed(42)
    chains = np.array([
        np.random.normal(-5, 1, 100),   # Chain 0: centered at -5
        np.random.normal(5, 1, 100),    # Chain 1: centered at 5
    ])
    
    rhat = DiagnosticsComputer.rhat(chains)
    assert rhat > 1.05, f"Expected >1.05, got {rhat}"


def test_small_rhat_requires_multiple_chains():
    """Test that Rhat requires at least 2 chains."""
    single_chain = np.random.randn(1, 100)
    
    try:
        DiagnosticsComputer.rhat(single_chain)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_small_ess_high_autocorr():
    """Test ESS with high autocorrelation."""
    # Persistent time series → low ESS
    n = 1000
    x = np.cumsum(np.random.randn(n)) / np.sqrt(n)
    
    ess = DiagnosticsComputer.ess(x)
    assert 0 < ess < n, f"ESS should be in (0, {n}), got {ess}"
    # High autocorr → ESS << n
    assert ess < 0.3 * n, f"Expected ESS < {0.3*n}, got {ess}"


def test_small_ess_white_noise():
    """Test ESS with white noise."""
    # Independent samples → ESS ≈ n
    np.random.seed(42)
    x = np.random.randn(1000)
    
    ess = DiagnosticsComputer.ess(x)
    # White noise → ESS should be close to n
    assert ess > 500, f"Expected ESS > 500, got {ess}"


def test_small_ess_constant():
    """Test ESS with constant series."""
    # No variation → ESS = n
    x = np.ones(1000)
    
    ess = DiagnosticsComputer.ess(x)
    assert ess == 1000, f"Expected 1000, got {ess}"


def test_small_nutsampler_init():
    """Test NUTSSampler initialization."""
    sampler = NUTSSampler(target_accept=0.85, max_treedepth=10)
    assert sampler.target_accept == 0.85
    assert sampler.max_treedepth == 10


def test_small_nutsampler_invalid_target_accept():
    """Test that invalid target_accept is rejected."""
    try:
        NUTSSampler(target_accept=0.5)  # Too low
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    try:
        NUTSSampler(target_accept=0.99)  # Too high
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_small_nutsampler_invalid_treedepth():
    """Test that invalid max_treedepth is rejected."""
    try:
        NUTSSampler(max_treedepth=3)  # Too small
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_small_inference_summary_init():
    """Test InferenceSummary initialization."""
    summary = InferenceSummary(
        idata=None,  # Placeholder
        n_draws=1000,
        n_tune=1000,
        n_chains=2,
        sampling_time=30.5,
    )
    
    assert summary.n_draws == 1000
    assert summary.n_chains == 2
    assert summary.total_samples == 2000
    assert summary.sampling_time == 30.5


def test_small_inference_summary_repr():
    """Test InferenceSummary string representation."""
    summary = InferenceSummary(
        idata=None,
        n_draws=1000,
        n_tune=1000,
        n_chains=2,
        sampling_time=15.0,
    )
    
    repr_str = repr(summary)
    assert "InferenceSummary" in repr_str
    assert "1000" in repr_str
    assert "2" in repr_str


def test_small_ppc_empty_variable():
    """Test posterior predictive check with missing variable."""
    # Create mock idata with empty posterior_predictive
    class MockIData:
        def __init__(self):
            self.posterior_predictive = {}
    
    idata = MockIData()
    observed = np.random.randn(100, 3)
    
    result = PosteriorPredictiveCheck.compute_ppcheck(idata, observed, dim_name="returns")
    assert result == {}, f"Expected empty dict, got {result}"


def test_small_diagnostics_computer_repr():
    """Test DiagnosticsComputer (static class)."""
    # Just verify the class exists and methods are callable
    assert hasattr(DiagnosticsComputer, 'rhat')
    assert hasattr(DiagnosticsComputer, 'ess')
    assert hasattr(DiagnosticsComputer, 'divergence_rate')


# ============================================================================
# MEDIUM TESTS: Multi-chain diagnostics
# ============================================================================

def test_medium_rhat_multiple_scenarios():
    """Test Rhat on various convergence scenarios."""
    np.random.seed(42)
    
    # Scenario 1: Good convergence
    good = np.random.normal(0, 1, (4, 1000))
    rhat_good = DiagnosticsComputer.rhat(good)
    assert 0.99 < rhat_good < 1.05, f"Good convergence: {rhat_good}"
    
    # Scenario 2: Bad convergence
    bad = np.vstack([
        np.random.normal(0, 1, (2, 1000)),
        np.random.normal(5, 1, (2, 1000)),
    ])
    rhat_bad = DiagnosticsComputer.rhat(bad)
    assert rhat_bad > 1.2, f"Bad convergence: {rhat_bad}"


def test_medium_ess_multiple_chains():
    """Test ESS computation across multiple chains."""
    np.random.seed(42)
    
    for n in [100, 500, 1000]:
        x = np.random.randn(n)
        ess = DiagnosticsComputer.ess(x)
        
        # ESS should be positive and <= n
        assert 0 < ess <= n, f"ESS for n={n}: {ess}"


def test_medium_sampler_parameter_ranges():
    """Test sampler with various valid parameters."""
    for target_accept in [0.65, 0.80, 0.90, 0.95]:
        for max_depth in [5, 8, 10, 15]:
            sampler = NUTSSampler(
                target_accept=target_accept,
                max_treedepth=max_depth,
            )
            assert sampler.target_accept == target_accept
            assert sampler.max_treedepth == max_depth


def test_medium_inference_summary_calculations():
    """Test InferenceSummary calculations."""
    for n_draws in [100, 500, 1000]:
        for n_chains in [1, 2, 4]:
            summary = InferenceSummary(
                idata=None,
                n_draws=n_draws,
                n_tune=n_draws,
                n_chains=n_chains,
                sampling_time=10.0,
            )
            
            expected_total = n_draws * n_chains
            assert summary.total_samples == expected_total


def test_medium_rhat_stability():
    """Test Rhat stability across repeated calls."""
    np.random.seed(42)
    chains = np.random.normal(0, 1, (3, 1000))
    
    rhat1 = DiagnosticsComputer.rhat(chains)
    rhat2 = DiagnosticsComputer.rhat(chains)
    
    assert np.isclose(rhat1, rhat2), f"Rhat not stable: {rhat1} vs {rhat2}"


def test_medium_ess_range():
    """Test ESS returns sensible values."""
    np.random.seed(42)
    
    # Pure noise → high ESS
    noise = np.random.randn(1000)
    ess_noise = DiagnosticsComputer.ess(noise)
    
    # AR(1) process → lower ESS
    ar1 = np.zeros(1000)
    ar1[0] = np.random.randn()
    for t in range(1, 1000):
        ar1[t] = 0.8 * ar1[t-1] + np.random.randn()
    ess_ar1 = DiagnosticsComputer.ess(ar1)
    
    # Noise should have higher ESS than AR(1)
    assert ess_noise > ess_ar1, f"ESS order: noise={ess_noise}, ar1={ess_ar1}"


# ============================================================================
# LARGE TESTS: Optional sampling (commented by default)
# ============================================================================

# def test_large_nuts_sampling_small_model():
#     """Test actual NUTS sampling on small model (optional, slow)."""
#     import pymc as pm
#     from inference.model_builder import ModelBuilder
#     
#     # Build tiny model
#     mb = ModelBuilder(n_assets=2, n_regimes=2, n_shocks=1, n_obs=20)
#     model = mb.build()
#     
#     # Sample with minimal settings
#     sampler = NUTSSampler()
#     summary = sampler.sample(
#         model,
#         draws=100,    # Tiny
#         tune=100,
#         chains=1,     # Single chain
#         cores=1,
#         random_seed=42,
#         progressbar=False,
#     )
#     
#     assert summary.n_draws == 100
#     assert summary.n_chains == 1
#     assert summary.idata is not None


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("NUTS SAMPLER & DIAGNOSTICS TEST SUITE (Progressive Sizing)")
    print("="*70)
    
    # Small tests
    print("\n" + "-"*70)
    print("SMALL TESTS: Synthetic Diagnostics (no sampling)")
    print("-"*70)
    
    small_tests = [
        ("Rhat: perfect convergence", test_small_rhat_perfect_convergence),
        ("Rhat: poor convergence", test_small_rhat_poor_convergence),
        ("Rhat: requires 2+ chains", test_small_rhat_requires_multiple_chains),
        ("ESS: high autocorr", test_small_ess_high_autocorr),
        ("ESS: white noise", test_small_ess_white_noise),
        ("ESS: constant series", test_small_ess_constant),
        ("NUTSSampler init", test_small_nutsampler_init),
        ("NUTSSampler: invalid target_accept", test_small_nutsampler_invalid_target_accept),
        ("NUTSSampler: invalid treedepth", test_small_nutsampler_invalid_treedepth),
        ("InferenceSummary init", test_small_inference_summary_init),
        ("InferenceSummary repr", test_small_inference_summary_repr),
        ("PPC: empty variable", test_small_ppc_empty_variable),
        ("DiagnosticsComputer", test_small_diagnostics_computer_repr),
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
    print("MEDIUM TESTS: Multi-chain Diagnostics")
    print("-"*70)
    
    medium_tests = [
        ("Rhat: multiple scenarios", test_medium_rhat_multiple_scenarios),
        ("ESS: multiple chains", test_medium_ess_multiple_chains),
        ("Sampler: parameter ranges", test_medium_sampler_parameter_ranges),
        ("InferenceSummary: calculations", test_medium_inference_summary_calculations),
        ("Rhat: stability", test_medium_rhat_stability),
        ("ESS: sensible range", test_medium_ess_range),
    ]
    
    medium_passed = 0
    medium_time = 0
    for name, func in medium_tests:
        passed, elapsed = run_test(name, func)
        if passed:
            medium_passed += 1
        medium_time += elapsed
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = len(small_tests) + len(medium_tests)
    total_passed = small_passed + medium_passed
    total_time = small_time + medium_time
    
    print(f"\nSmall tests:  {small_passed}/{len(small_tests)} passed [{small_time:.3f}s]")
    print(f"Medium tests: {medium_passed}/{len(medium_tests)} passed [{medium_time:.3f}s]")
    print(f"\nTOTAL: {total_passed}/{total_tests} passed [{total_time:.3f}s]")
    
    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED")
        print("\nNotes:")
        print("- Diagnostic computation working correctly")
        print("- Rhat and ESS calculations verified")
        print("- NUTS sampler initialized and configured")
        print("- All tests complete in <1 second (no expensive sampling)")
        print("\nOptional: Uncomment test_large_nuts_sampling_small_model()")
        print("  for actual MCMC sampling test (~5-30 seconds)")
        return 0
    else:
        print(f"\n❌ {total_tests - total_passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
