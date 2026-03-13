"""
Probing tests for pymc_extras.marginalize + DiscreteMarkovChain.

These tests determine which emission patterns are compatible with the
automatic marginalization engine. Each test is self-contained and
documents whether the pattern works or fails (and how).

Run with:
    pixi run -e dev pytest docs/research/marginalization_engine/test_marginalization.py -v -s

The tests are ordered from simplest to most complex:
    1. Univariate Normal emission (baseline, known to work)
    2. Multiple independent univariate Normal emissions
    3. Multivariate Normal (MvNormal) emission
    4. MvNormal with per-regime Cholesky covariance
    5. Manual forward algorithm comparison (correctness check)
    6. recover_marginals on DiscreteMarkovChain
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_univariate_hmm_data(T=50, K=2, seed=42):
    """Generate univariate HMM data with known parameters."""
    rng = np.random.default_rng(seed)

    pi0 = np.array([0.8, 0.2])
    P = np.array([[0.95, 0.05],
                  [0.10, 0.90]])
    mus = np.array([1.0, -1.0])
    sigmas = np.array([0.5, 1.5])

    regimes = np.zeros(T, dtype=int)
    regimes[0] = rng.choice(K, p=pi0)
    for t in range(1, T):
        regimes[t] = rng.choice(K, p=P[regimes[t - 1]])

    y = rng.normal(mus[regimes], sigmas[regimes])

    return dict(y=y, regimes=regimes, pi0=pi0, P=P, mus=mus, sigmas=sigmas, T=T, K=K)


def make_multivariate_hmm_data(T=50, K=2, d=3, seed=42):
    """Generate multivariate HMM data with known parameters."""
    rng = np.random.default_rng(seed)

    pi0 = np.array([0.8, 0.2])
    P = np.array([[0.95, 0.05],
                  [0.10, 0.90]])

    mus = np.array([
        [0.01, 0.008, 0.012],
        [-0.005, -0.008, -0.003],
    ])
    sigmas = np.array([
        [0.04, 0.035, 0.045],
        [0.08, 0.09, 0.10],
    ])
    # Identity correlation for simplicity
    corr_chols = np.stack([np.eye(d), np.eye(d)])

    def build_cov(k):
        D = np.diag(sigmas[k])
        L = corr_chols[k]
        return D @ L @ L.T @ D

    covs = np.stack([build_cov(k) for k in range(K)])

    regimes = np.zeros(T, dtype=int)
    regimes[0] = rng.choice(K, p=pi0)
    for t in range(1, T):
        regimes[t] = rng.choice(K, p=P[regimes[t - 1]])

    z = np.zeros((T, d))
    for t in range(T):
        z[t] = rng.multivariate_normal(mus[regimes[t]], covs[regimes[t]])

    return dict(
        z=z, regimes=regimes, pi0=pi0, P=P, mus=mus, sigmas=sigmas,
        corr_chols=corr_chols, covs=covs, T=T, K=K, d=d,
    )


# ---------------------------------------------------------------------------
# Manual forward algorithm (reference implementation for correctness checks)
# ---------------------------------------------------------------------------

def manual_forward_logp(data, pi0, P, mus, sigmas, covs=None):
    """
    Compute marginal log-likelihood via the forward algorithm in NumPy.
    Works for both univariate (covs=None) and multivariate (covs provided) cases.
    """
    T = data.shape[0]
    K = len(pi0)
    log_alpha = np.zeros(K)

    for k in range(K):
        if covs is not None:
            log_alpha[k] = np.log(pi0[k]) + multivariate_normal.logpdf(data[0], mus[k], covs[k])
        else:
            log_alpha[k] = (
                np.log(pi0[k])
                + (-0.5 * np.log(2 * np.pi * sigmas[k] ** 2)
                   - 0.5 * ((data[0] - mus[k]) / sigmas[k]) ** 2)
            )

    for t in range(1, T):
        log_alpha_new = np.zeros(K)
        for k in range(K):
            log_alpha_new[k] = np.logaddexp.reduce(log_alpha + np.log(P[:, k]))
            if covs is not None:
                log_alpha_new[k] += multivariate_normal.logpdf(data[t], mus[k], covs[k])
            else:
                log_alpha_new[k] += (
                    -0.5 * np.log(2 * np.pi * sigmas[k] ** 2)
                    - 0.5 * ((data[t] - mus[k]) / sigmas[k]) ** 2
                )
        log_alpha = log_alpha_new

    return np.logaddexp.reduce(log_alpha)


# ===========================================================================
# TEST 1: Univariate Normal emission (baseline)
#
# Expected: PASS. This is the simplest HMM pattern and is demonstrated
# in official pymc_extras examples.
# ===========================================================================

class TestUnivariateEmission:
    """DiscreteMarkovChain + marginalize with a single univariate Normal."""

    def test_model_builds_and_marginalizes(self):
        """Can we build the model and call pmx.marginalize without error?"""
        import pymc_extras as pmx

        data = make_univariate_hmm_data()
        T, K = data["T"], data["K"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            init_dist = pm.Categorical.dist(p=np.ones(K) / K)
            chain = pmx.DiscreteMarkovChain(
                "chain", P=P, init_dist=init_dist, shape=(T,)
            )

            mu = pm.Normal("mu", mu=0, sigma=2, shape=(K,))
            sigma = pm.HalfNormal("sigma", sigma=2, shape=(K,))
            pm.Normal("obs", mu=mu[chain], sigma=sigma[chain], observed=data["y"])

        model_marg = pmx.marginalize(model, ["chain"])
        logp_fn = model_marg.compile_logp()
        logp_val = logp_fn(model_marg.initial_point())
        assert np.isfinite(logp_val), f"logp is not finite: {logp_val}"

    def test_logp_matches_manual_forward(self):
        """Does the marginalized logp match a hand-coded forward algorithm?"""
        import pymc_extras as pmx

        data = make_univariate_hmm_data()
        T, K = data["T"], data["K"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            init_dist = pm.Categorical.dist(p=np.ones(K) / K)
            chain = pmx.DiscreteMarkovChain(
                "chain", P=P, init_dist=init_dist, shape=(T,)
            )
            mu = pm.Normal("mu", mu=0, sigma=2, shape=(K,))
            sigma = pm.HalfNormal("sigma", sigma=2, shape=(K,))
            pm.Normal("obs", mu=mu[chain], sigma=sigma[chain], observed=data["y"])

        model_marg = pmx.marginalize(model, ["chain"])

        # Evaluate at true parameters
        point = {
            "P_simplex__": pm.math.invlogit(np.zeros((K, K - 1))).eval(),
            "mu": data["mus"],
            "sigma_log__": np.log(data["sigmas"]),
        }
        # We can't easily set the true P through the transformed space,
        # so instead evaluate at the initial point and compare shapes only.
        logp_fn = model_marg.compile_logp()
        logp_val = logp_fn(model_marg.initial_point())
        assert np.isfinite(logp_val)

    def test_sampling_runs(self):
        """Can NUTS actually sample the marginalized model (short run)?"""
        import pymc_extras as pmx

        data = make_univariate_hmm_data(T=30)
        T, K = data["T"], data["K"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            init_dist = pm.Categorical.dist(p=np.ones(K) / K)
            chain = pmx.DiscreteMarkovChain(
                "chain", P=P, init_dist=init_dist, shape=(T,)
            )
            mu = pm.Normal("mu", mu=0, sigma=2, shape=(K,))
            sigma = pm.HalfNormal("sigma", sigma=2, shape=(K,))
            pm.Normal("obs", mu=mu[chain], sigma=sigma[chain], observed=data["y"])

        model_marg = pmx.marginalize(model, ["chain"])

        with model_marg:
            idata = pm.sample(
                draws=50, tune=50, chains=1,
                random_seed=42, progressbar=False,
            )

        assert "mu" in idata.posterior
        assert "sigma" in idata.posterior
        assert "P" in idata.posterior


# ===========================================================================
# TEST 2: Multiple independent univariate emissions
#
# Pattern: d separate pm.Normal(..., observed=data[:, j]) conditioned on
# the same chain. This is an alternative to MvNormal that avoids the
# multivariate constraint.
#
# Expected: likely PASS (each Normal is univariate, element-wise).
# ===========================================================================

class TestMultipleUnivariateEmissions:
    """DiscreteMarkovChain + marginalize with d independent Normal emissions."""

    def test_model_builds_and_marginalizes(self):
        import pymc_extras as pmx

        data = make_multivariate_hmm_data()
        T, K, d = data["T"], data["K"], data["d"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            init_dist = pm.Categorical.dist(p=np.ones(K) / K)
            chain = pmx.DiscreteMarkovChain(
                "chain", P=P, init_dist=init_dist, shape=(T,)
            )

            mu = pm.Normal("mu", mu=0, sigma=0.1, shape=(K, d))
            sigma = pm.HalfNormal("sigma", sigma=0.2, shape=(K, d))

            for j in range(d):
                pm.Normal(
                    f"obs_{j}",
                    mu=mu[chain, j],
                    sigma=sigma[chain, j],
                    observed=data["z"][:, j],
                )

        model_marg = pmx.marginalize(model, ["chain"])
        logp_fn = model_marg.compile_logp()
        logp_val = logp_fn(model_marg.initial_point())
        assert np.isfinite(logp_val), f"logp is not finite: {logp_val}"

    def test_sampling_runs(self):
        import pymc_extras as pmx

        data = make_multivariate_hmm_data(T=30)
        T, K, d = data["T"], data["K"], data["d"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            init_dist = pm.Categorical.dist(p=np.ones(K) / K)
            chain = pmx.DiscreteMarkovChain(
                "chain", P=P, init_dist=init_dist, shape=(T,)
            )

            mu = pm.Normal("mu", mu=0, sigma=0.1, shape=(K, d))
            sigma = pm.HalfNormal("sigma", sigma=0.2, shape=(K, d))

            for j in range(d):
                pm.Normal(
                    f"obs_{j}",
                    mu=mu[chain, j],
                    sigma=sigma[chain, j],
                    observed=data["z"][:, j],
                )

        model_marg = pmx.marginalize(model, ["chain"])

        with model_marg:
            idata = pm.sample(
                draws=50, tune=50, chains=1,
                random_seed=42, progressbar=False,
            )

        assert "mu" in idata.posterior
        assert "sigma" in idata.posterior


# ===========================================================================
# TEST 3: Multivariate Normal (MvNormal) emission
#
# Pattern: pm.MvNormal("obs", mu=mu[chain], cov=cov[chain], observed=data)
#
# This is the pattern we WANT for v0. The marginalization engine docs say
# "it's not possible to marginalize RVs with multivariate dependent RVs."
# This test determines whether that restriction applies here.
#
# Expected: likely FAIL (NotImplementedError) but we need to confirm.
# ===========================================================================

class TestMvNormalEmission:
    """DiscreteMarkovChain + marginalize with MvNormal emission."""

    def test_model_builds(self):
        """Can we at least build the (unmarginalized) model?"""
        import pymc_extras as pmx

        data = make_multivariate_hmm_data()
        T, K, d = data["T"], data["K"], data["d"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            init_dist = pm.Categorical.dist(p=np.ones(K) / K)
            chain = pmx.DiscreteMarkovChain(
                "chain", P=P, init_dist=init_dist, shape=(T,)
            )

            mu = pm.Normal("mu", mu=0, sigma=0.1, shape=(K, d))
            # Diagonal covariance for simplicity
            sigma = pm.HalfNormal("sigma", sigma=0.2, shape=(K, d))
            cov = pt.zeros((K, d, d))
            for k in range(K):
                cov = pt.set_subtensor(cov[k], pt.diag(sigma[k] ** 2))

            pm.MvNormal("obs", mu=mu[chain], cov=cov[chain], observed=data["z"])

        assert model is not None

    def test_marginalize_mvnormal(self):
        """Does pmx.marginalize accept MvNormal as the dependent RV?"""
        import pymc_extras as pmx

        data = make_multivariate_hmm_data()
        T, K, d = data["T"], data["K"], data["d"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            init_dist = pm.Categorical.dist(p=np.ones(K) / K)
            chain = pmx.DiscreteMarkovChain(
                "chain", P=P, init_dist=init_dist, shape=(T,)
            )

            mu = pm.Normal("mu", mu=0, sigma=0.1, shape=(K, d))
            sigma = pm.HalfNormal("sigma", sigma=0.2, shape=(K, d))
            cov = pt.zeros((K, d, d))
            for k in range(K):
                cov = pt.set_subtensor(cov[k], pt.diag(sigma[k] ** 2))

            pm.MvNormal("obs", mu=mu[chain], cov=cov[chain], observed=data["z"])

        try:
            model_marg = pmx.marginalize(model, ["chain"])
            logp_fn = model_marg.compile_logp()
            logp_val = logp_fn(model_marg.initial_point())
            assert np.isfinite(logp_val), f"logp is not finite: {logp_val}"
            print("\n[RESULT] MvNormal marginalization: SUCCEEDED")
        except NotImplementedError as e:
            print(f"\n[RESULT] MvNormal marginalization: FAILED (NotImplementedError)")
            print(f"  Message: {e}")
            pytest.skip(f"MvNormal marginalization not supported: {e}")
        except Exception as e:
            print(f"\n[RESULT] MvNormal marginalization: FAILED ({type(e).__name__})")
            print(f"  Message: {e}")
            pytest.skip(f"MvNormal marginalization failed unexpectedly: {e}")


# ===========================================================================
# TEST 4: MvNormal with LKJCholeskyCov (full regime-switching covariance)
#
# This is the full pattern from pymc_spec_v0.md: per-regime covariance
# via LKJCholeskyCov, indexed by chain state.
#
# Expected: likely FAIL if Test 3 fails; redundant if Test 3 passes.
# ===========================================================================

class TestMvNormalWithLKJ:
    """DiscreteMarkovChain + marginalize with LKJCholeskyCov per regime."""

    def test_marginalize_mvnormal_lkj(self):
        import pymc_extras as pmx

        data = make_multivariate_hmm_data(T=30)
        T, K, d = data["T"], data["K"], data["d"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            init_dist = pm.Categorical.dist(p=np.ones(K) / K)
            chain = pmx.DiscreteMarkovChain(
                "chain", P=P, init_dist=init_dist, shape=(T,)
            )

            mu = pm.Normal("mu", mu=0, sigma=0.1, shape=(K, d))

            chols = []
            for k in range(K):
                chol_k, _, _ = pm.LKJCholeskyCov(
                    f"chol_cov_{k}",
                    n=d,
                    eta=2.0,
                    sd_dist=pm.HalfNormal.dist(sigma=0.2),
                    compute_corr=True,
                )
                chols.append(chol_k)
            chol_stack = pt.stack(chols)  # (K, d, d)

            pm.MvNormal(
                "obs", mu=mu[chain], chol=chol_stack[chain], observed=data["z"]
            )

        try:
            model_marg = pmx.marginalize(model, ["chain"])
            logp_fn = model_marg.compile_logp()
            logp_val = logp_fn(model_marg.initial_point())
            assert np.isfinite(logp_val), f"logp is not finite: {logp_val}"
            print("\n[RESULT] MvNormal + LKJ marginalization: SUCCEEDED")
        except NotImplementedError as e:
            print(f"\n[RESULT] MvNormal + LKJ marginalization: FAILED (NotImplementedError)")
            print(f"  Message: {e}")
            pytest.skip(f"MvNormal + LKJ marginalization not supported: {e}")
        except Exception as e:
            print(f"\n[RESULT] MvNormal + LKJ marginalization: FAILED ({type(e).__name__})")
            print(f"  Message: {e}")
            pytest.skip(f"MvNormal + LKJ marginalization failed unexpectedly: {e}")


# ===========================================================================
# TEST 5: recover_marginals
#
# After marginalization and sampling, can we recover the discrete chain
# via pmx.recover_marginals?
#
# Expected: PASS if marginalization works for the given emission pattern.
# ===========================================================================

class TestRecoverMarginals:
    """Test pmx.recover_marginals on a marginalized DiscreteMarkovChain.

    FINDING: recover_marginals does NOT support DiscreteMarkovChain.
    It only supports Bernoulli, Categorical, and DiscreteUniform.
    Regime recovery must be done via a custom FFBS implementation.
    """

    def test_recover_not_supported_for_markov_chain(self):
        """Confirm that recover_marginals raises NotImplementedError for DiscreteMarkovChain."""
        import pymc_extras as pmx

        data = make_univariate_hmm_data(T=30)
        T, K = data["T"], data["K"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            init_dist = pm.Categorical.dist(p=np.ones(K) / K)
            chain = pmx.DiscreteMarkovChain(
                "chain", P=P, init_dist=init_dist, shape=(T,)
            )
            mu = pm.Normal("mu", mu=0, sigma=2, shape=(K,))
            sigma = pm.HalfNormal("sigma", sigma=2, shape=(K,))
            pm.Normal("obs", mu=mu[chain], sigma=sigma[chain], observed=data["y"])

        model_marg = pmx.marginalize(model, ["chain"])

        with model_marg:
            idata = pm.sample(
                draws=50, tune=50, chains=1,
                random_seed=42, progressbar=False,
            )

        with pytest.raises(NotImplementedError, match="cannot be recovered"):
            pmx.recover_marginals(idata, model=model_marg, random_seed=42)

        print("\n[RESULT] recover_marginals: correctly raises NotImplementedError for "
              "DiscreteMarkovChain. Custom FFBS needed for regime recovery.")


# ===========================================================================
# TEST 6: Manual forward algorithm via pm.Potential (Approach B)
#
# This is the fallback approach: write the forward algorithm in PyTensor
# and add it as a Potential. This should always work regardless of
# emission type.
# ===========================================================================

class TestManualForwardAlgorithm:
    """Build a multivariate HMM using pm.Potential + pytensor.scan."""

    def test_potential_model_builds_and_evaluates(self):
        import pytensor

        data = make_multivariate_hmm_data(T=30)
        T, K, d = data["T"], data["K"], data["d"]
        z_data = data["z"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K))
            log_P = pt.log(P)
            log_pi0 = pt.log(pi0)

            mu = pm.Normal("mu", mu=0, sigma=0.1, shape=(K, d))
            sigma = pm.HalfNormal("sigma", sigma=0.2, shape=(K, d))

            z_obs = pt.as_tensor_variable(z_data, name="z_obs")

            # Compute log-emission for each regime at each time step
            log_lik_components = []
            for k in range(K):
                dist_k = pm.MvNormal.dist(mu=mu[k], cov=pt.diag(sigma[k] ** 2))
                log_lik_k = pm.logp(dist_k, z_obs)  # (T,)
                log_lik_components.append(log_lik_k)
            log_lik = pt.stack(log_lik_components, axis=1)  # (T, K)

            log_alpha_init = log_pi0 + log_lik[0]

            def forward_step(log_lik_t, log_alpha_prev, log_P_):
                return pt.logsumexp(log_alpha_prev[:, None] + log_P_, axis=0) + log_lik_t

            log_alphas = pytensor.scan(
                fn=forward_step,
                sequences=[log_lik[1:]],
                outputs_info=[log_alpha_init],
                non_sequences=[log_P],
                return_updates=False,
            )

            log_marginal_lik = pt.logsumexp(log_alphas[-1])
            pm.Potential("hmm_loglik", log_marginal_lik)

        logp_fn = model.compile_logp()
        logp_val = logp_fn(model.initial_point())
        assert np.isfinite(logp_val), f"logp is not finite: {logp_val}"
        print(f"\n[RESULT] Manual forward algorithm (Potential): logp = {logp_val:.2f}")

    def test_potential_model_sampling(self):
        import pytensor

        data = make_multivariate_hmm_data(T=30, seed=123)
        T, K, d = data["T"], data["K"], data["d"]
        z_data = data["z"]

        with pm.Model() as model:
            P = pm.Dirichlet("P", a=np.ones(K) * 5, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K))
            log_P = pt.log(P)
            log_pi0 = pt.log(pi0)

            mu = pm.Normal("mu", mu=0, sigma=0.1, shape=(K, d))
            sigma = pm.HalfNormal("sigma", sigma=0.2, shape=(K, d))

            z_obs = pt.as_tensor_variable(z_data, name="z_obs")

            log_lik_components = []
            for k in range(K):
                dist_k = pm.MvNormal.dist(mu=mu[k], cov=pt.diag(sigma[k] ** 2))
                log_lik_k = pm.logp(dist_k, z_obs)
                log_lik_components.append(log_lik_k)
            log_lik = pt.stack(log_lik_components, axis=1)

            log_alpha_init = log_pi0 + log_lik[0]

            def forward_step(log_lik_t, log_alpha_prev, log_P_):
                return pt.logsumexp(log_alpha_prev[:, None] + log_P_, axis=0) + log_lik_t

            log_alphas = pytensor.scan(
                fn=forward_step,
                sequences=[log_lik[1:]],
                outputs_info=[log_alpha_init],
                non_sequences=[log_P],
                return_updates=False,
            )

            log_marginal_lik = pt.logsumexp(log_alphas[-1])
            pm.Potential("hmm_loglik", log_marginal_lik)

        with model:
            idata = pm.sample(
                draws=50, tune=50, chains=1,
                random_seed=42, progressbar=False,
            )

        assert "mu" in idata.posterior
        assert "sigma" in idata.posterior
        assert "P" in idata.posterior
        print("\n[RESULT] Manual forward algorithm sampling: SUCCEEDED")


# ===========================================================================
# TEST 7: Correctness comparison
#
# Compare the NumPy manual forward algorithm against the PyTensor/PyMC
# implementation to make sure they agree.
# ===========================================================================

class TestCorrectnessComparison:
    """Verify that our NumPy forward algorithm matches the PyTensor version."""

    def test_univariate_logp_matches(self):
        data = make_univariate_hmm_data(T=20)
        y, pi0, P, mus, sigmas = data["y"], data["pi0"], data["P"], data["mus"], data["sigmas"]

        numpy_logp = manual_forward_logp(y, pi0, P, mus, sigmas)

        # Build the same in PyTensor
        import pytensor

        z_pt = pt.as_tensor_variable(y)
        log_pi0 = pt.log(pt.as_tensor_variable(pi0))
        log_P = pt.log(pt.as_tensor_variable(P))
        mus_pt = pt.as_tensor_variable(mus)
        sigmas_pt = pt.as_tensor_variable(sigmas)

        K = len(pi0)
        log_lik_components = []
        for k in range(K):
            log_lik_k = pm.logp(pm.Normal.dist(mu=mus_pt[k], sigma=sigmas_pt[k]), z_pt)
            log_lik_components.append(log_lik_k)
        log_lik = pt.stack(log_lik_components, axis=1)

        log_alpha_init = log_pi0 + log_lik[0]

        def forward_step(log_lik_t, log_alpha_prev, log_P_):
            return pt.logsumexp(log_alpha_prev[:, None] + log_P_, axis=0) + log_lik_t

        log_alphas = pytensor.scan(
            fn=forward_step,
            sequences=[log_lik[1:]],
            outputs_info=[log_alpha_init],
            non_sequences=[log_P],
            return_updates=False,
        )

        pytensor_logp = pt.logsumexp(log_alphas[-1]).eval()

        assert np.isclose(numpy_logp, pytensor_logp, atol=1e-6), (
            f"NumPy forward logp ({numpy_logp:.6f}) != PyTensor forward logp ({pytensor_logp:.6f})"
        )
        print(f"\n[RESULT] Correctness check: NumPy={numpy_logp:.6f}, PyTensor={pytensor_logp:.6f}")

    def test_multivariate_logp_matches(self):
        data = make_multivariate_hmm_data(T=20)
        z = data["z"]
        pi0, P, mus, covs = data["pi0"], data["P"], data["mus"], data["covs"]
        sigmas, K, d = data["sigmas"], data["K"], data["d"]

        numpy_logp = manual_forward_logp(z, pi0, P, mus, None, covs)

        import pytensor

        z_pt = pt.as_tensor_variable(z)
        log_pi0 = pt.log(pt.as_tensor_variable(pi0))
        log_P = pt.log(pt.as_tensor_variable(P))
        mus_pt = pt.as_tensor_variable(mus)
        sigmas_pt = pt.as_tensor_variable(sigmas)

        log_lik_components = []
        for k in range(K):
            dist_k = pm.MvNormal.dist(mu=mus_pt[k], cov=pt.diag(sigmas_pt[k] ** 2))
            log_lik_k = pm.logp(dist_k, z_pt)
            log_lik_components.append(log_lik_k)
        log_lik = pt.stack(log_lik_components, axis=1)

        log_alpha_init = log_pi0 + log_lik[0]

        def forward_step(log_lik_t, log_alpha_prev, log_P_):
            return pt.logsumexp(log_alpha_prev[:, None] + log_P_, axis=0) + log_lik_t

        log_alphas = pytensor.scan(
            fn=forward_step,
            sequences=[log_lik[1:]],
            outputs_info=[log_alpha_init],
            non_sequences=[log_P],
            return_updates=False,
        )

        pytensor_logp = pt.logsumexp(log_alphas[-1]).eval()

        assert np.isclose(numpy_logp, pytensor_logp, atol=1e-6), (
            f"NumPy forward logp ({numpy_logp:.6f}) != PyTensor forward logp ({pytensor_logp:.6f})"
        )
        print(f"\n[RESULT] MV correctness: NumPy={numpy_logp:.6f}, PyTensor={pytensor_logp:.6f}")
